import os
import math
import argparse
import pickle as pkl

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import lib.datasets as datasets
import lib.models as models
from lib.core.config import config, update_config
from lib.core.engine import Engine
from lib.core.utils import AverageMeter
from lib.core import eval
import lib.models.loss as loss
import math

torch.manual_seed(3)
torch.cuda.manual_seed(3)

torch.set_printoptions(precision=2, sci_mode=False)

def parse_args():
    parser = argparse.ArgumentParser(description='Test localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # testing
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--split', default='val', required=True, type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.OUTPUT_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose

def save_scores(scores, data, dataset_name, split):
    results = {}
    for i, d in enumerate(data):
        results[d['video']] = scores[i]
    pkl.dump(results,open(os.path.join(config.RESULT_DIR, dataset_name, '{}_{}_{}.pkl'.format(config.MODEL.NAME,config.DATASET.VIS_INPUT_TYPE,
        split)),'wb'))

if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)

    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = getattr(models, config.MODEL.NAME)()

    model_path = config.MODEL.CHECKPOINT
    if args.split == 'test_iid':
        # model_path = model_path + 'best_iid.pkl'
        model_path = os.path.join(model_path, 'best_iid.pkl')
    elif args.split == 'test_ood':
        # model_path = model_path + 'best_ood.pkl'
        model_path = os.path.join(model_path, 'best_ood.pkl')
    else:
        raise NotImplementedError


    model_checkpoint = torch.load(model_path)
    model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model  = model.to(device)
    model.eval()

    dataset_name = config.DATASET.NAME
    if not config.DATASET.NO_IID:
        test_iid_dataset = getattr(datasets, dataset_name)('test_iid')
    if not config.DATASET.NO_OOD:
        test_ood_dataset = getattr(datasets, dataset_name)('test_ood')

    def iterator(split):
        if split == 'test_iid':
            dataloader = DataLoader(test_iid_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        elif split == 'test_ood':
            dataloader = DataLoader(test_ood_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.collate_fn)
        else:
            raise NotImplementedError

        return dataloader


    def network(sample):
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']
        word_label = sample['batch_word_label'].cuda()
        word_mask = sample['batch_word_mask'].cuda()
        gt_times = sample['batch_gt_times'].cuda()

        logits_text, logits_visual, logits_iou, iou_mask_map, visual_content_emb, text_query_emb, corss_similarity, \
             aug_logits_text, aug_logits_visual, aug_logits_iou, aug_iou_mask_map, aug_visual_content_emb, aug_text_query_emb, aug_corss_similarity, neg_iou = model(textual_input, textual_mask, word_mask, visual_input, visual_input)

        loss_value, joint_prob, iou_scores, regress = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, logits_text, logits_visual, logits_iou, iou_mask_map, map_gt, gt_times, word_label, word_mask, visual_content_emb, text_query_emb, corss_similarity)
        loss_value2, _, iou_scores2, _ = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, aug_logits_text, aug_logits_visual, aug_logits_iou, aug_iou_mask_map, map_gt, gt_times, word_label, word_mask, aug_visual_content_emb, aug_text_query_emb, aug_corss_similarity)
        iou_scores = iou_scores + config.TEST.WEIGHT * iou_scores2


        sorted_times = None if model.training else get_proposal_results(iou_scores, regress, duration)
        # pdb.set_trace()

        return loss_value, sorted_times

    def get_proposal_results(scores, regress, durations):
        # assume all valid scores are larger than one
        out_sorted_times = []


        T = scores.shape[-1]

        regress = regress.cpu().detach().numpy()

        for score, reg, duration in zip(scores, regress, durations):
            # pdb.set_trace()
            sorted_indexs = np.dstack(np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([ [reg[0,item[0],item[1]], reg[1,item[0],item[1]]] for item in sorted_indexs[0] if reg[0,item[0],item[1]] < reg[1,item[0],item[1]] ]).astype(float)
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())


        return out_sorted_times


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'test_iid':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_iid_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'test_ood':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_ood_dataset)/config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError

    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        min_idx = min(state['sample']['batch_anno_idxs'])
        batch_indexs = [idx - min_idx for idx in state['sample']['batch_anno_idxs']]
        sorted_segments = [state['output'][i] for i in batch_indexs]
        state['sorted_segments_list'].extend(sorted_segments)

    def on_test_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()
            print()

        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations, config.TEST.EVAL_METRIC ,verbose=False)
        if config.VERBOSE:
            state['progress_bar'].close()


        val_table, score_comp = eval.display_results(state['Rank@N,mIoU@M'], state['miou'],
                                    'performance on {} set'.format(args.split), config.TEST.BEST_METRIC)
        print(val_table)


    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.test(network, iterator(args.split), args.split) # test_iid, test_ood