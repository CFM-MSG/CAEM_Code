from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
import os
import time
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import lib.datasets as datasets
import lib.models as models
from lib.core.config import config, update_config
from lib.core.engine import Engine
from lib.core.utils import AverageMeter
from lib.core import eval
from lib.core.utils import create_logger
import lib.models.loss as loss
import math

import pdb

torch.manual_seed(3)
torch.cuda.manual_seed(3)
torch.autograd.set_detect_anomaly(True)
def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--no-log', default=False, action="store_true", help='enable logger')
    # parser.add_argument('--weight', default=0.0, required=True, type=float)
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
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag

    config.NO_LOG = args.no_log


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n'+pprint.pformat(config))

    local_time = time.strftime('%m_%d_%H_%M', time.localtime())
    print("local_time: ", local_time)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    train_dataset = getattr(datasets, dataset_name)('train')
    # pdb.set_trace()
    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_IID:
        test_iid_dataset = getattr(datasets, dataset_name)('test_iid')
    if not config.DATASET.NO_OOD:
        test_ood_dataset = getattr(datasets, dataset_name)('test_ood')

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(),lr=config.TRAIN.LR, betas=(0.9, 0.999), weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR, patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)

    best_score_iid = 0
    best_score_ood = 0
    best_table_iid = ""
    best_table_ood = ""

    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.train_collate_fn)
        elif split == 'test_iid':
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
        anno_idxs = sample['batch_anno_idxs']
        textual_input = sample['batch_word_vectors'].cuda()
        textual_mask = sample['batch_txt_mask'].cuda()
        visual_input = sample['batch_vis_input'].cuda()
        map_gt = sample['batch_map_gt'].cuda()
        duration = sample['batch_duration']
        word_label = sample['batch_word_label'].cuda()
        word_mask = sample['batch_word_mask'].cuda()
        gt_times = sample['batch_gt_times'].cuda()

        if not model.training:

            logits_text, logits_visual, logits_iou, iou_mask_map, visual_content_emb, text_query_emb, corss_similarity, \
                aug_logits_text, aug_logits_visual, aug_logits_iou, aug_iou_mask_map, aug_visual_content_emb, aug_text_query_emb, aug_corss_similarity = model(textual_input, textual_mask, word_mask, visual_input, visual_input)

            loss_value, joint_prob, iou_scores, regress = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, logits_text, logits_visual, logits_iou, iou_mask_map, map_gt, gt_times, word_label, word_mask, visual_content_emb, text_query_emb, corss_similarity)
            loss_value2, _, iou_scores2, _ = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, aug_logits_text, aug_logits_visual, aug_logits_iou, aug_iou_mask_map, map_gt, gt_times, word_label, word_mask, aug_visual_content_emb, aug_text_query_emb, aug_corss_similarity)
            iou_scores = iou_scores + config.TRAIN.WEIGHT * iou_scores2

            sorted_times = None if model.training else get_proposal_results(iou_scores, regress, duration)
            return loss_value, sorted_times
        else:
            flag = config.TRAIN.CTF
            aug_gt_times = sample['aug_batch_gt_times'].cuda()
            aug_map_gt = sample['aug_batch_map_gt'].cuda()
            aug_visual_input = sample['aug_batch_vis_input'].cuda()

            logits_text, logits_visual, logits_iou, iou_mask_map, visual_content_emb, text_query_emb, corss_similarity, \
             aug_logits_text, aug_logits_visual, aug_logits_iou, aug_iou_mask_map, aug_visual_content_emb, aug_text_query_emb, aug_corss_similarity = model(textual_input, textual_mask, word_mask, visual_input, aug_visual_input)
            loss_value1, _, _, _ = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, logits_text, logits_visual, logits_iou, iou_mask_map, map_gt, gt_times, word_label, word_mask, visual_content_emb, text_query_emb, corss_similarity, aug_logits_iou, aug_gt_times, aug_visual_content_emb)
            loss_value2, _, _, _ = getattr(loss, config.LOSS.NAME)(config.LOSS.PARAMS, aug_logits_text, aug_logits_visual, aug_logits_iou, aug_iou_mask_map, aug_map_gt, aug_gt_times, word_label, word_mask, aug_visual_content_emb, aug_text_query_emb, aug_corss_similarity, logits_iou, gt_times, visual_content_emb, flag=flag)

            return loss_value1 + loss_value2, None

    def get_proposal_results(scores, regress, durations):
        out_sorted_times = []
        T = scores.shape[-1]

        regress = regress.cpu().detach().numpy()

        for score, reg, duration in zip(scores, regress, durations):
            sorted_indexs = np.dstack(np.unravel_index(np.argsort(score.cpu().detach().numpy().ravel())[::-1], (T, T))).tolist()
            sorted_indexs = np.array([ [reg[0,item[0],item[1]], reg[1,item[0],item[1]]] for item in sorted_indexs[0] if reg[0,item[0],item[1]] < reg[1,item[0],item[1]] ]).astype(float)
            sorted_indexs = torch.from_numpy(sorted_indexs).cuda()
            target_size = config.DATASET.NUM_SAMPLE_CLIPS // config.DATASET.TARGET_STRIDE
            out_sorted_times.append((sorted_indexs.float() / target_size * duration).tolist())

        return out_sorted_times

    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset)/config.TRAIN.BATCH_SIZE*config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])

    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)

    def on_update(state):# Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            best_metric = config.TEST.BEST_METRIC
            global best_score_iid
            global best_score_ood
            global best_table_iid
            global best_table_ood

            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg)
            table_message = ''

            test_state = engine.test(network, iterator('test_iid'), 'test_iid')
            loss_message += ' test_iid loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()

            test_table, _ = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                              'performance on test_iid set', best_metric)
            table_message += '\n' + test_table



            if not config.DATASET.NO_IID:
                val_state = engine.test(network, iterator('test_iid'), 'test_iid')
                loss_message += ' test_iid loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table, score_iid = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                 'performance on test_iid set', best_metric)
                table_message += '\n'+ val_table

                if score_iid > best_score_iid:
                    best_table_iid = val_table
                    best_score_iid = score_iid

                    filename_comp = os.path.join(config.MODEL_DIR,'{}/{}/best_iid.pkl'.format(dataset_name, local_time))
                    rootfolder = os.path.dirname(filename_comp)

                    if config.TRAIN.SAVE_CHECKPOINT:
                        if not os.path.exists(rootfolder):
                            print('Make directory %s ...' % rootfolder)
                            os.mkdir(rootfolder)
                        if torch.cuda.device_count() > 1:
                            torch.save(model.module.state_dict(), filename_comp)
                        else:
                            torch.save(model.state_dict(), filename_comp)



            if not config.DATASET.NO_OOD:
                val_state = engine.test(network, iterator('test_ood'), 'test_ood')
                loss_message += ' test_ood loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table, score_ood = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                 'performance on test_ood set', best_metric)
                table_message += '\n'+ val_table

                if score_ood > best_score_ood:
                    best_table_ood = val_table
                    best_score_ood = score_ood

                    filename_word = os.path.join(config.MODEL_DIR,'{}/{}/best_ood.pkl'.format(dataset_name, local_time))
                    rootfolder = os.path.dirname(filename_word)

                    if config.TRAIN.SAVE_CHECKPOINT:
                        if not os.path.exists(rootfolder):
                            print('Make directory %s ...' % rootfolder)
                            os.mkdir(rootfolder)
                        if torch.cuda.device_count() > 1:
                            torch.save(model.module.state_dict(), filename_word)
                        else:
                            torch.save(model.state_dict(), filename_word)


            message = loss_message+table_message+'\n'

            saved_model_filename = os.path.join(config.MODEL_DIR,'{}/{}/iter{:06d}-{:.4f}-{:.4f}.pkl'.format( dataset_name, local_time, state['t'], test_state['Rank@N,mIoU@M'][0,0], test_state['Rank@N,mIoU@M'][0,1]))

            logger.info(message)

            best_msg = "\n Best Metric {}, test_iid {}, test_ood {}".format(best_metric, round(best_score_iid, 2), round(best_score_ood, 2))
            best_msg = best_msg + '\n' + '\n' + best_table_iid + '\n' + best_table_ood
            logger.info(best_msg)

            if config.TRAIN.SAVE_ALL_CHECKPOINT:
                rootfolder1 = os.path.dirname(saved_model_filename)
                rootfolder2 = os.path.dirname(rootfolder1)
                rootfolder3 = os.path.dirname(rootfolder2)
                if not os.path.exists(rootfolder3):
                    print('Make directory %s ...' % rootfolder3)
                    os.mkdir(rootfolder3)
                if not os.path.exists(rootfolder2):
                    print('Make directory %s ...' % rootfolder2)
                    os.mkdir(rootfolder2)
                if not os.path.exists(rootfolder1):
                    print('Make directory %s ...' % rootfolder1)
                    os.mkdir(rootfolder1)

                if config.TRAIN.SAVE_CHECKPOINT:
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), saved_model_filename)
                    else:
                        torch.save(model.state_dict(), saved_model_filename)


            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            state['loss_meter'].reset()

    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()


    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['sorted_segments_list'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset)/config.TEST.BATCH_SIZE))
            elif state['split'] == 'test_iid':
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
        annotations = state['iterator'].dataset.annotations
        state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['sorted_segments_list'], annotations, config.TRAIN.EVAL_METRIC, verbose=False)
        if config.VERBOSE:
            state['progress_bar'].close()

    engine = Engine()
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)
