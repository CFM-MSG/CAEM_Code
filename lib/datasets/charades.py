""" Dataset loader for the ActivityNet Captions dataset """
import os
import json
import random
from collections import OrderedDict
import numpy as np

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
import tqdm
import nltk
import pdb

from . import average_to_fixed_length
from lib.core.eval import iou
from lib.core.config import config

random.seed(1111)

class Charades(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.840B.300d"](cache="./data")
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(Charades, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.json_dir = config.JSON_DIR
        self.split = split

        with open('./data/Charades/words_vocab_charades.json', 'r') as f:
            tmp = json.load(f)
            self.itos = tmp['words']
        self.stoi = OrderedDict()
        for i, w in enumerate(self.itos):
            self.stoi[w] = i
        print(len(self.stoi))

        with open(os.path.join(self.json_dir, '{}.json'.format(split)),'r') as f:
            annotations = json.load(f)
        # self.tmp_annotations = annotations
        anno_pairs = []
        max_sent_len = 0
        for vid, video_anno in annotations.items():
            duration = video_anno['duration']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1] and timestamp[0] < duration:
                    sentence = sentence.replace(',',' ').replace('/',' ').replace('\"',' ').replace('-',' ').replace(';',' ').replace('.',' ').replace('&',' ').replace('?',' ').replace('!',' ').replace('(',' ').replace(')',' ')
                    anno_pairs.append(
                        {
                            'video': vid,
                            'duration': duration,
                            'times':[max(timestamp[0],0),min(timestamp[1],duration)],
                            'description':sentence,
                        }
                    )
                    if len(sentence.split()) > max_sent_len:
                        max_sent_len = len(sentence.split())

        self.annotations = anno_pairs
        print('max_sent_len', max_sent_len)
        # pdb.set_trace()


    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        sentence = self.annotations[index]['description']
        duration = self.annotations[index]['duration']

        word_label = [self.stoi.get(w.lower(), 10727) for w in sentence.split()]
        range_i = range(len(word_label))
        # if np.random.uniform(0,1)<0.8:
        word_mask = [1. if np.random.uniform(0,1)<0.15 else 0. for _ in range_i]
        if np.sum(word_mask) == 0.:
            mask_i = np.random.choice(range_i)
            word_mask[mask_i] = 1.
        if np.sum(word_mask) == len(word_mask):
            unmask_i = np.random.choice(range_i)
            word_mask[unmask_i] = 0.
        # else:
        #     word_mask = [0. for _ in range_i]

        word_label = torch.tensor(word_label, dtype=torch.long)
        word_mask = torch.tensor(word_mask, dtype=torch.float)

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        visual_input, visual_mask = self.get_video_features(video_id)

        if self.split == 'train':
            aug_data = self.get_aug_data(gt_s_time, gt_e_time, duration, visual_input)
        else:
            aug_data = {
            'visual_input': None,
            'duration': None,
            'map_gt': None,
            'gt_times': None
            }

        # Time scaled to same size
        if config.DATASET.NUM_SAMPLE_CLIPS > 0:
            # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
            visual_input = average_to_fixed_length(visual_input)
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        # Time unscaled NEED FIXED WINDOW SIZE
        else:
            num_clips = visual_input.shape[0]//config.DATASET.TARGET_STRIDE
            raise NotImplementedError
            # torch.arange(0,)

        map_gt = np.zeros((5, num_clips+1), dtype=np.float32)

        clip_duration = duration/num_clips
        gt_s = gt_s_time/clip_duration
        gt_e = gt_e_time/clip_duration
        gt_length = gt_e - gt_s
        gt_center = (gt_e + gt_s) / 2.
        map_gt[0, :] = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_s)/(0.25*gt_length) ) )
        map_gt[0, map_gt[0, :]>=0.6] = 1.
        map_gt[0, map_gt[0, :]<0.1353] = 0.
        map_gt[1, :] = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_e)/(0.25*gt_length) ) )
        map_gt[1, map_gt[1, :]>=0.6] = 1.
        map_gt[1, map_gt[1, :]<0.1353] = 0.
        # map_gt[2, gt_s_idx:gt_e_idx] = 1.
        map_gt[2, :] = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_center)/(0.21233*gt_length) ) )
        map_gt[2, map_gt[2, :]>=0.78] = 1.
        map_gt[2, map_gt[2, :]<0.0625] = 0.
        map_gt[3, :] = gt_s - np.arange(num_clips+1)
        map_gt[4, :] = gt_e - np.arange(num_clips+1)
        if (map_gt[0, :]>0.4).sum() == 0:
            p = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_s)/(0.25*gt_length) ) )
            idx = np.argsort(p)
            map_gt[0, idx[-1]] = 1.
        if (map_gt[1, :]>0.4).sum() == 0:
            p = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_e)/(0.25*gt_length) ) )
            idx = np.argsort(p)
            map_gt[1, idx[-1]] = 1.
        if map_gt[2, :].sum() == 0:
            p = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_center)/(0.21233*gt_length) ) )
            idx = np.argmax(p)
            map_gt[2, idx] = 1.

        item = {
            'visual_input': visual_input,
            'vis_mask': visual_mask,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': duration,
            'txt_mask': torch.ones(word_vectors.shape[0],),
            'map_gt': torch.from_numpy(map_gt),
            'word_label': word_label,
            'word_mask': word_mask,
            'gt_times': torch.from_numpy(np.array([gt_s, gt_e], dtype=np.float32)),
            'aug_visual_input': aug_data['visual_input'],
            'aug_duration': aug_data['duration'],
            'aug_map_gt': aug_data['map_gt'],
            'aug_gt_times': aug_data['gt_times']
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'i3d'
        with h5py.File(os.path.join(self.data_dir, 'i3d_features.hdf5'), 'r') as f:
            features = torch.from_numpy(f[vid][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

    def get_aug_data(self, gt_s_time, gt_e_time, duration, visual_input):
        offset_flag = random.random() < config.DATASET.RANDOM_FLAG  # for CG

        if offset_flag and gt_s_time / duration < 0.12 and gt_e_time / duration < 0.55:
            num_frames = visual_input.size(0)
            st1 = int(gt_s_time / duration * num_frames)
            ed1 = int(gt_e_time / duration * num_frames)
            tmp_feature = torch.zeros_like(visual_input)
            offset_dis = random.uniform(0.1, 0.45)
            gt_s_time += offset_dis * duration
            gt_e_time += offset_dis * duration

            offset_dis = int(offset_dis * num_frames)
            st2 = st1 + offset_dis
            ed2 = ed1 + offset_dis

            tmp_feature[0:st1,:] = visual_input[0:st1,:]
            tmp_feature[st1:st2,:] = visual_input[ed1:ed2,:]
            tmp_feature[st2:ed2,:] = visual_input[st1:ed1,:]
            tmp_feature[ed2:,:] = visual_input[ed2:,:]
            visual_input = tmp_feature

        if config.DATASET.NUM_SAMPLE_CLIPS > 0:
            visual_input = average_to_fixed_length(visual_input)
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
        # Time unscaled NEED FIXED WINDOW SIZE
        else:
            num_clips = visual_input.shape[0]//config.DATASET.TARGET_STRIDE
            raise NotImplementedError

        map_gt = np.zeros((5, num_clips+1), dtype=np.float32)

        clip_duration = duration/num_clips
        gt_s = gt_s_time/clip_duration
        gt_e = gt_e_time/clip_duration
        gt_length = gt_e - gt_s
        gt_center = (gt_e + gt_s) / 2.
        map_gt[0, :] = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_s)/(0.25*gt_length) ) )
        map_gt[0, map_gt[0, :]>=0.6] = 1.
        map_gt[0, map_gt[0, :]<0.1353] = 0.
        map_gt[1, :] = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_e)/(0.25*gt_length) ) )
        map_gt[1, map_gt[1, :]>=0.6] = 1.
        map_gt[1, map_gt[1, :]<0.1353] = 0.
        # map_gt[2, gt_s_idx:gt_e_idx] = 1.
        map_gt[2, :] = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_center)/(0.21233*gt_length) ) )
        map_gt[2, map_gt[2, :]>=0.78] = 1.
        map_gt[2, map_gt[2, :]<0.0625] = 0.
        map_gt[3, :] = gt_s - np.arange(num_clips+1)
        map_gt[4, :] = gt_e - np.arange(num_clips+1)
        if (map_gt[0, :]>0.4).sum() == 0:
            p = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_s)/(0.25*gt_length) ) )
            idx = np.argsort(p)
            map_gt[0, idx[-1]] = 1.
        if (map_gt[1, :]>0.4).sum() == 0:
            p = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_e)/(0.25*gt_length) ) )
            idx = np.argsort(p)
            map_gt[1, idx[-1]] = 1.
        if map_gt[2, :].sum() == 0:
            p = np.exp( -0.5 * np.square( (np.arange(num_clips+1)-gt_center)/(0.21233*gt_length) ) )
            idx = np.argmax(p)
            map_gt[2, idx] = 1.

        item = {
            'visual_input': visual_input,
            'duration': duration,
            'map_gt': torch.from_numpy(map_gt),
            'gt_times': torch.from_numpy(np.array([gt_s, gt_e], dtype=np.float32))
        }
        return item