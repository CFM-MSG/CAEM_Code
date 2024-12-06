import torch
from torch import nn
from lib.core.config import config
import lib.models.frame_modules as frame_modules
import lib.models.bert_modules as bert_modules

class CAEM(nn.Module):
    def __init__(self):
        super(CAEM, self).__init__()

        self.frame_layer = getattr(frame_modules, config.CAEM.FRAME_MODULE.NAME)(config.CAEM.FRAME_MODULE.PARAMS)
        self.bert_layer = getattr(bert_modules, config.CAEM.VLBERT_MODULE.NAME)(config.DATASET.NAME, config.CAEM.VLBERT_MODULE.PARAMS)

    def forward(self, textual_input, textual_mask, word_mask, visual_input, aug_visual_input=None):

        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        vis_h = vis_h.transpose(1, 2)
        logits_text, logits_visual, logits_iou, iou_mask_map, visual_content_emb, text_query_emb, corss_similarity = self.bert_layer(textual_input, textual_mask, word_mask, vis_h, num=0)
        logits_visual = logits_visual.transpose(1, 2)


        aug_vis_h = self.frame_layer(aug_visual_input.transpose(1,2))
        aug_vis_h = aug_vis_h.transpose(1, 2)
        aug_logits_text, aug_logits_visual, aug_logits_iou, aug_iou_mask_map, aug_visual_content_emb, aug_text_query_emb, aug_corss_similarity = self.bert_layer(textual_input, textual_mask, word_mask, aug_vis_h, num=1)
        aug_logits_visual = aug_logits_visual.transpose(1, 2)

        return logits_text, logits_visual, logits_iou, iou_mask_map, visual_content_emb, text_query_emb, corss_similarity, \
                aug_logits_text, aug_logits_visual, aug_logits_iou, aug_iou_mask_map, aug_visual_content_emb, aug_text_query_emb, aug_corss_similarity

    def extract_features(self, textual_input, textual_mask, visual_input):
        pass
