'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import torch.nn as nn
import torch


class CrossBert(nn.Module):
    def __init__(self,
                 vq_bert=None,
                 text_bert=None
                 ):
        super().__init__()
        self.set_mode = 'pretrain'
        self.vq_bert = vq_bert
        self.text_bert = text_bert
        # for fairseq version error (later version is encoder)
        try:
            self.vq_lm_head = vq_bert.model.encoder.lm_head
        except:
            self.vq_lm_head = vq_bert.model.decoder.lm_head

    def pretrain(self):
        self.set_mode = 'pretrain'

    def finetuning(self):
        self.set_mode = 'finetuning'

    def forward(self, audio, text=None):
        vq_out_emb = self.vq_bert.extract_features(audio.squeeze(1), return_all_hiddens=False)
        vq_cls = vq_out_emb[:,0,:]

        if self.set_mode == 'pretrain':
            with torch.no_grad():
                text_out_emb = self.text_bert.extract_features(text.squeeze(1), return_all_hiddens=False)
                text_cls = text_out_emb[:, 0, :]
            vq_logit = self.vq_lm_head(vq_out_emb)
            return vq_cls, text_cls, vq_logit

        else:
            return vq_cls, vq_out_emb