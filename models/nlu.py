'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import torch.nn as nn

class TextModel(nn.Module):
    def __init__(self, text_bert, hidden_size=768, num_class=336):
        super().__init__()
        self.text_bert = text_bert
        self.intenter = nn.Linear(hidden_size, num_class)
        self.text_extractor = self.text_bert.extract_features

    def forward(self, texts):
        out_embedding = self.text_extractor(texts.squeeze(1), return_all_hiddens=False)
        cls_output = out_embedding[:, 0, :]
        intent_logit = self.intenter(cls_output)

        return intent_logit, cls_output
