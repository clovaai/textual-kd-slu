'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import torch.nn as nn
from models import SpeechModel, CrossBert

def build_inference_model(vq_bert, intent_dict, device):


    premodel = CrossBert(vq_bert=vq_bert) #build vq-bert
    premodel = nn.DataParallel(premodel) # for add .module to dict_keys
    num_class = len(intent_dict)
    model = SpeechModel(vq_bert=premodel,
                        text_bert=None,
                        text_ic=None,
                        num_class=num_class,
                        augment=False)
    model.to(device)

    return model