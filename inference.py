'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import os, sys
import sys
import torch
from utility import build_inference_model, build_roberta_model, vq_tokenize
import collections
from fairseq.models.wav2vec import Wav2VecModel
import argparse
import json

parser = argparse.ArgumentParser(description='inference two-kd')
parser.add_argument('--config', type=str)
parser.add_argument('--wav', type=str)
args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

with open(args.config, 'r') as config_file:
    config = json.load(config_file)

intent_dict = collections.OrderedDict()
idx = 0
with open(config['intent_dict']) as intents:
    for intent in intents:
        intent_dict.update({intent.strip(): idx})
        idx += 1
intent_keys = list(intent_dict.keys())

arg_path = 'configs/args/vq_roberta.args'
dict_path = config["vq_bert_dict"]
cp = torch.load(config["vq-wav2vec"])
vq_wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
vq_wav2vec.load_state_dict(cp['model'])
vq_wav2vec.eval()
vq_bert = build_roberta_model(dict_path, arg_path)
model = build_inference_model(vq_bert, intent_dict, device)
trained = torch.load(config["checkpoint"], map_location=device)
model.load_state_dict(trained, strict=False) # for discard text weight

audios = vq_tokenize(args.wav, vq_wav2vec, vq_bert.task.source_dictionary)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

with torch.no_grad():
    model.eval()
    audio_logit = model(audios.unsqueeze(1), None, torch.LongTensor([len(audios)]))
    audio_logit = audio_logit.squeeze(1)
    intent = audio_logit.max(1)[1]
action, object, location = intent_keys[intent].split(' ')
print('action : ', action)
print('object : ', object)
print('location : ', location)




