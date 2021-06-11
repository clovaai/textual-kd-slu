'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
from fairseq.data import Dictionary
import torch
from fairseq import tasks
from fairseq.models.roberta import RobertaModel
from fairseq.models.roberta import RobertaHubInterface

def build_roberta_model(dict_path, arg_path):
    arg = torch.load(arg_path)
    task = tasks.get_task('masked_lm')
    diction = Dictionary.load(dict_path)
    task = task(arg, diction)
    roberta = RobertaModel.build_model(arg, task=task)
    model = RobertaHubInterface(arg, task, roberta)

    return model

