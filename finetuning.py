'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import os, sys
import logging
import torch.nn.functional as F

logger = logging.getLogger('root')
FORMAT = "[%(asctime)s] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
logger.info('logger created')
logger.info('try to importing the others')
import collections
import argparse
import random
import time
import numpy as np
import torch.nn as nn
from utility.util import *
from models import CrossBert, SpeechModel
from loader.data_loader import IntentDataset, IntentDataLoader, BucketingSampler, DistributedBucketingSampler
import json
from utility import build_roberta_model
from fairseq.models.roberta import RobertaModel

logger.info(torch.cuda.is_available())
parser = argparse.ArgumentParser(description='Uniter training')
parser.add_argument('--train-manifest', metavar='Path',
                    help='path to train manifest csv', default='./manifest/vq_fsc_train.csv')
parser.add_argument('--val-manifest', metavar='Path',
                    help='path to validation manifest csv', default='./manifest/vq_fsc_valid.csv')
parser.add_argument('--intent-path', metavar='Path',
                    help="path to intent file", default='./manifest/intent_dict')
parser.add_argument('--infer-manifest', metavar='DIR', default='./manifest/vq_fsc_test.csv')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--smoothing', type=float, default=0)
parser.add_argument('--config', type=str)

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def SoftCrossEntropyLoss(label, pred, temp=1):
    return (-1*F.softmax(label, dim=0)*F.log_softmax(pred/temp, dim=0)).sum(dim=0).mean()

if __name__ == '__main__':

    print("train's main entered")
    args = parser.parse_args()

    # Set seeds for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ##load config file
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    device = torch.device("cuda")
    main_proc = True
    loss_results = torch.Tensor(config['epoch'])
    torch.Tensor(config['epoch'])
    torch.Tensor(config['epoch'])
    avg_loss, start_epoch, start_iter = 0, 0, 0
    best_val_loss = 999999
    best_acc = 0
    inf_best = 0

    vq_arg_path = 'configs/args/vq_roberta.args'
    vq_dict_path = config["vq_bert_dict"]

    vq_bert = build_roberta_model(vq_dict_path, vq_arg_path)
    text_bert = RobertaModel.from_pretrained(config['text_bert_path'], checkpoint_file='model.pt')
    vq_dict = vq_bert.task.source_dictionary
    text_dict = text_bert.encode

    idx = 0
    intent_dict = collections.OrderedDict()
    with open(args.intent_path) as intents:
        for intent in intents:
            intent_dict.update({intent.strip(): idx})
            idx += 1
    num_class = len(intent_dict)

    premodel = CrossBert(vq_bert=vq_bert, text_bert=text_bert)
    train_dataset = IntentDataset(manifest_filepath=args.train_manifest,
                                  vq_dict=vq_dict,
                                  text_dict=text_dict,
                                  augment=config['token_masking'],
                                  masking_type=config['masking_type'],
                                  augment_prob=config['augment_prob'],
                                  max_span=config['max_span'],
                                  min_span=config['min_span'],
                                  select_prob=config['select_prob'],
                                  intent_dict=intent_dict
                                  )
    test_dataset = IntentDataset(manifest_filepath=args.val_manifest,
                                 vq_dict=vq_dict,
                                 text_dict=text_dict,
                                 augment=False,
                                 intent_dict=intent_dict)
    infer_dataset = IntentDataset(manifest_filepath=args.infer_manifest,
                                  vq_dict=vq_dict,
                                  text_dict=text_dict,
                                  augment=False,
                                  intent_dict=intent_dict)

    train_sampler = BucketingSampler(train_dataset, batch_size=config['batch_size'])

    train_loader = IntentDataLoader(train_dataset,
                                    num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = IntentDataLoader(test_dataset, batch_size=config['eval_size'],
                                   num_workers=args.num_workers)
    infer_loader = IntentDataLoader(infer_dataset, batch_size=config['eval_size'],
                                    num_workers=args.num_workers)

    model_dict = torch.load(config["pre_kd_model"])
    text_ic = torch.load(config["text_fine_model"])

    premodel = nn.DataParallel(premodel)
    premodel.load_state_dict(model_dict)
    model_dict = None
    del model_dict
    premodel = premodel.to(device)

    model = SpeechModel(vq_bert=premodel,
                        text_bert=text_bert,
                        text_ic=text_ic,
                        time_prob=config['time_prob'],
                        time_span=config['time_span'],
                        channel_prob=config['channel_prob'],
                        channel_span=config['channel_span'],
                        num_class=num_class,
                        augment=config['emb_masking'])
    
    if config['am_pre_model'] is not None:
        logger.info("")
        logger.info("ASR parameter loaded")
        logger.info("")
        with torch.no_grad():
            vq_asr = torch.load(config['am_pre_model'])

            new_state_dict = collections.OrderedDict()

            for k, v in vq_asr['state_dict'].items():
                if 'vq_bert.vq_bert' in k:
                    k = k.replace('vq_bert.vq_bert', 'vq_bert')
                if 'text_bert' in k or 'vq_lm_head' in k:
                    continue
                new_state_dict[k] = v

            key_list = list(new_state_dict.keys())
            # for load asr parameter, update soon
            vq_key = key_list[:203]
            crnn_key = key_list[203:277]

        for i in crnn_key:
            model.state_dict()[i].copy_(new_state_dict[i])

        new_state_dict = None
        del new_state_dict

    model = model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=config['lr'], betas=(0.9, 0.999), eps=1e-06, weight_decay=0.01,
                                 amsgrad=False)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    L1Loss = torch.nn.L1Loss(reduction='sum')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(start_epoch, config['epoch']):
        acc_avg = 0
        train_sampler.shuffle(epoch)
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        avg_loss = 0
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break

            audios, texts, intents, audio_percentages, text_percentages, audio_len, text_len = data
            audios = audios.type(torch.LongTensor).to(device)
            texts = texts.type(torch.LongTensor).to(device)
            intents = intents.type(torch.LongTensor).to(device).reshape(-1)
            input_sizes = audio_percentages.mul_(int(audios.size(2))).int()

            audio_logit, text_logit = model(audios, texts, input_sizes)
            audio_logit = audio_logit.squeeze(1)
            text_logit = text_logit.squeeze(1)
            acc = (intents == audio_logit.max(1)[1]).type(torch.FloatTensor).sum() / audios.size(0) * 100
            loss = criterion(audio_logit, intents).to(device)
            loss += L1Loss(audio_logit, text_logit)/audios.size(0)

            acc_avg += acc
            loss_value = loss.item()

            # Check to ensure valid loss was calculated
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:
                optimizer.zero_grad()
                # compute gradient
                loss.backward()
                optimizer.step()

            else:
                logger.info(error)
                logger.info('Skipping grad update')
                loss_value = 0

            avg_loss += loss_value
            losses.update(loss_value, audios.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            logger.info('Epoch: [%d][%d/%d]\t'
                        'Time %.3f (%.3f)\t'
                        'Data %.3f (%.3f)\t'
                        'Loss %.4f (%.4f) acc : %.3f (%.3f)\t',
                        (epoch + 1), (i + 1), len(train_sampler),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg,
                        losses.val, losses.avg, acc, acc_avg / (i + 1))


        avg_loss /= len(train_sampler)
        acc_avg /= len(train_sampler)

        logger.info("Evaluation Start")
        with torch.no_grad():
            val_len = 0
            val_avg_loss = 0
            val_true = 0
            val_acc_avg = 0
            model.eval()
            for i, (data) in enumerate(test_loader, start=start_iter):

                audios, _, intents, audio_percentages, _, audio_len, _ = data
                audios = audios.type(torch.LongTensor).to(device)
                intents = intents.type(torch.LongTensor).to(device).reshape(-1)
                input_sizes = audio_percentages.mul_(int(audios.size(2))).int()
                val_len += intents.size()[0]
                audio_logit = model(audios, None, input_sizes)
                audio_logit = audio_logit.squeeze(1)
                val_true += (intents == audio_logit.max(1)[1]).type(torch.FloatTensor).sum()
            val_acc_avg = val_true/val_len*100
            logger.info("num of valid data : "+str(val_len))

        val_acc_avg = float(val_acc_avg)
        acc_avg = float(acc_avg)

        if val_acc_avg >= best_acc:
            logger.info("find new best check point, exec inference")
            best_acc = val_acc_avg
            with torch.no_grad():
                inf_avg_loss = 0
                inf_true = 0
                inf_acc_avg = 0
                inf_len = 0
                model.eval()
                for i, (data) in enumerate(infer_loader, start=start_iter):

                    audios, _, intents, audio_percentages, _, audio_len, _ = data
                    audios = audios.type(torch.LongTensor).to(device)
                    intents = intents.type(torch.LongTensor).to(device).reshape(-1)
                    input_sizes = audio_percentages.mul_(int(audios.size(2))).int()
                    inf_len += intents.size()[0]
                    audio_logit = model(audios, None, input_sizes)
                    audio_logit = audio_logit.squeeze(1)
                    inf_true += (intents == audio_logit.max(1)[1]).type(torch.FloatTensor).sum()

                inf_acc_avg = inf_true / inf_len * 100
                logger.info("num of infer data : " + str(inf_len))
                inf_acc_avg = float(inf_acc_avg)

        for g in optimizer.param_groups:
            if g['lr'] >= 1e-8:
                g['lr'] = g['lr'] / config['anneal_rate']
        print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))

