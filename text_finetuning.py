'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import logging
import sys
logger = logging.getLogger('root')
FORMAT = "[%(asctime)s] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)
logger.info('logger created')
logger.info('try to importing the others')
import argparse
import json
import math
import collections
import random
import time
import numpy as np
from utility.util import *
from models import TextModel
from fairseq.models.roberta import RobertaModel
from loader.text_data_loader import IntentDataset, IntentDataLoader, BucketingSampler

logger.info(torch.cuda.is_available())
parser = argparse.ArgumentParser(description='Uniter training')
parser.add_argument('--train-manifest', metavar='Path',
                    help='path to train manifest csv', default='./manifest/vq_fsc_train.csv')
parser.add_argument('--val-manifest', metavar='Path',
                    help='path to validation manifest csv', default='./manifest/vq_fsc_valid.csv')
parser.add_argument('--intent-path', metavar='Path',
                    help="path to intent file", default='./manifest/intent_dict')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
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
    avg_loss, start_epoch, start_iter = 0, 0, 0
    best_val_loss = 999999
    best_val_acc = 0
    inf_best = 0

    text_bert = RobertaModel.from_pretrained(config['bert_dir'], checkpoint_file=config['checkpoint'])
    text_dict = text_bert.encode

    idx = 0
    intent_dict = collections.OrderedDict()
    with open(config['intent_path']) as intents:
        for intent in intents:
            intent_dict.update({intent.strip(): idx})
            idx += 1
    num_class = len(intent_dict)

    train_dataset = IntentDataset(manifest_filepath=args.train_manifest, intent_dict=intent_dict, text_dict=text_dict)
    valid_dataset = IntentDataset(manifest_filepath=args.val_manifest, intent_dict=intent_dict, text_dict=text_dict)

    model = TextModel(text_bert, num_class=num_class)

    train_sampler = BucketingSampler(train_dataset, batch_size=config['batch_size'])

    train_loader = IntentDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)

    valid_loader = IntentDataLoader(valid_dataset, batch_size=config['batch_size'],
                                  num_workers=args.num_workers)

    model = model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=config['lr'], betas=(0.9, 0.999), eps=1e-06, weight_decay=0.01,
                                 amsgrad=False)
    fst = 0
    criterion = torch.nn.CrossEntropyLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    for epoch in range(start_epoch, config['epoch']):
        train_sampler.shuffle(epoch)
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        train_count = 0
        valid_count = 0
        train_acc = 0
        valid_acc = 0
        avg_loss = 0
        val_avg_loss = 0

        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            texts, intents = data

            texts = texts.type(torch.LongTensor).to(device)
            intents = intents.type(torch.LongTensor).to(device).reshape(-1)
            logit, _ = model(texts)
            logit = logit.squeeze(1)
            acc = (intents == logit.max(1)[1]).type(torch.FloatTensor).sum()
            train_acc += acc
            train_count += texts.size(0)
            loss = criterion(logit, intents).to(device)
            loss_value = loss.item()
            valid_loss, error = check_loss(loss, loss_value)

            if valid_loss:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                logger.info(error)
                logger.info('Skipping grad update')
                loss_value = 0

            avg_loss += loss_value
            losses.update(loss_value, texts.size(0))

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
                        losses.val, losses.avg, acc/texts.size(0)*100, train_acc/train_count*100 )
        avg_loss /= len(train_sampler)

        logger.info("Evaluation Start")
        with torch.no_grad():

            model.eval()
            for i, (data) in enumerate(valid_loader, start=start_iter):

                texts, intents = data
                texts = texts.type(torch.LongTensor).to(device)
                intents = intents.type(torch.LongTensor).to(device).reshape(-1)
                logit, _ = model(texts)
                logit = logit.squeeze(1)
                acc = (intents == logit.max(1)[1]).type(torch.FloatTensor).sum()
                valid_acc += acc
                valid_count += texts.size(0)
                loss = criterion(logit, intents).to(device)
                loss_value = loss.item()
                val_avg_loss += loss_value

                losses.update(loss_value, texts.size(0))

            val_avg_loss /= math.ceil(len(valid_dataset)/config['batch_size'])
            val_acc_avg = valid_acc/valid_count*100
            val_acc_avg = val_acc_avg.item()
            train_acc_avg = train_acc/train_count*100
            train_acc_avg = train_acc_avg.item()

            if best_val_loss >= val_avg_loss and main_proc:
                logger.info("update best loss epoch, save new model.")
                best_val_loss = val_avg_loss
                torch.save(model.state_dict(), './best_loss.pt')
            if best_val_acc < val_acc_avg and main_proc:
                logger.info("update best loss epoch, save new model.")
                best_val_acc = val_acc_avg
                torch.save(model.state_dict(), './best_acc.pt')
            if val_acc_avg == 100 and fst==0:
                torch.save(model.state_dict(), './first100.pt')
                fst+=1

            logger.info("best epoch : " + str(epoch + 1))
            logger.info("train loss : " + str(avg_loss))
            logger.info("valid loss : " + str(val_avg_loss))
            logger.info("train acc : " + str(train_acc_avg))
            logger.info("valid acc : " + str(val_acc_avg))

            for g in optimizer.param_groups:
                if g['lr'] >= 1e-8:
                    g['lr'] = g['lr'] / config['anneal_rate']
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))