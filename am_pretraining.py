'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0

---
MIT License

Copyright (c) 2017 Sean Naren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
import os
import random
import time
import numpy as np
import torch
from warpctc_pytorch import CTCLoss

from loader.asr_data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from utility.decoder import GreedyDecoder
from utility import build_roberta_model
from models.ammodel import DeepSpeech, supported_rnns
from utility.util import check_loss
from models import CrossBert

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--config', type=str)





def to_np(x):
    return x.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def evaluate(test_loader, device, model, decoder, target_decoder, save_output=False, verbose=True, half=False):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    for i, (data) in enumerate(test_loader):

        inputs, targets, input_percentages, target_sizes = data

        input_sizes = input_percentages.mul_(int(inputs.size(2))).int()
        inputs = inputs.type(torch.LongTensor).to(device)
        if half:
            inputs = inputs.half()
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out, output_sizes = model(inputs, input_sizes)

        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu().numpy(), output_sizes.numpy(), target_strings))
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))
            if verbose and i%300 == 0:
                logger.info('Ref:%s', reference.lower())
                logger.info('Hyp:%s', transcript.lower())
                logger.info('WER:%.4f\t'  
                            'CER:%.4f',(float(wer_inst) / len(reference.split())), (float(cer_inst) / len(reference.replace(' ', ''))))
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    return wer * 100, cer * 100, output_data

if __name__ == '__main__':
    args = parser.parse_args()

    ##load config file
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    # Set seeds for determinism
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_proc = True
    save_folder = config['save_folder']
    os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists
    loss_results, cer_results, wer_results = torch.Tensor(config['epochs']), torch.Tensor(config['epochs']), torch.Tensor(
        config['epochs'])
    best_cer = None
    avg_loss, start_epoch, start_iter = 0, 0, 0

    # vq_bert = RobertaModel.from_pretrained(config['vq-bert_dir'],
    #                                        checkpoint_file=config['vq-bert'])
    # text_bert = RobertaModel.from_pretrained(config['text-bert_dir'], checkpoint_file=config['text-bert'])

    vq_arg_path = 'configs/args/vq_roberta.args'
    vq_dict_path = config["vq_bert_dict"]
    text_arg_path = 'configs/args/text_roberta.args'
    text_dict_path = config["text_bert_dict"]
    vq_bert = build_roberta_model(vq_dict_path, vq_arg_path)
    text_bert = build_roberta_model(text_dict_path, text_arg_path)


    rnn_type = config['rnn-type'].lower()

    with open(config['label']) as label_file:
        labels = str(''.join(json.load(label_file)))

    premodel = CrossBert(vq_bert=vq_bert, text_bert=text_bert)
    model_dict = torch.load(config['pre_kd_model'])
    from collections import OrderedDict

    new_state_dict = OrderedDict()

    for k, v in model_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    premodel.load_state_dict(new_state_dict)
    model_dict = None
    new_state_dict = None
    del model_dict, new_state_dict

    premodel = premodel.to(device)
    model = DeepSpeech(rnn_hidden_size=config['hidden_size'],
                       nb_layers=config['hidden_layers'],
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       bidirectional=True,
                       vq_bert=premodel)


    decoder = GreedyDecoder(labels)
    train_dataset = SpectrogramDataset(manifest_filepath=args.train_manifest, labels=labels, vq_bert=vq_bert)
    test_dataset = SpectrogramDataset(manifest_filepath=args.val_manifest, labels=labels, vq_bert=vq_bert)

    train_sampler = BucketingSampler(train_dataset, batch_size=config['batch_size'])
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=config['batch_size'],
                                  num_workers=args.num_workers)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=config['lr'],
                                momentum=args.momentum, nesterov=True, weight_decay=1e-5)

    criterion = CTCLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data

            input_sizes = input_percentages.mul_(int(inputs.size(2))).int()
            # measure data loading time
            data_time.update(time.time() - end)
            inputs = inputs.type(torch.LongTensor).to(device)

            out, output_sizes = model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH

            float_out = out.float()  # ensure float32 for loss
            loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch

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
            losses.update(loss_value, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:
                logger.info('Epoch: [%d][%d/%d]\t'
                            'Time %.3f (%.3f)\t'
                            'Data %.3f (%.3f)\t'
                            'Loss %.4f (%.4f)\t',
                            (epoch + 1), (i + 1), len(train_sampler),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg,
                            losses.val, losses.avg)

            del loss, out, float_out

        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        logger.info('Training Summary Epoch: [%d]\t'
                    'Time taken (s): %.0f\t'
                    'Average Loss %.3f\t',
                    epoch + 1,
                    epoch_time,
                    avg_loss)

        start_iter = 0  # Reset start iteration for next epoch
        with torch.no_grad():
            wer, cer, output_data = evaluate(test_loader=test_loader,
                                             device=device,
                                             model=model,
                                             decoder=decoder,
                                             target_decoder=decoder)
        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        logger.info('Validation Summary Epoch: [%d]\t'
                    'Average WER %.3f\t'
                    'Average CER %.3f\t',
                    epoch + 1,
                    wer,
                    cer)

        values = {
            'loss_results': loss_results,
            'cer_results': cer_results,
            'wer_results': wer_results
        }


        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / config['learning_anneal']
        logger.info('Learning rate annealed to: %.6f', g['lr'])

        if main_proc and (best_cer is None or best_cer > cer):
            logger.info('Found better validated model, saving to %s', config['save_path'])
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results)
                       , config['save_path'])
            best_cer = cer
            avg_loss = 0

        logger.info('Shuffling batches...')
        train_sampler.shuffle(epoch)
