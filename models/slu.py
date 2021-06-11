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

import torch
import torch.nn as nn
from models import TextModel
import math
from utility.emb_augment import time_masking, channel_masking
from collections import OrderedDict
import torch.nn.functional as F

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)

        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class SpeechModel(nn.Module):
    def __init__(self, vq_bert,
                 text_bert,
                 text_ic,
                 num_class,
                 augment=True,
                 time_prob=0.5,
                 time_span=10,
                 channel_prob=0.05,
                 channel_span=10):
        super().__init__()
        ### temp args
        self.augment = augment
        self.time_prob = time_prob
        self.time_span = time_span
        self.channel_prob = channel_prob
        self.channel_span = channel_span
        ###

        self.vq_bert = vq_bert
        self.vq_bert.module.finetuning()
        if text_bert is not None:
            self.text_bert = TextModel(text_bert, num_class=num_class)
            self.text_bert.load_state_dict(text_ic)
            self.audio_intenter = nn.Linear(768, num_class)
            self.text_intenter = nn.Linear(768, num_class)
        else:
            self.audio_intenter = nn.Linear(768, num_class)


        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = 768
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32
        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=768, rnn_type=nn.LSTM,
                       bidirectional=True, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(5 - 1):
            rnn = BatchRNN(input_size=768, hidden_size=768, rnn_type=nn.LSTM,
                           bidirectional=True)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))


    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)

        return seq_len.int()

    def forward(self, audios, texts=None, audio_len=0):
        if texts is not None:
            self.text_bert.eval()
        with torch.no_grad():
            audio_cls, audio_prob = self.vq_bert(audios)
            if texts is not None:
                text_logit, _ = self.text_bert(texts.squeeze(1))
        if self.training and self.augment:
            for i in range(len(audio_prob)):
                audio_prob[i] = time_masking(audio_prob[i], audio_len[i], prob=self.time_prob, span=self.time_span, replace=0)
                audio_prob[i] = channel_masking(audio_prob[i], 768, prob=self.channel_prob, span=self.channel_span, replace=0)
        out_lengths = self.get_seq_lens(audio_len)
        audio_prob = audio_prob.unsqueeze(1).transpose(2, 3)
        audio_prob, audio_len = self.conv(audio_prob, out_lengths)
        sizes = audio_prob.size()
        x = audio_prob.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH
        for rnn in self.rnns:
            x = rnn(x, out_lengths)
        x = x.transpose(0, 1)
        x = x.max(dim=1)[0]
        audio_logit = self.audio_intenter(x)

        if texts is not None:
            return audio_logit, text_logit
        else:
            return audio_logit
