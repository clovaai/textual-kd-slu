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

from torch.utils.data.sampler import Sampler
import numpy as np
import torch
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class IntentDataset(Dataset):
    def __init__(self, manifest_filepath,
                 intent_dict,
                 text_dict=None,
                 ):
        self.intent_dict = intent_dict
        self.text_dict = text_dict

        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)

    def __getitem__(self, index):
        sample = self.ids[index]
        transcript, intent = ','.join(sample[6:]), ' '.join([sample[3], sample[4], sample[5]])
        transcript = self.parse_transcript(transcript)
        intent = self.parse_intent(intent)
        return transcript, intent

    def parse_transcript(self, transcript_path):
        transcript = transcript_path.lower()
        transcript = self.text_dict(transcript)[:-1].type(torch.LongTensor)
        return transcript

    def parse_intent(self, intent):
        parsed_intent = self.intent_dict[intent]
        parsed_intent = torch.LongTensor([parsed_intent])

        return parsed_intent

    def __len__(self):
        return self.size



def _collate_fn(batch):
    def func(p):
        return p[0].shape[0]

    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    longest_textsample = max(batch, key=func)[0]
    minibatch_size = len(batch)
    max_textlength = longest_textsample.shape[0]
    input_texts = torch.zeros(minibatch_size, 1, max_textlength)
    input_intents = torch.zeros(minibatch_size, 1, 1, dtype=torch.long)
    for x in range(minibatch_size):
        sample = batch[x]
        text_tensor = sample[0]
        intent = sample[1]
        TexttensorShape = text_tensor.shape
        TexttensorNew = np.pad(text_tensor, (0, abs(TexttensorShape[0] - max_textlength)), 'constant', constant_values=(1))
        input_texts[x][0].copy_(torch.IntTensor(TexttensorNew))
        input_intents[x][0].copy_(torch.LongTensor(intent))
    return input_texts, input_intents


class IntentDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(IntentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):

        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)




