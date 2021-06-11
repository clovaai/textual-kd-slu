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

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler
import numpy as np
import torch
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utility import span_masking

class IntentDataset(Dataset):
    def __init__(self, manifest_filepath,
                 intent_dict,
                 vq_dict=None,
                 text_dict=None,
                 augment=True,
                 masking_type=None,
                 augment_prob=0,
                 max_span=10,
                 min_span=0,
                 select_prob=0.1,
                 ):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        """
        self.intent_dict = intent_dict
        self.vq_dict = vq_dict
        self.text_dict = text_dict
        self.augment = augment
        self.masking_type = masking_type
        self.augment_prob = augment_prob
        self.max_span = max_span
        self.min_span = min_span
        self.select_prob = select_prob

        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript, intent = sample[1], ','.join(sample[6:]), ' '.join([sample[3], sample[4], sample[5]])
        wav = self.parse_audio(audio_path)
        transcript = self.parse_transcript(transcript)
        intent = self.parse_intent(intent)
        return wav, transcript, intent

    def parse_transcript(self, transcript_path):
        transcript = transcript_path.lower()
        transcript = self.text_dict(transcript)[:-1].type(torch.LongTensor)
        return transcript

    def parse_intent(self, intent):
        parsed_intent = self.intent_dict[intent]
        parsed_intent = torch.LongTensor([parsed_intent])

        return parsed_intent

    def parse_audio(self, audio_path):
        with open(audio_path, 'r', encoding='utf8') as transcript_file:
            seq = transcript_file.read().replace('\n', '')
        seq = torch.cat([torch.LongTensor([0]),self.vq_dict.encode_line(seq, append_eos=False, add_if_not_exist=False).type(torch.LongTensor)],dim=0)
        if self.augment:
            masking_idx = self.vq_dict.indices['<mask>']
            if self.masking_type == 'dynamic':
                dynamic = True
            else:
                dynamic = False
            seq, _ = span_masking(seq, seqfreq=self.select_prob/self.max_span, mask_idx=masking_idx, span=self.max_span, dynamic_span=dynamic)

        return seq

    def __len__(self):
        return self.size



def _collate_fn(batch):
    def afunc(p):
        return p[0].shape[0]
    def tfunc(p):
        return p[1].shape[0]

    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    longest_audiosample = max(batch, key=afunc)[0]
    longest_textsample = max(batch, key=tfunc)[1]
    minibatch_size = len(batch)
    max_audiolength = longest_audiosample.shape[0]
    max_textlength = longest_textsample.shape[0]
    input_audios = torch.zeros(minibatch_size, 1, max_audiolength)
    input_texts = torch.zeros(minibatch_size, 1, max_textlength)
    input_intents = torch.zeros(minibatch_size, 1, 1, dtype=torch.long)
    input_audio_percentages = torch.FloatTensor(minibatch_size)
    input_text_percentages = torch.FloatTensor(minibatch_size)
    audio_len = []
    text_len = []
    for x in range(minibatch_size):
        sample = batch[x]
        audio_tensor = sample[0]
        text_tensor = sample[1]
        intent = sample[2]
        audio_length = audio_tensor.shape[0]
        text_length = text_tensor.shape[0]
        audio_len.append(audio_length)
        text_len.append(text_length)
        AudiotensorShape = audio_tensor.shape
        TexttensorShape = text_tensor.shape
        AudiotensorNew = np.pad(audio_tensor, (0, abs(AudiotensorShape[0] - max_audiolength)), 'constant', constant_values=(1))
        TexttensorNew = np.pad(text_tensor, (0, abs(TexttensorShape[0] - max_textlength)), 'constant', constant_values=(1))
        input_audios[x][0].copy_(torch.FloatTensor(AudiotensorNew))
        input_texts[x][0].copy_(torch.IntTensor(TexttensorNew))
        input_intents[x][0].copy_(torch.LongTensor(intent))

        input_audio_percentages[x] = audio_length / float(max_audiolength)
        input_text_percentages[x] = text_length / float(max_textlength)
    return input_audios, input_texts, input_intents, input_audio_percentages, input_text_percentages, audio_len, text_len


class IntentDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(IntentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
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


class DistributedBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1, num_replicas=None, rank=None):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(DistributedBucketingSampler, self).__init__(data_source)
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.ids = list(range(0, len(data_source)))
        self.batch_size = batch_size
        self.bins = [self.ids[i:i + batch_size] for i in range(0, len(self.ids), batch_size)]
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.bins) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        offset = self.rank
        # add extra samples to make it evenly divisible
        bins = self.bins + self.bins[:(self.total_size - len(self.bins))]
        assert len(bins) == self.total_size
        samples = bins[offset::self.num_replicas]  # Get every Nth bin, starting from rank
        return iter(samples)

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(epoch)
        bin_ids = list(torch.randperm(len(self.bins), generator=g))
        self.bins = [self.bins[i] for i in bin_ids]




