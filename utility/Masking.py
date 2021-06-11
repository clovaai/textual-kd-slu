'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import random
import math
import torch

def masking(seq, seqfreq=0.15, maskfreq=0.8, repfreq=0.1, vocab_size=0, mask_idx=1):
    seq_len = len(seq)
    seq_idx = list(range(1, seq_len))
    masking_idx = random.sample(seq_idx, math.ceil(seq_len*seqfreq))
    for token in masking_idx:
        seqrand = random.random()
        if seqrand < maskfreq:
            seq[token] = mask_idx
        elif seqrand < maskfreq+repfreq:
            # for prevent replacing special tokens ( CLS, PAD, SEP, UNK )
            seq[token] = random.randrange(4, vocab_size)
    seq = torch.LongTensor(seq)

    return seq, torch.LongTensor(sorted(masking_idx))

import numpy as np
def span_masking(seq, seqfreq=0.025, mask_idx=1, span=20, dynamic_span=False):
    seq_len = len(seq)
    masking_num = math.ceil(seq_len * seqfreq)
    masking_idx = []
    # for no masking on CLS and SEP token + prevent masking span explode array len
    seq_idx = list(range(1, seq_len-span))

    sampling_idx = np.random.choice(seq_idx, masking_num, replace=False)
    for i in sampling_idx:
        chunk = []
        if dynamic_span:
            span = int(np.random.uniform(low=1, high=span))
        for c in range(span):
            chunk.append(i+c)
        masking_idx += chunk
    masking_idx = sorted(list(set(masking_idx)))

    seq = torch.LongTensor(seq)
    seq[masking_idx] = mask_idx

    return seq, torch.LongTensor(masking_idx)
