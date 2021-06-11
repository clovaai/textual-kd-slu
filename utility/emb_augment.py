'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import numpy as np


def time_masking(x, x_len, prob, span, replace=0):
    num_idx = int(x_len*prob/float(span) + np.random.rand())
    x_idx = np.asarray(np.arange(x_len.item()-span))
    if x_idx == []:
        return x
    mask_idx = np.random.choice(x_idx, num_idx, replace=False)
    mask_idx = np.asarray(
        [
            mask_idx[j] + offset
            for j in range(len(mask_idx))
            for offset in range(span)
        ]
    )
    mask_idx = sorted(np.asarray(np.unique(mask_idx)))

    x[mask_idx] = replace

    return x


def channel_masking(x, dim, prob, span, replace=0):
    num_idx = int(dim*prob/float(span)+ np.random.rand())
    x_idx = np.asarray(np.arange(dim-span))
    mask_idx = np.random.choice(x_idx, num_idx, replace=False)
    mask_idx = np.asarray(
        [
            mask_idx[j] + offset
            for j in range(len(mask_idx))
            for offset in range(span)
        ]
    )
    mask_idx = sorted(np.asarray(np.unique(mask_idx)))
    x = x.transpose(0,1)
    x[mask_idx] = replace
    x = x.transpose(0,1)
    return x
