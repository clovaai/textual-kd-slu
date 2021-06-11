'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import torch
import soundfile as sf
def vq_tokenize(wav_path, vq_wav2vec, vq_bert_dict):
    wav, sr = sf.read(wav_path)
    wav = torch.from_numpy(wav).unsqueeze(0).float()
    z = vq_wav2vec.feature_extractor(wav)
    _, idxs = vq_wav2vec.vector_quantizer.forward_idx(z)
    idxs = idxs.squeeze(0)
    vq_input = ""
    for i in idxs:
        vq_input += (str(i[0].item()) + '-' + str(i[1].item()) + ' ')
    vq_input = vq_input.strip()
    seq = torch.cat([torch.LongTensor([0]), vq_bert_dict.encode_line(vq_input, append_eos=False,
                                                                                       add_if_not_exist=False).type(
        torch.LongTensor)], dim=0)
    audios = seq.type(torch.LongTensor)
    return audios