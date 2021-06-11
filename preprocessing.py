'''
Textual KD SLU
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import wave
import argparse
import json
import os

parser = argparse.ArgumentParser(description='pre-processing')
parser.add_argument('--config', type=str)
args = parser.parse_args()

with open(args.config, 'r') as config_file:
    config = json.load(config_file)
print("Step 1 / 3")
print("making tsv file for tokenize with vq-wav2vec")
print("making train tsv")
with open(os.path.join(config['tsv_path'], 'train.tsv'),'w') as tsv:
    with open(os.path.join(config['manifest_path'], 'wav_fsc_train.csv')) as trains:
        for train in trains:
            path = train.split(',')[1]
            wav = wave.open(path)
            tsv.write(path+'\t'+str(wav.getnframes())+'\n')

print("making valid tsv")
with open(os.path.join(config['tsv_path'], 'valid.tsv'),'w') as tsv:
    with open(os.path.join(config['manifest_path'], 'wav_fsc_valid.csv')) as valids:
        for valid in valids:
            path = valid.split(',')[1]
            wav = wave.open(path)
            tsv.write(path+'\t'+str(wav.getnframes())+'\n')

print("making test tsv")
with open(os.path.join(config['tsv_path'], 'test.tsv'),'w') as tsv:
    with open(os.path.join(config['manifest_path'], 'wav_fsc_test.csv')) as tests:
        for test in tests:
            path = test.split(',')[1]
            wav = wave.open(path)
            tsv.write(path+'\t'+str(wav.getnframes())+'\n')

print("end making tsv")

print("Step 2 / 3")
print("extract tokenized file with vq-wav2vec")
import os

os.system('python ./utility/vq-wav2vec_featurize.py' +
          ' --data-dir ' + config['tsv_path'] +
          ' --output-dir ' + config['output_dir'] +
          ' --checkpoint ' + config['vq-wav2vec'] +
          ' --split train valid test --extension tsv')

print("end extracting tokenized file")

print("Step 3 / 3")
print("split src file to each files")

with open(os.path.join(config['tsv_path'], 'train.tsv')) as tsvs:
    with open(os.path.join(config['output_dir'],'train.src')) as srcs:
        tsv = tsvs.readlines()
        src = srcs.readlines()
        path = config['vqs_output_dir']
        assert len(tsv) == len(src)
        for i in range(len(tsv)):
            filename = tsv[i].strip('\n').split('\t')[0].split('/')[-1].split('.')[0]
            ext = '.txt'
            with open(os.path.join(path,filename)+ext, 'w') as f:
                f.write(src[i].strip('\n'))

print("End all preprocessing")