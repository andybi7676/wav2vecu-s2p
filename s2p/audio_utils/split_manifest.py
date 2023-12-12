import os
import random

src_fp = '/work/b07502072/corpus/u-s2s/audio/cv4/fr/w2v_manifest/wo_sil/train_all/train.tsv'
tgt_fp = '/work/b07502072/corpus/u-s2s/audio/cv4/fr/w2v_manifest/wo_sil/train_70h/train.tsv'
sr = 16_000
tgt_hr = 70

with open(src_fp, 'r') as fr, open(tgt_fp, 'w') as fw:
    root = fr.readline()
    fw.write(root)
    lines = fr.readlines()
    total_len = len(lines)
    random.shuffle(lines)
    tgt_frames = tgt_hr*3600 * sr
    frames = 0
    for l in lines:
        frame = int(l.split('\t')[1].strip())
        frames += frame
        fw.write(l)
        if frames >= tgt_frames:
            break