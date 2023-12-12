import os
import os.path as osp
from collections import defaultdict
import glob
import tqdm

train_tsv = "w2v_manifest/train/train_all.tsv"
dev_tsv = "w2v_manifest/dev/dev.tsv"
test_tsv = "w2v_manifest/test/test.tsv"

fids_dict = defaultdict(lambda: False)
fids = []
for tsv_fp in [train_tsv, dev_tsv, test_tsv]:
    with open(tsv_fp) as fr:
        root = fr.readline()
        for l in fr:
            fid = l.split('\t')[0]
            fids_dict[fid] = True

wavs_fpath = glob.glob("./clips/*.wav")
accu_rm_num = 0
for fp in tqdm.tqdm(wavs_fpath):
    fid = osp.basename(fp)
    if not fids_dict[fid]:
        os.remove(fp)
        accu_rm_num += 1

print(f"Removed {accu_rm_num} .wav files.")


