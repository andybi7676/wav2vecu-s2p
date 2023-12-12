import subprocess
import glob
import os
import os.path as osp
import tqdm
from collections import defaultdict

root = '/work/b07502072/corpus/u-s2s/audio/cv4/es'
included_tsvs = [
    "train.tsv",
    "dev.tsv",
    "test.tsv",
]

valid_fids = []
print("Getting valid fids...")
for tsv_split in included_tsvs:
    tsv_fp = osp.join(root, tsv_split)
    with open(tsv_fp, 'r') as fr:
        cols = fr.readline()
        for l in fr.readlines():
            fid = l.split('\t')[1].split('.')[0]
            valid_fids.append(fid)

# unsuccessed = []
dest_dir = osp.join(root, "clips_wav")
os.makedirs(dest_dir, exist_ok=True)
with open(osp.join(root, 'mp3_to_wav_unsuccessed.log'), 'w') as fw:
    for fid in tqdm.tqdm(valid_fids):
        file_path = osp.join(root, f"clips/{fid}.mp3")
        new_file_path = osp.join(dest_dir, f"{fid}.wav")
        if not os.path.exists(new_file_path):
            try:
                subprocess.call(['ffmpeg', '-i', file_path, new_file_path])
            except:
                print(f"failed to generate wav file, fid={fid}.", file=fw)
                # unsuccessed.append(file_path)




