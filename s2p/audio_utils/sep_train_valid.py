import os
import os.path as osp
import shutil
import tqdm

dest_dir = './wo_sil'
train_dir = 'train'
dev_dir = '../dev'

train_tsv = open('w2v_manifest/wo_sil/train.tsv', 'w')

with open('w2v_manifest/wo_sil/train_old.tsv', 'r') as fr:
    root = fr.readline().strip()
    root = osp.join(root, 'train')
    train_tsv.write(root + '\n')
    for line in tqdm.tqdm(fr.readlines()):
        fname = line.split('\t')[0].split('/')[-1]
        length = line.split('\t')[1]
        train_tsv.write(f"{fname}\t{length}")
        

# with open('w2v_manifest/wo_sil/valid.tsv', 'r') as fr:
#     root = fr.readline().strip()
#     for line in tqdm.tqdm(fr.readlines()):
#         fname = line.split('\t')[0]
#         old_fp = osp.join(root, fname)
#         new_fp = osp.join(root, dev_dir, fname)
#         shutil.move(old_fp, new_fp)



