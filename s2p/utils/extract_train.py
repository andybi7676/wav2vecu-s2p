import os
import numpy as np

in_file = './train.npy'
length_file = './train.lengths'
out_file = './train_smaller.npy'
ratio = 0.3

train = np.load(in_file)

total_l = train.shape[0]
ratio_l = round(total_l*ratio)
print(f"total_length: {total_l}, ratio_length: {ratio_l}")
out_l = 0
l_count = 0
with open(length_file, 'r') as fr_l:
    for line in fr_l:
        out_l += int(line.strip())
        l_count += 1
        if out_l > ratio_l: break

print(f"out_length: {out_l}, line_count: {l_count}")
train_smaller = train[:out_l]
np.save(out_file, train_smaller)
