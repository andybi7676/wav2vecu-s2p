import editdistance
import os
import sys

err_t = 0
ref_length_t = 0
sep = "|"
ref_idx = 10
hyp_idx = 8
for i, line in enumerate(sys.stdin):
    if i == 0: continue
    items = line.strip().split(sep)
    hyp = items[hyp_idx]
    ref = items[ref_idx]

    err_t += editdistance.eval(ref.split(), hyp.split())
    ref_length_t += len(ref.split())
wer = err_t / ref_length_t
print(f"WER={wer*100:.5f}%")