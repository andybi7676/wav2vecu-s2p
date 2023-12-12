import editdistance
import os
import sys
from sacrebleu.metrics import BLEU
bleu_metric = BLEU()

err_t = 0
ref_length_t = 0
sep = "\t"
ref_idx = 1
hyp_idx = 2
refs = []
hyps = []
for i, line in enumerate(sys.stdin):
    if i == 0: continue
    items = line.strip().split(sep)
    try:
        hyp = items[hyp_idx]
    except:
        hyp = ""
    ref = items[ref_idx]
    refs.append(ref)
    hyps.append(hyp)
corpus_bleu = bleu_metric.corpus_score(hyps, [refs])

# print(f"WER={wer*100:.5f}%")
print(f"bleu={corpus_bleu}")