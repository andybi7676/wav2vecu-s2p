import os
from collections import defaultdict


letter_dict = defaultdict(lambda: 0)

with open('/work/b07502072/corpus/u-s2s/text/cv_wiki/de/prep/lm.upper.lid.txt', 'r') as fr:
    max_count = 10000
    count = 0
    for line in fr:
        count += 1
        line = line.strip()
        sents = line.lower().replace(" ", "|")
        for l in sents:
            letter_dict[l] += 1
        if count >= max_count: break

letter_list = []
for k, v in letter_dict.items():
    letter_list.append((k, v))
letter_list.sort(key=lambda x: x[1], reverse=True)
with open('dict.ltr.txt', 'w') as fw:
    for k, v in letter_list:
        fw.write(f"{k} {v}\n")


