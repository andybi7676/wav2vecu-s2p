import os
import os.path as osp
from collections import defaultdict

src_dir = "./src"
splits = ["train", "valid", "asr_test"]
phns_dict = defaultdict(lambda : 0)
for split in splits:
    src_fpath = osp.join(src_dir, f"{split}.hmm.phones.txt")
    with open(src_fpath, 'r') as fr, open(f"./{split}.ltr", 'w') as fw:
        for line in fr:
            items = line.strip().split()
            if len(items) > 1:
                for phn in items[1:]:
                    phns_dict[phn] += 1
                phns = " ".join(items[1:]) + " |\n"
            else:
                phns = " |\n"
            phns_dict["|"] += 1
            # ltrs = phns.strip().replace(" ", "|")
            # new_line = ' '.join(list(phns)) + " |\n"
            fw.write(phns)

with open("./dict.ltr.txt", 'w') as fw:
    dict_list = list(phns_dict.items())
    dict_list.sort(key=lambda x: x[1], reverse=True)
    for k, v in dict_list:
        print(f"{k} {v}", file=fw)
    