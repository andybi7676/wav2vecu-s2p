import kenlm
import os
import os.path as osp
import editdistance
import math
import tqdm
import argparse
import numpy as np

def parse_dir(d):
    aw, bw = d.split('/')[-1].split('_')[-2:]
    return float(aw), float(bw)

def get_entropies(f_path, kenlm):
    lengths_t = 0
    sent_t = 0
    lm_score = 0
    lm_entropies = []
    with open(f_path, 'r') as fr:
        for line in fr:
            line.replace("<SIL>", "")
            line.replace("  ", " ")
            line = line.strip()
            words = line.split()
            lengths_t += len(words)
            lm_score = kenlm.score(line)
            sent_t += 1
            lm_entropy =  -lm_score / (len(words) + 2)
            lm_entropies.append(lm_entropy)
    return np.array(lm_entropies)

def read_file(fpath):
    lines = []
    with open(fpath, 'r') as fr:
        for line in fr:
            line = line.strip()
            lines.append(line)
    return lines

def get_vt_diffs(hyp_fpath, vit_trans):
    hyps = read_file(hyp_fpath)
    vt_errs = np.array([
        editdistance.eval(vt, h) for vt, h in zip(vit_trans, hyps)
    ])

    vt_lengths = np.array([ len(vt.split()) for vt in vit_trans ])
    return vt_errs / vt_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_root", "-r", 
        help="the root dir of kaldi decoding",
        default=False
    )
    parser.add_argument(
        "--training_data_dir", '-t',
        help="for calculating min ppl"
    )
    parser.add_argument(
        "--viterbi_fpath", "-v",
        help="The viterbi transcript",
    )
    parser.add_argument(
        "--kenlm", "-k",
        help="path of kenlm",
        default="Use the 4-gram in training_data_dir"
    )
    parser.add_argument(
        "--min_vit_uer", "-u", default=0.03,
        type=float,
        help="the min vt_uer for u-tuning"
    )
    parser.add_argument(
        "--subset", "-s",
        default="valid_small"
    )
    args = parser.parse_args()

    save_root = args.save_root
    vit_fpath = args.viterbi_fpath
    kenlm_path = args.kenlm
    train_data_dir = args.training_data_dir
    subset = args.subset
    min_vit_uer = args.min_vit_uer
    if kenlm_path == "Use the 4-gram in training_data_dir":
        kenlm_path = osp.join(train_data_dir, "lm.phones.filtered.04.bin")
    lm_train_data = osp.join(train_data_dir, "lm.phones.filtered.txt")

    print("Loading kenlm......")
    kenlm_model = kenlm.Model(kenlm_path)
    print("Kenlm loaded.")


    if lm_train_data:
        print("Getting training lm_ppl......")
        min_lm_ppl = 7.884893153191544
        min_lm_entropy = math.log(min_lm_ppl)
        print(f"training_lm_ppl={min_lm_ppl}")
    if not save_root: return

    vit_trans = read_file(vit_fpath)
    best_score_params = None
    best_score = float("inf")
    uer = float("inf")
    ppl = float("inf")
    ppl_dict = {}
    uer_dict = {}
    score_dict = {}
    for root, dirs, files in os.walk(osp.join(save_root, "details")):
        if len(dirs) == 0: continue
        for d in tqdm.tqdm(dirs):
            aw, bw = parse_dir(d)
            hyp_fpath = osp.join(root, d, f"{subset}.txt")
            lm_entropies = get_entropies(hyp_fpath, kenlm_model)
            # print(lm_entropies)
            lm_ppl = math.pow(10, lm_entropies.mean())
            ppl_dict[(aw, bw)] = lm_ppl
            uer_dict[(aw, bw)] = lm_ppl
            vt_diffs = get_vt_diffs(hyp_fpath, vit_trans)
            # weighted_score = lm_entropies * max(vt_diff, min_vit_uer)
            weighted_score = sum([max(lm_entorpy, min_lm_entropy) * max(vt_diff, min_vit_uer) for lm_entorpy, vt_diff in zip(lm_entropies, vt_diffs)])
            score_dict[(aw, bw)] = weighted_score
            if weighted_score <= best_score:
                best_score = weighted_score
                best_score_params = (aw, bw)
    
    with open(osp.join(save_root, "best_unsup_score.txt"), 'w') as bfw:
        weights = list(score_dict.keys())
        weights.sort()
        last_aw = weights[0][0]
        print("Score matrix:", file=bfw)
        for aw, bw in weights:
            if aw != last_aw:
                print("\n", file=bfw)
                last_aw = aw
            # else:
            score = f"{score_dict[(aw, bw)]:3.3f}"
            if score == "inf":
                score = "*inf*"
            print(f"{score}", file=bfw, end='\t')
        print("\n\nResults:", file=bfw)
        print(f"Best unsup score: {best_score_params}, score={best_score}", file=bfw)

if __name__ == "__main__":
    main()