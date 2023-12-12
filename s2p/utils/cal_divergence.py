import os
import argparse
import math
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

def normalize_npy(input_npy):
    min_val = input_npy.min()
    return (input_npy - min_val) / (input_npy.max() - input_npy.min())

def counts_to_prob(input_npy):
    return input_npy / np.sum(input_npy, axis=-1)

def count_ngram_of_file(fpath, ngram=4, bucket=-1, maxline=500000):
    # count_dicts = []
    keys_dict = {}
    with open(fpath, 'r') as fr:
        lines = fr.readlines()[:maxline]
        bucket = len(lines) if bucket<0 else bucket
        total_buckets = math.ceil(len(lines) / bucket)
        count_dict = defaultdict(lambda: [0 for _ in range(total_buckets)])
        cur_bucket = -1
        for idx, line in tqdm(enumerate(lines), total=len(lines)):
            if idx % bucket == 0:
                cur_bucket += 1
            line = line.replace('<SIL>', '')
            line = line.replace('  ', ' ')
            phns = line.rstrip().split()
            for s in range(len(phns)-(ngram-1)):
                e = s + ngram
                key = tuple(phns[s:e])
                keys_dict[key] = True
                count_dict[key][cur_bucket] += 1
    return count_dict, keys_dict

def cal_JS_div(ref_ngram_counts_npy, hyp_ngram_counts_npy, outdir="", start_idx=0):
    p = counts_to_prob(np.sum(ref_ngram_counts_npy, axis=-1))
    q = counts_to_prob(np.sum(hyp_ngram_counts_npy, axis=-1))
    # print(p, q)
    return (jensenshannon(p, q))**2

def cal_hyp_part_JS_div(ref_ngram_counts_npy, hyp_ngram_counts_npy, outdir="", start_idx=0, *, draw=True):
    p = normalize_npy(np.sum(ref_ngram_counts_npy, axis=-1))
    qs = [normalize_npy(hyp_part_counts_npy) for hyp_part_counts_npy in np.transpose(hyp_ngram_counts_npy)]
    JSs = np.array([(jensenshannon(p, q))**2 for q in qs[:-1]])
    avg = np.mean(JSs)
    std = np.std(JSs)
    if draw:
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, len(JSs)), JSs)
        fig.savefig(f'{outdir}/hyp_part_JS_s-idx={start_idx}', dpi=200)
    return JSs, avg, std

def cal_ref_part_JS_div(ref_ngram_counts_npy, hyp_ngram_counts_npy, outdir="", start_idx=0, *, draw=True, sample=True):
    p = normalize_npy(np.sum(ref_ngram_counts_npy, axis=-1))
    qs_orig = np.transpose(ref_ngram_counts_npy)[:-1]
    if sample and len(qs_orig) > 1000:
        rand_filter = np.random.rand(len(qs_orig)) < 1000 / len(qs_orig)
        qs_orig = qs_orig[rand_filter]
    qs = [normalize_npy(ref_part_counts_npy) for ref_part_counts_npy in qs_orig]
    JSs = np.array([(jensenshannon(p, q))**2 for q in qs])
    avg = np.mean(JSs)
    std = np.std(JSs)
    if draw:
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, len(JSs)), JSs)
        fig.savefig(f'{outdir}/ref_part_JS', dpi=200)
    return JSs, avg, std

def cal_intra_ref_JS_div(ref_ngram_counts_npy, hyp_ngram_counts_npy, outdir="", start_idx=0, *, draw=True, sample=True):
    ps_orig = np.transpose(ref_ngram_counts_npy)[:-1]
    if sample and len(ps_orig) > 100:
        ps_orig = np.concatenate((ps_orig[:50,], ps_orig[-50:,]), axis=0)
    ps = [normalize_npy(ref_part_counts_npy) for ref_part_counts_npy in ps_orig]
    JSgrid = np.array([[(jensenshannon(p, q))**2 for q in ps] for p in ps])
    JSary = JSgrid.reshape(-1)
    avg = np.mean(JSary)
    std = np.std(JSary)
    if draw:
        fig, ax = plt.subplots()
        im = ax.imshow(JSgrid)
        fig.colorbar(im, ax=ax, label='Interactive colorbar')
        fig.savefig(f'{outdir}/intra_ref_JS', dpi=200)
        plt.close()
    return JSgrid, avg, std

def cal_part_JS_div():
    pass

def main(args):
    outdir = osp.join(args.outdir, args.exp, "bucket"+str(args.bucket).replace('-', '_'))
    os.makedirs(outdir, exist_ok=True)
    ref_count_dicts, ref_keys_dict = count_ngram_of_file(args.ref, ngram=args.ngram, bucket=args.bucket)
    hyp_count_dicts, _ = count_ngram_of_file(args.hyp, ngram=args.ngram, bucket=args.bucket)
    ref_keys_npy = np.array([key for key in ref_keys_dict.keys()])
    ref_ngram_counts_buckets_npy = np.array([ref_count_dicts[key] for key in ref_keys_dict.keys()])
    hyp_ngram_counts_buckets_npy = np.array([hyp_count_dicts[key] for key in ref_keys_dict.keys()])
    print("Calculated and bucketed ngram counts")
    ref_ngram_counts_npy = np.sum(ref_ngram_counts_buckets_npy, axis=-1)
    hyp_ngram_counts_npy = np.sum(hyp_ngram_counts_buckets_npy, axis=-1)
    sorted_idx = np.argsort(ref_ngram_counts_npy)
    ref_ngram_counts_sorted_npy = ref_ngram_counts_npy[sorted_idx]
    hyp_ngram_counts_sorted_npy = hyp_ngram_counts_npy[sorted_idx]
    ref_keys_sorted = ref_keys_npy[sorted_idx]
    print(f"{args.ngram}-gram size={len(ref_ngram_counts_sorted_npy)}")
    print(ref_ngram_counts_sorted_npy[-5:], [" ".join(key) for key in ref_keys_sorted[-5:]])
    fig, ax = plt.subplots()
    start_idx = -10000
    ax.plot(np.arange(0, len(hyp_ngram_counts_sorted_npy[start_idx:])), hyp_ngram_counts_sorted_npy[start_idx:] / np.sum(hyp_ngram_counts_sorted_npy[start_idx:]), c="blue")
    ax.plot(np.arange(0, len(ref_ngram_counts_sorted_npy[start_idx:])), ref_ngram_counts_sorted_npy[start_idx:] / np.sum(ref_ngram_counts_sorted_npy[start_idx:]), c="orange")
    fig.savefig(f'{outdir}/counts_s-idx={start_idx}', dpi=200)
    plt.close()
    metric = eval(f"cal_{args.type}_div")
    distance = metric(ref_ngram_counts_buckets_npy[sorted_idx][start_idx:], hyp_ngram_counts_buckets_npy[sorted_idx][start_idx:], outdir, start_idx)
    print(distance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref",
        default="",
        help="ref file path for calculating n-gram",
    )
    parser.add_argument(
        "--hyp",
        default="",
        help="hyp file path for calculating n-gram",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=4,
        help="determine ngram",
    )
    parser.add_argument(
        "--bucket",
        type=int,
        default=-1,
        help="determine ngram",
    )
    parser.add_argument(
        "--type",
        default="JS",
        help="choose a evaluate metric",
    )
    parser.add_argument(
        "--outdir",
        default="./analysis",
        help="output directory for the analysis",
    )
    parser.add_argument(
        "--exp",
        default="lang/gp2.0/seed1",
        help="output directory for the analysis",
    )
    
    args = parser.parse_args()

    main(args)