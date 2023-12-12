import os
import os.path as osp
import argparse
import regex

def normalize_text(line: str) -> str:
    filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")
    line = line.strip()
    line = line.replace('’', '\'')
    line = filter_r.sub(" ", line)
    line = line.replace("-", " ")
    line = " ".join(line.split())
    return line.lower()

def load_manifest(manifest_path):
    fids = []
    with open(manifest_path, 'r') as mfr:
        lines = mfr.readlines()
        for line in lines[1:]:
            fids.append(line.split('\t')[0].split('.')[0])
    return fids

def read_file(fpath, skip_first=False):
    with open(fpath, 'r') as fr:
        if skip_first:
            splitted_lines = [line.strip().split() for line in fr]
            lines = [" ".join(sp_line[min(1, len(sp_line)):]) for sp_line in splitted_lines]
        else:
            lines = [line.strip() for line in fr]
    return lines

def parse_file(fpath, ignore_first_row=False, normalize=False, tgt_idx=1):
    res = {}
    with open(fpath, 'r') as fr:
        if ignore_first_row:
            fr.readline()
        for line in fr:
            items = line.strip().split('\t')
            fid = items[0].split('.')[0]
            try:
                target = items[tgt_idx]
            except:
                target = ""
            if normalize:
                target = normalize_text(target)
            res[fid] = target
    return res


def get_parser():
    parser = argparse.ArgumentParser(
        description="turn words into espeak phones. "
    )
    # fmt: off
    parser.add_argument('--tsv_dir', '-t', default='', help='file name of transcriptions')
    parser.add_argument('--split', '-s', default='', help='manifest file for generating phonemized text file')
    parser.add_argument("--mt_res", '-m')
    parser.add_argument("--lang", '-l')
    parser.add_argument("--asr_res", '-a')
    parser.add_argument("--out_dir", '-o', default='')
    parser.add_argument("--normalize", '-n', default=False, action="store_true")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    tsv_dir = args.tsv_dir
    split = args.split
    mt_res = args.mt_res
    asr_res = args.asr_res
    normalize = args.normalize
    out_dir = args.out_dir
    lang = args.lang

    trans_dir = osp.join(tsv_dir, "trans")
    if normalize:
        trans_fpath = osp.join(trans_dir, f"{split}.words.txt")
    else:
        trans_fpath = osp.join(trans_dir, f"{split}.trans.txt")
    tsv_fpath = osp.join(tsv_dir, f"{split}.tsv")
    
    fids = load_manifest(tsv_fpath)
    asr_refs = read_file(trans_fpath)
    asr_hyps = read_file(asr_res, skip_first=True)
    assert len(asr_refs) == len(asr_hyps) and len(fids) == len(asr_hyps)
    mt_refs = parse_file(f"./{lang}/covost2_{lang}-en/covost_v2.{lang}_en.{split}.tsv", ignore_first_row=True, normalize=True, tgt_idx=2)
    mt_hyps = parse_file(mt_res)
    # assert len(mt_refs.keys()) == len(mt_hyps.keys()) and len(fids) == len(mt_hyps.keys())

    out_fpath = osp.join(out_dir, f"combined_asr_mt_result.{split}.txt")
    fids.sort(key=lambda fid: int(fid.split('_')[-1]))
    with open(out_fpath, 'w') as fw:
        print("fid|asr_ref|asr_hyp|mt_ref|mt_hyp", file=fw)
        for idx, fid in enumerate(fids):
            try:
                print(f"{fid}|{asr_refs[idx]}|{asr_hyps[idx]}|{mt_refs[fid]}|{mt_hyps[fid]}", file=fw)
            except:
                print(f"{fid} failed.")



    





if __name__ == "__main__":
    main()