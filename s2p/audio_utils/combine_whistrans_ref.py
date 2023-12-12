import os
import os.path as osp
import sys
import argparse
import regex
from lhz.text.normalizer import filter_and_normalize

main_delimiter = '\t'
tgt_delimiter = ','
def get_parser():
    parser = argparse.ArgumentParser(
        description="combine new columns into previous combined file"
    )
    # fmt: off
    parser.add_argument('--main_fpath', '-m', required=True, help='uasr results root directory')
    parser.add_argument('--new_col_fpath', '-n', required=True, default='./trans', help='the output dir you want.')
    parser.add_argument('--new_col_name', '-c', required=True, default='new_content', help='the output dir you want.')
    parser.add_argument('--out_dir', '-o', default='./', help='the output file name you want.')
    parser.add_argument('--out_fname', '-f', default='new_combined', help='the output file name you want.')
    parser.add_argument("--normalize", default=False, action="store_true")
    parser.add_argument("--skip_first", default=False, action="store_true")
    parser.add_argument("--tgt_idx", default=1, type=int)
    # parser.add_argument('--delimiter', '-d', default='|', help='the output file name you want.')

    return parser


def normalize_text(line: str) -> str:
    filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")
    line = line.strip()
    line = line.lower()
    line = line.replace('’', '\'')
    line = filter_r.sub(" ", line)
    line = line.replace("-", " ")
    line = " ".join(line.split())
    return line.lower()

def get_main_content(main_fp):
    assert osp.exists(main_fp), f"{main_fp}"
    main_res = {}
    with open(main_fp, 'r') as fr:
        col_names = fr.readline().strip().split(main_delimiter)
        print(col_names)
        assert col_names[0] == "fids" and len(col_names) > 1
        for l in fr:
            items = l.strip().split(main_delimiter)
            assert len(items) == len(col_names)
            fid = items[0].split('.')[0]
            main_res[fid] = main_delimiter.join(items[1:])
    return main_res, col_names

def get_new_content(new_fp, skip_first=False, tgt_idx=1, normalize=False):
    assert osp.exists(new_fp)
    new_res = {}
    with open(new_fp, 'r') as fr:
        if skip_first: fr.readline()
        for l in fr:
            items = l.strip().split(tgt_delimiter)
            if len(items) > tgt_idx:
                fid = items[0].strip().split('/')[-1].split('.')[0]
                tgt = items[tgt_idx].strip()
            else:
                tgt = ""
            if normalize:
                tgt = filter_and_normalize(tgt)
            # if len(tgt) > 0:
            new_res[fid] = tgt
    return new_res

def main():
    parser = get_parser()
    args = parser.parse_args()

    main_fp = args.main_fpath
    new_col_fp = args.new_col_fpath
    new_col_name = args.new_col_name
    out_dir = args.out_dir
    out_fname = args.out_fname
    normalize = args.normalize
    skip_first = args.skip_first
    tgt_idx = args.tgt_idx

    main_res, col_names = get_main_content(main_fp)
    new_res = get_new_content(new_col_fp, skip_first=skip_first, tgt_idx=tgt_idx, normalize=normalize)

    valid_fids = list(main_res.keys())
    valid_fids.sort(key=lambda fid: int(fid.split('_')[-1]))

    with open(osp.join(out_dir, f"{out_fname}.tsv"), 'w') as fw:
        new_col_names = main_delimiter.join(col_names) + f"{main_delimiter}{new_col_name}"
        print(new_col_names, file=fw) 
        for fid in valid_fids:
            new_item = new_res.get(fid, False)
            if new_item != False:
                new_line = f"{fid}{main_delimiter}{main_res[fid]}{main_delimiter}{new_item}"
                if f"{main_delimiter}{main_delimiter}" not in new_line:
                    # print(new_line)
                    print(new_line, file=fw)

if __name__ == "__main__":
    main()

