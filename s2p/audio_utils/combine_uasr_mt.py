import regex
import sys
import os
import os.path as osp
import pandas as pd
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer import phonemize
from collections import defaultdict
from g2p_en import G2p
import argparse
import tqdm
import pickle
import json

# sep = Separator(phone=' ', syllable='', word='')
# to see support langs: 
# print(EspeakBackend.supported_languages())
def get_parser():
    parser = argparse.ArgumentParser(
        description="turn words into espeak phones. "
    )
    # fmt: off
    parser.add_argument('--uasr_res_fpath', '-a', required=True, help='uasr results file path')
    parser.add_argument('--lang', '-l', help='language to be converted.', default='en-us')
    parser.add_argument('--split', '-s', help='split.', default='asr_test')
    parser.add_argument('--delimiter', '-d', help='delimiter for asr_res_file', default='\t')
    parser.add_argument('--phonemize', '-p', action='store_true', default=True, help='whether to phonemize translation')
    parser.add_argument('--mt_hyp_fpath', '-mh', required=True, default='', help='the output dir you want.')
    parser.add_argument('--mt_ref_fpath', '-mr', required=True, default='', help='the output file name you want.')
    parser.add_argument('--out_dir', '-o', default='./', help='the output file name you want.')

    return parser

def get_g2p_phones(sents, compact=True, only_phonemes=True):
    wrd_to_phn = defaultdict(lambda: False)
    g2p = G2p()
    phn_sents = []
    valid_phonemes = [p for p in g2p.phonemes]
    for p in g2p.phonemes:
        if p[-1].isnumeric():
            valid_phonemes.append(p[:-1])
    for line in sents:
        words = line.strip().split()
        phones = []
        for w in words:
            if not wrd_to_phn[w]:
                phns = g2p(w)
                if compact:
                    phns = [
                        p[:-1] if p[-1].isnumeric() else p for p in phns
                    ]
                if only_phonemes:
                    phns = list(filter(lambda x: x in valid_phonemes, phns))
                    # print(phns)
                wrd_to_phn[w] = phns
            phones.extend(wrd_to_phn[w])
        try:
            # print(" ".join(phones))
            phn_sents.append(" ".join(phones).replace("  ", " "))
        except:
            print(wrd_to_phn, words, phones, file=sys.stderr)
            raise
    return phn_sents

def normalize_text(line: str) -> str:
    filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")
    line = line.strip()
    line = line.replace('’', '\'')
    line = filter_r.sub(" ", line)
    line = line.replace("-", " ")
    line = " ".join(line.split())
    return line.lower()

def load_uasr_all(uasr_fp, delimiter='\t'):
    assert osp.exists(uasr_fp), f"{uasr_fp}"
    uasr_res = {}
    with open(uasr_fp, 'r') as fr:
        col_names = fr.readline().strip().split(delimiter)
        assert col_names[0] == "fids" and len(col_names) > 1, col_names
        for l in fr:
            items = l.strip().split(delimiter)
            assert len(items) == len(col_names)
            fid = items[0].split('.')[0]
            uasr_res[fid] = "|".join(items[1:])
    return uasr_res, col_names[1:]


def load_mt_fp(mt_fp, skip_first=False, tgt_idx=1, normalize=False):
    assert osp.exists(mt_fp), mt_fp
    mt_res = {}
    with open(mt_fp, 'r') as fr:
        if skip_first: fr.readline()
        for l in fr:
            items = l.strip().split('\t')
            assert len(items) > tgt_idx
            fid = items[0].split('.')[0]
            tgt = items[tgt_idx]
            if normalize:
                tgt = normalize_text(tgt)
            if len(tgt) > 0:
                mt_res[fid] = tgt
    return mt_res


def main():
    parser = get_parser()
    args = parser.parse_args()

    uasr_res_fpath = args.uasr_res_fpath
    lang = args.lang
    split = args.split
    delimiter = args.delimiter
    need_phonemize = args.phonemize
    mt_hyp_fp = args.mt_hyp_fpath
    mt_ref_fp = args.mt_ref_fpath
    out_dir = args.out_dir

    uasr_res_dict, uasr_col_names = load_uasr_all(uasr_res_fpath, delimiter=delimiter)
    print(f"uasr col names: {uasr_col_names}")
    mt_hyps_dict = load_mt_fp(mt_hyp_fp)
    mt_refs_dict = load_mt_fp(mt_ref_fp, skip_first=True, tgt_idx=2, normalize=True)
    uasr_dict_fids = list(uasr_res_dict.keys())
    uasr_dict_fids.sort(key=lambda x: int(x.split('_')[-1]))

    valid_fids = []
    uasr_res_all = []
    mt_words_hyps = []
    mt_words_refs = []
    for fid in uasr_dict_fids:
        uasr_res = uasr_res_dict[fid]
        mt_hyp = mt_hyps_dict.get(fid, "")
        mt_ref = mt_refs_dict.get(fid, "")
        if mt_hyp == "" or mt_ref == "":
            continue
        else:
            valid_fids.append(fid)
            uasr_res_all.append(uasr_res)
            mt_words_hyps.append(mt_hyp)
            mt_words_refs.append(mt_ref)
    
    assert len(valid_fids) == len(uasr_res_all) and len(uasr_res_all) == len(mt_words_hyps) and len(mt_words_hyps) == len(mt_words_refs)
    print(f"Valid rows: {len(valid_fids)}")
    mt_phones_hyps = None
    mt_g2p_hyps = None
    mt_phones_refs = None
    mt_g2p_refs = None
    if need_phonemize:
        sep = Separator(phone=' ', syllable='', word='')
        print("Generating IPA...")
        mt_phones_hyps = [ ph.strip() for ph in phonemize(mt_words_hyps, language=lang, separator=sep, language_switch="remove-flags")]
        mt_phones_hyps = [ ph.replace('  ', ' ') for ph in mt_phones_hyps ]
        mt_phones_refs = [ ph.strip() for ph in phonemize(mt_words_refs, language=lang, separator=sep, language_switch="remove-flags")]
        mt_phones_refs = [ ph.replace('  ', ' ') for ph in mt_phones_refs ]
        print("Generating G2P...")
        mt_g2p_hyps = get_g2p_phones(mt_words_hyps)
        mt_g2p_refs = get_g2p_phones(mt_words_refs)
    assert len(uasr_res_all) == len(mt_phones_hyps) and len(mt_phones_hyps) == len(mt_phones_refs) and len(mt_phones_hyps) == len(mt_g2p_hyps) and len(mt_g2p_hyps) == len(mt_g2p_refs)

    with open(osp.join(out_dir, f"combine_uasr_mt_all.{split}.tsv"), 'w') as fw:
        uasr_col_names_str = "|".join(uasr_col_names)
        if need_phonemize:
            print(f"fids|{uasr_col_names_str}|mt_words_hyp|mt_phones_hyp|mt_g2p_hyp|mt_words_ref|mt_phones_ref|mt_g2p_ref", file=fw)
            for fid, uasr_res, mt_wrd_hyp, mt_phn_hyp, mt_g2p_hyp, mt_wrd_ref, mt_phn_ref, mt_g2p_ref in zip(valid_fids, uasr_res_all, mt_words_hyps, mt_phones_hyps, mt_g2p_hyps, mt_words_refs, mt_phones_refs, mt_g2p_refs):
                print(f"{fid}|{uasr_res}|{mt_wrd_hyp}|{mt_phn_hyp}|{mt_g2p_hyp}|{mt_wrd_ref}|{mt_phn_ref}|{mt_g2p_ref}", file=fw)
        else:
            print(f"fids|{uasr_col_names_str}|mt_words_hyp|mt_words_ref", file=fw)
            for fid, uasr_res, mt_wrd_hyp, mt_wrd_ref in zip(valid_fids, uasr_res_all, mt_words_hyps, mt_words_refs):
                print(f"{fid}|{uasr_res}|{mt_wrd_hyp}|{mt_wrd_ref}", file=fw)




if __name__ == "__main__":
    # test()
    main()