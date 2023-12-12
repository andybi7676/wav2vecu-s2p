import regex
import sys
import os
import os.path as osp
import pandas as pd
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer import phonemize
from lhz.text.normalizer import filter_and_normalize
import argparse
import tqdm
import pickle
import json

sep = Separator(phone=' ', syllable='', word='')
# to see support langs: 
# print(EspeakBackend.supported_languages())
def get_parser():
    parser = argparse.ArgumentParser(
        description="turn words into espeak phones. "
    )
    # fmt: off
    parser.add_argument('--manifest', '-m', default='', help='manifest file for generating phonemized text file')
    parser.add_argument('--lang', help='language to be converted.', default='de')
    parser.add_argument('--phonemize', '-p', default=False, help='whether to phonemize trans by a manifest-specified order.')
    parser.add_argument('--outdir', '-o', default='./trans', help='the output dir you want.')
    parser.add_argument('--outfname', '-f', default='', help='the output file name you want.')
    parser.add_argument('--post_process', action="store_true", default=False)

    return parser

def post_process_fr_fr(phones):
    replace_dict = {
        # 'ɑ̃': 'ɔ̃'
    }
    reduced_set = [chr(720), chr(771)]
    
    def _replace(phns):
        for k, v in replace_dict.items():
            phns = phns.replace(k, v)
        return phns
    
    def _reduce(phns):
        for re in reduced_set:
            phns = phns.replace(re, "")
        return phns

    new_phones = [_reduce(_replace(phns)) for phns in phones]
    return new_phones

def normalize_text(line: str) -> str:
    line = filter_and_normalize(line)
    line = line.replace('\'', '')
    return line

def phonemize_text(words: str, lang='de'):
    ph = phonemize(words, language=lang, separator=sep)
    return ph.strip().replace('  ', ' ')

def load_trans(manifest_path, mapping=False):
    if mapping:
        trans_data = {}
    else:
        trans_data = []
    with open(manifest_path, 'r') as manifest_fr:
        lines = manifest_fr.readlines()
        root = lines[0].strip()
        root = "/work/c/LibriTTS/valid"
        for line in tqdm.tqdm(lines[1:]):
            items = line.strip().split('\t')
            fpath = items[0].split('.')[0]
            with open(osp.join(root, f"{fpath}.normalized.txt"), 'r') as trans_fr:
                trans = trans_fr.readlines()[0].strip()
            normalized_trans = normalize_text(trans)
            new_trans_dict = {
                'trans': trans,
                'words': normalized_trans
            }
            if mapping:
                trans_data[fpath] = new_trans_dict
            else:
                trans_data.append(new_trans_dict)
    # json.dump(trans_data, open('trans_data.tmp.json', 'w'), indent=4)
    return trans_data

def load_manifest(manifest_path):
    fids = []
    with open(manifest_path, 'r') as mfr:
        lines = mfr.readlines()
        for line in lines[1:]:
            fids.append(line.strip().split('\t')[0].split('.')[0])
    return fids

def main():
    parser = get_parser()
    args = parser.parse_args()

    outdir = args.outdir
    get_phones = args.phonemize
    lang = args.lang
    manifest_path = args.manifest
    post_process = args.post_process

    trans_data = load_trans(manifest_path, mapping=get_phones)
    if get_phones:
        sep = Separator(phone=' ', syllable='', word='')
        valid_fids = list(trans_data.keys())
        fids = load_manifest(manifest_path)
    
    if args.outfname == '':
        if manifest_path == '':
            trans_fname = args.trans.split('/')[-1].split('.')[0]
        else:
            trans_fname = manifest_path.split('/')[-1].split('.')[0]
    else:
        trans_fname = args.outfname
    os.makedirs(outdir, exist_ok=True)
    transfw = open(f"{outdir}/{trans_fname}.trans.txt", 'w')
    wordsfw = open(f"{outdir}/{trans_fname}.words.txt", 'w')
    phonesfw = open(f"{outdir}/{trans_fname}.phones.txt", 'w')

    words = []
    phones = []
    if get_phones:
        for fid in tqdm.tqdm(fids):
            assert fid in valid_fids, f'found invalid file id: {fid}'
            trans = trans_data[fid]
            words.append(trans['words'])
            transfw.write(trans['trans'] + '\n')
            wordsfw.write(trans['words'] + '\n')
    else:
        for trans in tqdm.tqdm(trans_data):
            transfw.write(trans['trans'] + '\n')
            wordsfw.write(trans['words'] + '\n')
    if get_phones:
        phones = [ ph.strip() for ph in phonemize(words, language=lang, separator=sep, language_switch="remove-flags")]
        phones = [ ph.replace('  ', ' ') for ph in phones ]
        if post_process:
            post_process_code = lang.replace('-', '_')
            post_process_fn = eval(f"post_process_{post_process_code}")
            phones = post_process_fn(phones)
        for ph in phones:
            phonesfw.write(ph + '\n')
        phonesfw.close()
    transfw.close()
    wordsfw.close()

def test():
    test = normalize_text("as regards the private teaching of prisoners; and they went on to say that \"a resolved adherence, in spite of discouragements the most disheartening,")
    test_ph = phonemize_text(test)
    print(test)
    print(test_ph)

if __name__ == "__main__":
    # test()
    main()