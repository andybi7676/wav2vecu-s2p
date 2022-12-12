from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer import phonemize
import os
import argparse

def post_process_fr_fr(phones):
    replace_dict = {
        'ɑ̃': 'ɔ̃'
    }
    reduced_set = [chr(720), chr(771)] # 'ː', '̃ '
    
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

     
# to see support langs: 
# print(EspeakBackend.supported_languages())
def main():
    parser = get_parser()
    args = parser.parse_args()

    lang = args.lang
    root = args.root
    fname = args.fname
    post_process = args.post_process
    sep = Separator(phone=' ', syllable='', word='')
    if fname != '':
        word_path = os.path.join(root, fname+".words.txt")
        out_path = os.path.join(root, fname+".phones.txt")
    else:
        word_path = os.path.join(root, "words.txt")
        out_path = os.path.join(root, "phones.txt")

    phones = []
    words = []
    with open(word_path, 'r') as fr:
        for line in fr:
            words.append(line.strip())
        phones = [ ph.strip() for ph in phonemize(words, language=lang, separator=sep, language_switch="remove-flags")]
        phones = [ph.replace('  ', ' ') for ph in phones]
        if post_process:
            post_process_code = lang.replace('-', '_')
            post_process_fn = eval(f"post_process_{post_process_code}")
            phones = post_process_fn(phones)

    with open(out_path, 'w') as fw:
        for phn in phones:
            fw.write(f"{phn}\n")

def get_parser():
    parser = argparse.ArgumentParser(
        description="turn words into espeak phones. "
    )
    # fmt: off
    parser.add_argument('root', help='root dir of input words.txt and phones.txt')
    parser.add_argument('--fname', default='', help='file name of input words.txt and phones.txt')
    parser.add_argument('--lang', help='language to be converted.', default='en-us')
    parser.add_argument('--post_process', '-p', action="store_true", default=False)

    return parser

if __name__ == "__main__":
    main()