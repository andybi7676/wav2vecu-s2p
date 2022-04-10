from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer import phonemize
import os
import argparse

# to see support langs: 
# print(EspeakBackend.supported_languages())
def main():
    parser = get_parser()
    args = parser.parse_args()

    lang = args.lang
    root = args.root
    sep = Separator(phone=' ', syllable='', word='')

    word_path = os.path.join(root, "words.txt")
    out_path = os.path.join(root, "phones.txt")

    phones = []
    with open(word_path, 'r') as fr:
        words = [line.strip() for line in fr]
        phones = [ ph.strip() for ph in phonemize(words, language=lang, separator=sep)]

    with open(out_path, 'w') as fw:
        for ph in phones:
            fw.write(ph+'\n')

def get_parser():
    parser = argparse.ArgumentParser(
        description="turn words into espeak phones. "
    )
    # fmt: off
    parser.add_argument('root', help='root dir of input words.txt and phones.txt')
    parser.add_argument('--lang', help='language to be converted.', default='en-us')

    return parser

if __name__ == "__main__":
    main()