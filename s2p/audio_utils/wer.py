import editdistance
import os
import sys
import regex

wrd_err_t = 0
ref_wrd_length_t = 0
normalize = True

def normalize_text(line: str) -> str:
    filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")
    line = line.strip()
    line = line.replace('’', '\'')
    line = filter_r.sub(" ", line)
    line = line.replace("-", " ")
    line = " ".join(line.split())
    return line.lower()

hyp_fpath = "/work/b07502072/corpus/u-s2s/audio/cv4/de/whisper_on_cvss/whisper_hyp.test.sorted.txt"
ref_fpath = "/work/b07502072/corpus/u-s2s/audio/cv4/de/whisper_on_cvss/cvss_ref.test.txt"

with open(hyp_fpath, 'r') as hyp_fr, open(ref_fpath, 'r') as ref_fr:
    hyps = hyp_fr.readlines()
    refs = ref_fr.readlines()
    for hyp, ref in zip(hyps, refs):
        # if i == 0: continue
        hyp = normalize_text(hyp.split('\t')[1])
        ref = normalize_text(ref.split('\t')[1])
        wrd_err_t += editdistance.eval(ref.split(), hyp.split())
        ref_wrd_length_t += len(ref.split())
wer = wrd_err_t / ref_wrd_length_t
print(f"WER={wer*100:.5f}%")