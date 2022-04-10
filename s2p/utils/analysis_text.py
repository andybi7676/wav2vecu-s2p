import os

f_path = '/work/b07502072/corpus/u-s2s/text/voxpopuli_trans/asr_train.txt'
f_path = '/work/b07502072/corpus/u-s2s/text/wiki/en/wiki_2_7.txt'
f_path = '/work/b07502072/corpus/u-s2s/text/wiki/en/wiki_1_7/wiki_1_7.txt'

with open(f_path, 'r') as fr:
    total_line = 0
    words_counts = []
    for line in fr:
        total_line += 1
        words_counts.append(len(line.split(' ')))
    total_words = sum(words_counts)
    mean = total_words / total_line
    std = (sum([(wc-mean)**2 for wc in words_counts]) / total_line)**0.5

    print(f"file name: {f_path}")
    print(f"total lines: {total_line}")
    print(f"mean of words: {mean}")
    print(f"std of words: {std}")
