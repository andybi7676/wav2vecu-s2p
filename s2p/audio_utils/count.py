txt = "/work/b07502072/corpus/u-s2s/audio/cv4/fr/w2v_manifest/wo_sil/train_70h/trans/train.words.txt"
with open(txt, 'r') as fr:
    c1 = 0
    c2 = 0
    sent_l = 0
    for l in fr:
        for c in l:
            if c == '’':
                c1 += 1
            if c == '\'':
                c2 += 1
        sent_l += 1
    print(c1, c2, sent_l)
