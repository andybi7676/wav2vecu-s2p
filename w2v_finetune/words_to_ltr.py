import os

split = 'test'
with open(f"./data/cv4_de/ltr/src/{split}.words.txt", 'r') as fr, open(f"./data/cv4_de/ltr/{split}.ltr", 'w') as fw:
    for line in fr:
        ltrs = line.strip().replace(" ", "|")
        new_line = ' '.join(list(ltrs)) + " |\n"
        fw.write(new_line)

    
