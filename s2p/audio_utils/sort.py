import os
import sys


lines = []
for line in sys.stdin:
    line = line.strip()
    lines.append(line)
lines.sort(key=lambda l: int(l.split('\t')[0].split('.')[0].split('_')[-1]))
for line in lines:
    print(line)