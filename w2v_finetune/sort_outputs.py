import sys

lines = []
for line in sys.stdin:
    lines.append(line.strip())

lines.sort(key=lambda l: int(l.split("(None-")[-1].split(')')[0]))

for new_line in lines:
    print(new_line.split('(')[0].strip())