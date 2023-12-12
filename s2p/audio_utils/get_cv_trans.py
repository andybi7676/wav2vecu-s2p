#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import regex
import sys


def main():
    filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")

    for i, line in enumerate(sys.stdin):
        if i==0: continue
        line = line.strip()
        trans = line.split('\t')[2]
        trans = filter_r.sub(" ", trans)
        trans = trans.replace("-", " ")
        trans = trans.replace("  ", " ")
        trans = trans.replace("   ", " ")
        trans = " ".join(trans.split())
        print(trans.lower())


if __name__ == "__main__":
    main()
