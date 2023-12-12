#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import random
import subprocess
import tqdm

import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tsv", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--src", default=".", type=str, metavar="DIR", help="audio files directory"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--output_fname", default="train", type=str, help="Enter your expected output file name."
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser


def main(args):

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    
    output_fname = args.output_fname
    tsv_path = os.path.realpath(args.tsv)

    dir_path = os.path.realpath(args.src)
    fnames = []
    assert args.ext != 'mp3', 'please convert .mp3 to .wav or .flac first.'
    with open(tsv_path, 'r') as tsv_fr:
        lines = tsv_fr.readlines()
        for line in lines[1:]:
            fname = line.split('\t')[0]
            fname = fname.split('.')[0] + '.wav'
            fname = os.path.join(dir_path, fname)
            fnames.append(fname)

    invalid_f = open(os.path.join(args.dest, f"{output_fname}_invalid.log"), "w")

    with open(os.path.join(args.dest, f"{output_fname}.tsv"), "w") as train_f:
        print(dir_path, file=train_f)

        for fname in tqdm.tqdm(fnames):
            # fname = os.path.join(dir_path, fname)
            file_path = os.path.realpath(fname)

            if args.path_must_contain and args.path_must_contain not in file_path:
                continue
            try:
                frames = soundfile.info(fname).frames
                dest = train_f
                print(
                    "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=dest
                )
            except:
                print(fname, file=invalid_f)
                print(f"Invalid fname: {fname}!")
    invalid_f.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)