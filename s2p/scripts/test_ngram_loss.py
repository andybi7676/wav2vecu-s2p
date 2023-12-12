import os
import os.path as osp
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time

def main(args):
    # print(args)
    raw_dense_x = torch.randn(160, 100, 50)
    raw_dense_x.requires_grad = True
    orig_dense_x = raw_dense_x[:,10:40,1:]
    print(orig_dense_x.is_contiguous())
    dense_x = F.gumbel_softmax(orig_dense_x, tau=1, hard=False)
    # print(dense_x)
    print(dense_x.is_contiguous())
    # ngram_dims = [(0, 2, 1), (0, 1, 4), (2, 3, 1)]
    ngram_dims = np.random.randint(45, size=(10000, 4))
    st = time.perf_counter()
    for nd in ngram_dims:
        counts = (dense_x[:,0:-3,nd[0]] * dense_x[:,1:-2,nd[1]] * dense_x[:,2:-1,nd[2]] * dense_x[:,3:,nd[3]]).sum()
    et = time.perf_counter()
    print(et-st)
    # print(out_0_2_1)
    # print(dense_x[:,0:-2,0].is_contiguous())
    # out_0_2_1.backward()
    # print(raw_dense_x.grad)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="",
        help="a sample arg",
    )
    args = parser.parse_args()

    main(args)