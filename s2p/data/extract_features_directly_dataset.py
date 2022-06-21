# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import os.path as osp
import contextlib

import numpy as np
import sys
import torch
import torch.nn.functional as F
import faiss

import soundfile as sf
import fairseq
import pandas as pd
from fairseq.data import FairseqDataset, data_utils
import random
import math
from extracted_features_dataset import ExtractedFeaturesDataset

logger = logging.getLogger(__name__)

class Wav2VecFeatureReader(object):
    def __init__(self, cp_file, layer):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file]
        )
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model
        self.task = task
        self.layer = layer

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            m_res = self.model(source=source, mask=False, features_only=True, layer=self.layer)
            return m_res["x"].squeeze(0)

# This dataset is for evaluation only.
class ExtractFeaturesDirectlyDataset(FairseqDataset):
    def __init__(
        self,
        path,
        split,
        min_length=3,
        max_length=None,
        labels=None,
        label_dict=None,
        shuffle=True,
        sort_by_length=True,
        checkpoint=None,
        layer=14,
        centroids_path='CLUS128',
        pca_path='pca',
        pca_dim=512,
        merge_cluster=True,
        adjacent_pooling=True,
        pooling='mean',
        subsample_rate=0.5,
        remove_extra=False,
    ):
        super().__init__()

        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.label_dict = label_dict

        if labels is not None:
            assert label_dict is not None
        
        self.sizes = []
        self.offsets = []
        self.labels = []

        data_path = os.path.join(path, split)
        # self.data = np.load(data_path + ".npy", mmap_mode="r")
        # obtain all files' paths
        self.files = []
        with open(data_path + ".tsv", 'r') as fr:
            dir_root = fr.readline().strip()
            for l in fr.readlines():
                f_path = l.split('\t')[0].strip()
                self.files.append(osp.join(dir_root, f_path))
        # print(self.files[:1])
        self.reader = Wav2VecFeatureReader(checkpoint, layer)
        self.centroids_path = osp.join(path, centroids_path)
        self.centroids = np.load(osp.join(self.centroids_path, "centroids.npy"))
        # prepare to load centroids
        # print("Loaded centroids", self.centroids.shape, file=sys.stderr)
        # print(self.centroids)
        # return
        res = faiss.StandardGpuResources()
        index_flat = (
            faiss.IndexFlatL2(self.centroids.shape[1])
            # faiss.IndexFlatIP(self.centroids.shape[1])
        )
        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        self.faiss_index.add(self.centroids)
        self.pca_path = osp.join(path, pca_path)
        self.pca_A = torch.from_numpy(np.load(self.pca_path + f"/{pca_dim}_pca_A.npy")).cuda()
        self.pca_b = torch.from_numpy(np.load(self.pca_path + f"/{pca_dim}_pca_b.npy")).cuda()
        self.pooling = pooling
        self.subsample_rate = subsample_rate
        self.remove_extra = remove_extra
        
        offset = 0
        skipped = 0

        if not osp.exists(path + f".{labels}"):
            labels = None

        # with open(data_path + ".lengths", "r") as len_f, open(
        #     path + f".{labels}", "r"
        # ) if labels is not None else contextlib.ExitStack() as lbl_f:
        #     for line in len_f:
        #         length = int(line.rstrip())
        #         lbl = None if labels is None else next(lbl_f).rstrip().split()
        #         if length >= min_length and (
        #             max_length is None or length <= max_length
        #         ):
        #             self.sizes.append(length)
        #             self.offsets.append(offset)
        #             if lbl is not None:
        #                 self.labels.append(lbl)
        #         offset += length

        # self.sizes = np.asarray(self.sizes)
        # self.offsets = np.asarray(self.offsets)

        logger.info(f"loaded {len(self.files)}, skipped {skipped} samples")
    
    def merge(self, feats, clust):
        # feats = torch.from_numpy(feats.copy())
        clust = torch.LongTensor(clust).cuda()
        _, counts = clust.unique_consecutive(return_counts=True)
        curr = 0

        merged = []
        for c in counts:
            c = c.item()
            start = curr
            end = curr + c
            curr += c
            if self.pooling == "mean":
                new_x = feats[start:end].mean(dim=0)
            elif self.pooling == "sample":
                new_x = feats[start + int(random.random() * c)]
            else:
                raise NotImplementedError()
            merged.append(new_x)

        return torch.stack(merged, dim=0)

    def __getitem__(self, index):
        fname = self.files[index]
        f = self.reader.get_feats(fname)

        _, z = self.faiss_index.search(f.cpu().numpy(), 1)
        # print(z.shape)
        with torch.no_grad():
            x = torch.matmul(f, self.pca_A) + self.pca_b
        # print(x.shape)
        
        m = self.merge(x, z)
        # print(m.shape)

        fsz = m.shape[-1]
        length = len(m)
        target_num = math.ceil(length * self.subsample_rate)
        rem = length % target_num
        if rem > 0:
            if self.remove_extra:
                to_rem = target_num - rem
                target_num -= 1
                m = m[:-to_rem]
            else:
                to_add = target_num - rem
                m = F.pad(m, [0, 0, 0, to_add])
                m[-to_add:] = m[-to_add - 1]

        m = m.view(target_num, -1, fsz)
        feats = m.mean(dim=-2)
        # print(feats.shape)
        # print(feats)



        # print(" ".join(str(x.item()) for x in z), file=fp)

        # offset = self.offsets[index]
        # end = self.sizes[index] + offset
        # feats = torch.from_numpy(self.data[offset:end].copy()).float()

        res = {"id": index, "features": feats}
        # if len(self.labels) > 0:
        #     res["target"] = self.label_dict.encode_line(
        #         self.labels[index],
        #         line_tokenizer=lambda x: x,
        #         append_eos=False,
        #     )

        return res

    def __len__(self):
        return len(self.files)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        features = [s["features"] for s in samples]
        sizes = [len(s) for s in features]

        target_size = max(sizes)

        collated_features = features[0].new_zeros(
            len(features), target_size, features[0].size(-1)
        )
        padding_mask = torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
        for i, (f, size) in enumerate(zip(features, sizes)):
            collated_features[i, :size] = f
            padding_mask[i, size:] = True

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {"features": collated_features, "padding_mask": padding_mask},
        }

        if len(self.labels) > 0:
            target = data_utils.collate_tokens(
                [s["target"] for s in samples],
                pad_idx=self.label_dict.pad(),
                left_pad=False,
            )
            res["target"] = target
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        if self.sort_by_length:
            order.append(self.sizes)
            return np.lexsort(order)[::-1]
        else:
            return order[0]


if __name__ == '__main__':
    path = '/work/b07502072/corpus/u-s2s/audio/en_feats/voxpopuli/large_noisy_new'
    dataset = ExtractFeaturesDirectlyDataset(
        path,
        'train',
        shuffle=False,
        checkpoint='/work/b07502072/pretrained_models/w2v_large_lv_fsh_swbd_cv.pt',
    )
    feats = dataset.__getitem__(0)
    # loaded_feats = np.load(osp.join(path, "train.npy"))[:1430]
    # print(loaded_feats)
    
    print(feats)

    from_load_dataset = ExtractedFeaturesDataset(
        os.path.join(path, 'precompute_pca512_cls128_mean_pooled'), 
        'train',
        shuffle=False
    )

    feats_from_load = from_load_dataset.__getitem__(0)
    print(feats_from_load)