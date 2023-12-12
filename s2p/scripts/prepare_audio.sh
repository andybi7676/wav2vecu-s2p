#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
sleep 120m
python scripts/mp_rm_sil.py --tsv /work/c/LibriTTS/sr_16k/manifest/train.tsv --vads /work/c/LibriTTS/sr_16k/manifest/train_1-54403.vads --out /work/c/LibriTTS/rm_sil/train
python /home/andybi7676/Desktop/wav2vecu-s2p/fairseq/examples/wav2vec/wav2vec_manifest.py /work/c/LibriTTS/rm_sil/train --valid-percent 0.0 --output-fname train --dest /work/c/LibriTTS/rm_sil/manifest --ext wav
set -e
set -u
set -o pipefail

source_dir=$1
tgt_dir=$2
model=$3

FAIRSEQ_ROOT=~/Desktop/wav2vecu-s2p/fairseq

# if [ -z "$4" ]
#   then
#     dim=512
#   else
#     dim=$4
# fi
dim=512

echo "using $dim dim for PCA"

# if [ -z "$5" ]
#   then
#     layer=14
#   else
#     layer=$5
# fi
layer=14

echo "extracting from layer $layer"

train_split=train
valid_split=valid
test_split=test

all_splits="train valid test"

# if [[ -f "$source_dir/$train_split.tsv" ]]; then
#     all_splits+=($train_split)
# fi

# if [[ -f "$source_dir/$valid_split.tsv" ]]; then
#     all_splits+=($valid_split)
# fi

# if [[ -f "$source_dir/$test_split.tsv" ]]; then
#     all_splits+=($test_split)
# fi

echo "processing splits: $all_splits, dim=$dim, layer=$layer"

mkdir -p $tgt_dir

# cp $source_dir/*.tsv $tgt_dir
# cp $source_dir/*.wrd $tgt_dir
# cp $source_dir/*.ltr $tgt_dir
# cp $source_dir/*.phn $tgt_dir
# cp $source_dir/dict* $tgt_dir

# setopt shwordsplit
# set -o shwordsplit

for split in $all_splits; do
  # python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py $source_dir --split $split \
  # --save-dir $tgt_dir --checkpoint $model --layer $layer
  python scripts/wav2vec_extract_features.py $source_dir --split $split \
  --save-dir $tgt_dir --checkpoint $model --layer $layer
done
echo "Finished extract features."

echo "Clustering..."
python scripts/wav2vec_cluster_faiss.py $tgt_dir/${train_split}.tsv \
--checkpoint $model --save-dir $tgt_dir -f "CLUS128" --sample-pct 1.0
echo "Finished clustering."

for split in $all_splits; do
  python scripts/wav2vec_apply_cluster_faiss.py $tgt_dir \
  --checkpoint $model --path $tgt_dir/CLUS128 --split $split
done
echo "Applied cluster features."

python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/pca.py $tgt_dir/${train_split}.npy --output $tgt_dir/pca --dim $dim
echo "Ran PCA."

for split in $all_splits; do
  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/apply_pca.py $tgt_dir --split $split --save-dir $tgt_dir/precompute_pca$dim --pca-path $tgt_dir/pca/${dim}_pca --batch-size 1048000

  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py $tgt_dir/precompute_pca$dim --cluster-dir $tgt_dir/CLUS128 \
  --split $split --save-dir $tgt_dir/precompute_pca${dim}_cls128_mean --pooling mean

  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/mean_pool.py $tgt_dir/precompute_pca${dim}_cls128_mean \
  --save-dir $tgt_dir/precompute_pca${dim}_cls128_mean_pooled --split $split
done

echo "Post processed."