export HYDRA_FULL_ERROR=1
TASK_DATA=/work/b07502072/corpus/u-s2p/audio/large_clean/precompute_pca512_cls128_mean_pooled
TEXT_DATA=/work/b07502072/corpus/u-s2p/text/wiki_3/phones
SAVE_DIR=/w2vu/wav2vecu-s2p/s2p/multirun/2022-10-19/15-14-46/0

cp $TEXT_DATA/* $TASK_DATA
env PYTHONPATH=$FAIRSEQ_ROOT python w2vu_generate.py --config-dir config/generate --config-name viterbi \
beam=50 \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=${TASK_DATA} \
fairseq.common_eval.path=${SAVE_DIR}/checkpoint_best.pt \
fairseq.dataset.gen_subset=asr_test results_path=${SAVE_DIR}/asr_test
rm $TASK_DATA/lm* $TASK_DATA/dict* $TASK_DATA/*log $TASK_DATA/train.bin $TASK_DATA/train.idx
