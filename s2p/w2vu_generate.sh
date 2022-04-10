export HYDRA_FULL_ERROR=1
# cp /work/b07502072/corpus/u-s2s/text/wiki/en/prep/phones/* /work/b07502072/corpus/u-s2s/audio/features/precompute_pca512_cls128_mean_pooled
python w2vu_generate.py --config-dir config/generate --config-name viterbi \
beam=50 \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=/work/b07502072/corpus/u-s2s/audio/features/large_noisy/precompute_pca512_cls128_mean_pooled \
fairseq.common_eval.path=/home/b07502072/u-speech2speech/s2p/multirun/2022-04-09/14-52-16/0/checkpoint_best.pt \
fairseq.dataset.gen_subset=asr_test results_path=/home/b07502072/u-speech2speech/s2p/multirun/2022-04-09/14-52-16/0/asr_test
