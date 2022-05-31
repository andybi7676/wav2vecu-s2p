export HYDRA_FULL_ERROR=1
TASK_DATA=/work/b07502072/corpus/u-s2s/audio/en_feats/voxpopuli/xlsr/precompute_pca512_cls128_mean_pooled
TEXT_DATA=/work/b07502072/corpus/u-s2s/text/voxpopuli_trans/en/prep/phones
SUBSET=asr_test
SAVE_DIR=voxpopuli_en/xlsr/vox_trans/cp4_gp1.5_sw0.5/seed0
DECODE_METHOD=viterbi
DECODE_TYPE=phones
TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/voxpopuli/en
# words or phones
if test "$DECODE_TYPE" = 'words'; then
    TARGET_DATA=$TARGET_DATA_DIR/$SUBSET.words.txt
    LM_PATH=$TEXT_DATA/../kenlm.wrd.o40003.bin
    LEXICON_PATH=$TEXT_DATA/../lexicon_filtered.lst
elif test "$DECODE_TYPE" = 'phones'; then
    TARGET_DATA=$TARGET_DATA_DIR/$SUBSET.phones.txt
    LM_PATH=$TEXT_DATA/lm.phones.filtered.04.bin
    LEXICON_PATH=$TEXT_DATA/lexicon.phones.lst
else
    echo "Invalid decode type, please choose from {words/phones}"
fi
# TARGET_DATA=/home/b07502072/u-speech2speech/s2p/utils/goldens/voxpopuli/asr_test.phones.txt

echo "AUDIO_DATA: $TASK_DATA"
echo "TEXT_DATA: $TEXT_DATA"
echo "SAVE_DIR: $SAVE_DIR"
echo "SUBSET: $SUBSET"
echo "DECODE_METHOD: $DECODE_METHOD"
echo "DECODE_TYPE: $DECODE_TYPE"
echo "LM_PATH: $LM_PATH"
echo "LEXICON_PATH: $LEXICON_PATH"
echo "TARGET_DATA: $TARGET_DATA"

cp $TEXT_DATA/* $TASK_DATA
python w2vu_generate.py --config-dir config/generate --config-name ${DECODE_METHOD} \
beam=200 \
lm_weight=5.0 \
lm_model=${LM_PATH} \
lexicon=${LEXICON_PATH} \
targets=${TARGET_DATA} \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=${TASK_DATA} \
fairseq.common_eval.path=/home/b07502072/u-speech2speech/s2p/multirun/${SAVE_DIR}/checkpoint_best.pt \
fairseq.dataset.gen_subset=${SUBSET} results_path=/home/b07502072/u-speech2speech/s2p/multirun/${SAVE_DIR}/${SUBSET}_${DECODE_METHOD}.${DECODE_TYPE}
rm $TASK_DATA/lm* $TASK_DATA/dict* $TASK_DATA/*log $TASK_DATA/train.bin $TASK_DATA/train.idx
