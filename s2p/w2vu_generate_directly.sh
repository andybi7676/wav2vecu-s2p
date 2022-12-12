export HYDRA_FULL_ERROR=1
# CHECKPOINT=/work/b07502072/pretrained_models/w2v_large_lv_fsh_swbd_cv.pt # large_noisy
# CHECKPOINT=/work/b07502072/pretrained_models/wav2vec2_large_west_germanic_v2.pt # large_vox
CHECKPOINT=/work/b07502072/pretrained_models/xlsr_53_56k.pt # xlsr
# CHECKPOINT=/work/b07502072/pretrained_models/wav2vec_vox_new.pt # large_clean
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/fr_feats/cv4/xlsr
TASK_DATA=/work/b07502072/corpus/u-s2s/audio/de_feats/cv4/xlsr
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/es_feats/cv4/xlsr
# TASK_DATA=/work/b07502072/corpus/u-s2s/audio/en_feats/LJ_speech/large_clean
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/de/prep_sil_0-5/phones
TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/de/prep/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/es/prep/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/cv_wiki/fr/prep_new_reduced_sil_0-5/phones
# TEXT_DATA=/work/b07502072/corpus/u-s2s/text/ls_wo_lv/prep_g2p/phones

SUBSET=asr_test

# SAVE_DIR=LJ_speech/large_clean/ls_wo_lv_g2p_3k/cp4_gp1.5_sw0.5/seed1
# SAVE_DIR=cv4_fr/xlsr/cv_wiki_new_reduced_sil_0-5_300k/cp4_gp2.0_sw0.5/seed1
# SAVE_DIR=cv4_fr/xlsr/cv_wiki_new_reduced_all/cp4_gp2.0_sw0.5/seed5
# SAVE_DIR=cv4_es/xlsr/cv_wiki_all/cp4_gp1.5_sw0.5/seed2
SAVE_DIR=cv4_de/xlsr/cv_wiki_3k/cp4_gp2.0_sw0.5/seed3
DECODE_METHOD=kenlm
DECODE_TYPE=phones
BEAM=200
LM_WEIGHT=5.0
# TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_fr/train_70h
TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_de
# TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/cv4_es/train_all
CKPT_SELECTION=best
# TARGET_DATA_DIR=/home/b07502072/u-speech2speech/s2p/utils/goldens/mls_en
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

echo "AUDIO_DATA: $TASK_DATA"
echo "TEXT_DATA: $TEXT_DATA"
echo "SAVE_DIR: $SAVE_DIR"
echo "SUBSET: $SUBSET"
echo "DECODE_METHOD: $DECODE_METHOD"
echo "DECODE_TYPE: $DECODE_TYPE"
echo "LM_PATH: $LM_PATH"
echo "LEXICON_PATH: $LEXICON_PATH"
echo "TARGET_DATA: $TARGET_DATA"
echo "BEAM: $BEAM"
echo "LM_WEIGHT: $LM_WEIGHT"

cp $TEXT_DATA/* $TASK_DATA
python w2vu_generate_directly.py --config-dir config/generate/directly --config-name ${DECODE_METHOD} \
beam=${BEAM} \
lm_weight=${LM_WEIGHT} \
lm_model=${LM_PATH} \
lexicon=${LEXICON_PATH} \
targets=${TARGET_DATA} \
post_process=silence \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=${TASK_DATA} \
fairseq.task.directly.checkpoint=${CHECKPOINT} \
fairseq.common_eval.path=/home/b07502072/u-speech2speech/s2p/multirun/${SAVE_DIR}/checkpoint_${CKPT_SELECTION}.pt \
fairseq.dataset.gen_subset=${SUBSET} results_path=/home/b07502072/u-speech2speech/s2p/multirun/${SAVE_DIR}/ckpt_${CKPT_SELECTION}_${SUBSET}_${DECODE_METHOD}_${BEAM}-${LM_WEIGHT}.${DECODE_TYPE}
rm $TASK_DATA/lm* $TASK_DATA/dict* $TASK_DATA/*log $TASK_DATA/train.bin $TASK_DATA/train.idx

wait
