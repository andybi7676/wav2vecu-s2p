#!/usr/bin/env zsh
SUBSET=test
TASK_DATA=/home/b07502072/u-speech2speech/w2v_finetune/data/cv4_de/ltr
EXP_DIR=/work/b07502072/results/u-s2s/w2v_finetune/cv4_de/ltr
DECODE_METHOD=viterbi

# LM_PATH=/work/b07502072/corpus/u-s2s/text/cv_wiki/fr/prep_new_reduced_sil_0-5/kenlm.wrd.o40003.bin
# LM_PATH=/work/b07502072/corpus/u-s2s/text/cv_wiki/es/prep/kenlm.wrd.o40003.bin
LM_PATH=/work/b07502072/corpus/u-s2s/text/cv_wiki/de/prep/kenlm.wrd.o40003.bin

MODEL_CKPT=$EXP_DIR/checkpoints/checkpoint_best.pt
RES_DIR=$EXP_DIR/results/${DECODE_METHOD}_$SUBSET

# python $FAIRSEQ_ROOT/examples/speech_recognition/infer.py ${TASK_DATA} --task audio_finetuning \
# --nbest 1 --path ${MODEL_CKPT} --gen-subset $SUBSET --results-path ${RES_DIR} --w2l-decoder ${DECODE_METHOD} \
# --lm-model ${LM_PATH} --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
# --post-process letter

cp $TASK_DATA/src/$SUBSET.words.txt $RES_DIR
cp $TASK_DATA/src/$SUBSET.tsv $RES_DIR
tail +2 $RES_DIR/$SUBSET.tsv | cut -d '.' -f1 > $RES_DIR/$SUBSET.fids

for fname in ref hypo; do
    cat $RES_DIR/$fname.word-checkpoint_best.pt-$SUBSET.txt | python /home/b07502072/u-speech2speech/w2v_finetune/sort_outputs.py > $RES_DIR/$fname.$SUBSET.sorted.txt
done

echo -e "fids|w2v2_ft_hyps|w2v2_ft_refs" > $RES_DIR/w2v2_ft_$DECODE_METHOD.$SUBSET.tsv
paste -d '|' $RES_DIR/$SUBSET.fids $RES_DIR/hypo.$SUBSET.sorted.txt $RES_DIR/$SUBSET.words.txt >> $RES_DIR/w2v2_ft_$DECODE_METHOD.$SUBSET.tsv
cat $RES_DIR/w2v2_ft_$DECODE_METHOD.$SUBSET.tsv | python /home/b07502072/u-speech2speech/w2v_finetune/cal_wer.py > $RES_DIR/res.$SUBSET.txt

wait