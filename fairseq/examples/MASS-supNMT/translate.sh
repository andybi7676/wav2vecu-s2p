tgt_lg=en
SAVE_DIR=/work/b07502072/results/denorm/$tgt_lg/checkpoints/wiki_cc100_10M_mt_small
DATA_ROOT=/work/b07502072/corpus/denorm/$tgt_lg
DATA_SUBDIR=noisy/2M_cc100

CKPT_PATH=$SAVE_DIR/checkpoint_best.pt
DATA_DIR=$DATA_ROOT/$DATA_SUBDIR/data-bin
OUT_DIR=$DATA_ROOT/$DATA_SUBDIR/denorm_results
mkdir -p $OUT_DIR
SPM_MODEL_PATH=$DATA_ROOT/spm/spm.model
OUT_NAME=$OUT_DIR/test.norm-$tgt_lg

fairseq-generate $DATA_DIR \
	--user-dir mass --fp16 \
	--langs norm,$tgt_lg \
	-s norm -t $tgt_lg \
	--source-langs norm,$tgt_lg --target-langs norm,$tgt_lg \
	--mt_steps norm-$tgt_lg \
	--gen-subset test \
	--task xmasked_seq2seq \
	--path $CKPT_PATH \
	--beam 3 \
	--remove-bpe > $OUT_NAME.raw

# cat $OUT_NAME.raw | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[de\]//g' | spm_decode --model=$SPM_MODEL_PATH --input_format=piece > $OUT_NAME.hyp
# cat $OUT_NAME.raw | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[de\]//g' | spm_decode --model=$SPM_MODEL_PATH --input_format=piece > $OUT_NAME.ref
# echo "unormalized:  " > $OUT_NAME.res
# python ./utils/cal_wer.py --hyp $OUT_NAME.hyp --ref $OUT_NAME.ref >> $OUT_NAME.res
# echo "normalized:   " >> $OUT_NAME.res
# python ./utils/cal_wer.py --hyp $OUT_NAME.hyp --ref $OUT_NAME.ref --normalize >>$OUT_NAME.res
# cat $OUT_NAME.res