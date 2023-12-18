para_data_dir=./data/small_cc100_1M_de
save_dir=$para_data_dir/data-bin
src_lg=phn
tgt_lg=de
user_dir=./mass

fairseq-preprocess \
  --user-dir $user_dir \
  --task xmasked_seq2seq \
  --source-lang en --target-lang zh \
  --trainpref $para_data_dir/train.phn-de --validpref $para_data_dir/valid.phn-de --testpref $para_data_dir/test.phn-de \
  --destdir $save_dir \
  --source-lang ${src_lg} \
  --target-lang ${tgt_lg} \
  --srcdict $para_data_dir/dict.$src_lg.txt \
  --tgtdict $para_data_dir/dict.$tgt_lg.txt \
  --workers 8