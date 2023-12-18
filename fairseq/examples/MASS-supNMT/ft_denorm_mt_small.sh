data_dir=/work/b07502072/corpus/denorm/en/denorm_en_wiki_cc100/data-bin
save_dir=/work/b07502072/results/denorm/en/checkpoints/wiki_cc100_10M_mt_small
user_dir=./mass
tgt_lg=en
# ckpt_path=/home/b07502072/u-speech2speech/fairseq/examples/MASS-supNMT/checkpoints/pretrained/mass_ende_1024.pth

seed=4771
max_tokens=4096 # for 16GB GPUs 
update_freq=1
dropout=0.1
attention_heads=16
embed_dim=1024
ffn_embed_dim=4096
encoder_layers=4
decoder_layers=4

mkdir -p $save_dir
cp ./ft_denorm_mt_small.sh $save_dir

fairseq-train $data_dir \
	--user-dir $user_dir \
    --task xmasked_seq2seq \
	--source-langs norm,$tgt_lg \
	--target-langs norm,$tgt_lg \
    --langs norm,$tgt_lg \
	--arch xtransformer \
    --mt_steps norm-$tgt_lg \
    --save-dir $save_dir \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --fp16 \
    --lr-scheduler inverse_sqrt --lr 0.00005 \
    --criterion label_smoothed_cross_entropy \
    --lm-bias --seed ${seed} \
    --log-format json --log-file ${save_dir}/train.log --tensorboard-logdir ${save_dir}/tb \
    --max-tokens ${max_tokens} --update-freq ${update_freq} \
    --encoder-normalize-before  --decoder-normalize-before \
    --dropout ${dropout} --relu-dropout 0.1 --attention-dropout 0.1 \
    --decoder-attention-heads ${attention_heads} --encoder-attention-heads ${attention_heads} \
    --decoder-embed-dim ${embed_dim} --encoder-embed-dim ${embed_dim} \
    --decoder-ffn-embed-dim ${ffn_embed_dim} --encoder-ffn-embed-dim ${ffn_embed_dim} \
    --encoder-layers ${encoder_layers} --decoder-layers ${decoder_layers} \
    --max-update 1000000 --max-epoch 50 \
    --keep-interval-updates 1 --save-interval-updates 5000 --log-interval 50 --no-epoch-checkpoints \
    --share-decoder-input-output-embed \
    --valid-lang-pairs norm-$tgt_lg \
	--ddp-backend=no_c10d 
