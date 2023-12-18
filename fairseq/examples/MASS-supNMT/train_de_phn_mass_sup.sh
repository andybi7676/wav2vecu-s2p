data_dir=/work/b07502072/corpus/p2w/data/wiki_cc100_2M_de/data-bin
save_dir=./checkpoints/wiki_cc100/2M_de/mass_memt
user_dir=./mass

seed=4771
max_tokens=4096 # for 16GB GPUs 
update_freq=1
dropout=0.1
attention_heads=16
embed_dim=1024
ffn_embed_dim=4096
encoder_layers=10
decoder_layers=6
word_mask=0.3

mkdir -p $save_dir

fairseq-train $data_dir \
	--user-dir $user_dir \
    --task xmasked_seq2seq \
	--source-langs phn,de \
	--target-langs phn,de \
    --langs phn,de \
	--arch xtransformer \
    --mass_steps phn-phn,de-de \
    --memt_steps phn-de \
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
    --keep-interval-updates 1 --save-interval-updates 5000  --log-interval 100 --no-epoch-checkpoints \
    --share-decoder-input-output-embed \
    --valid-lang-pairs phn-de \
	--word_mask ${word_mask} \
	--ddp-backend=no_c10d
