#!/usr/bin/env zsh

TASK_DATA=/home/b07502072/u-speech2speech/w2v_finetune/data/cv4_de/ltr
MODEL_PATH=/work/b07502072/pretrained_models/xlsr_53_56k.pt
CONFIG_NAME=vox_100h_ltr
OUT_DIR=/work/b07502072/results/u-s2s/w2v_finetune
EXP_NAME=cv4_de/ltr

fairseq-hydra-train \
    hydra.run.dir=${OUT_DIR}/${EXP_NAME} \
    task.data=${TASK_DATA} \
    model.w2v_path=${MODEL_PATH} \
    --config-dir config \
    --config-name ${CONFIG_NAME}

wait