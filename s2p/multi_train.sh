rm core.*
PREFIX=w2v_unsup_gan_xp
TASK_DATA=/work/b07502072/corpus/u-s2s/audio/en_feats/voxpopuli/xlsr/precompute_pca512_cls128_mean_pooled
TEXT_DATA=/work/b07502072/corpus/u-s2s/text/voxpopuli_trans/en/prep/phones
KENLM_PATH=/work/b07502072/corpus/u-s2s/text/voxpopuli_trans/en/prep/phones/lm.phones.filtered.04.bin
EXP_NAME=voxpopuli_en/xlsr/vox_trans
# EXP_NAME={voxpopuli_{en,de}, mls_{en,de}}/{xlsr, large_{clean,noisy,vox}}/{vox_trans,wiki_{1,2,3,1-5},mls_trans}
export HYDRA_FULL_ERROR=1

if [ -d ./multirun/$EXP_NAME ] 
then
    echo "Directory $EXP_NAME already exists." 
    exit 9999 # die with error code 9999
fi
echo "Exp name(save dir): $EXP_NAME"

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    -m --config-dir config/gan \
    --config-name w2vu \
    hydra.sweep.dir=multirun/${EXP_NAME} \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    dataset.num_workers=0 \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=4 model.gradient_penalty=2.0,1.5 \
    model.smoothness_weight=0.5  \
    distributed_training.distributed_world_size=1 'common.seed=range(0,1)' \
    optimization.max_update=150000 \
    +description=${EXP_NAME}
