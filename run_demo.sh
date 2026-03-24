#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1

ROOT_DIR="."
if [ -d "$ROOT_DIR" ]; then
    cd "$ROOT_DIR" || exit
fi


LISA_MODEL="./pretrain_model/LISA-13B-llama2-v1-explanatory"


SEGAGENT_MODEL="./pretrain_model/segagent/SegAgent-Model"


SIMPLECLICK_CKPT="./pretrain_model/simpleclick/cocolvis_vit_large.pth"

echo ">>> 正在启动 Combined Pipeline..."
echo ">>> LISA: $LISA_MODEL"
echo ">>> SegAgent: $SEGAGENT_MODEL"

python demo.py \
    --lisa_version="${LISA_MODEL}" \
    --segagent_version="${SEGAGENT_MODEL}" \
    --checkpoint="${SIMPLECLICK_CKPT}" \
    --seg_model="simple_click" \
    --grounding_model="qwen-full" \
    --precision='fp16' \
    --load_in_4bit \
    --n_clicks=7 \
    --use_previous_mask \
    --device="cuda:1" \
    "$@"