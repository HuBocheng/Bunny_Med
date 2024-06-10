#!/bin/bash

ROOT=/zhuzixuan/Bunny
LANG=en
MODEL_TYPE=phi-2
MODEL_BASE=$ROOT/weight/phi-2/
VLM_NAME=Bunny-lora-phi-2-siglip
FINETUNE_MODEL=/defaultShare/archive/zhuzixuan/temp_weight_bak/Bunny-lora-phi-2-EndoVis
# FINETUNE_MODEL=/defaultShare/archive/zhuzixuan/temp_weight_bak/Bunny-lora-phi-2-Med-all # finetune model by Med-VQA19 all/C1/C2/C3
# FINETUNE_MODEL=$ROOT/weight/Bunny-lora-phi-2-EndoVis
# FINETUNE_MODEL=$ROOT/weight/Bunny-lora-phi-2-siglip

# 原始的--model-path参数取值为： $ROOT/weight/$VLM_NAME \


HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m bunny.eval.model_vqa_EndoVis \
    --model-path $FINETUNE_MODEL \
    --model-type $MODEL_TYPE \
    --output-file $BUNNY_DATASET/EndoVis-18/output.txt\
    --data-floder $BUNNY_DATASET/EndoVis-18 \
    --dataType test \
    --lang $LANG \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode bunny \
    --dataType test
