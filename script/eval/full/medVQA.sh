#!/bin/bash

ROOT=/zhuzixuan/Bunny
SPLIT="MED_VQA19"
LANG=en
MODEL_TYPE=phi-2
MODEL_BASE=$ROOT/weight/phi-2/
VLM_NAME=Bunny-lora-phi-2-siglip
CAT1=all # 使用的模型是从Med-VQA19 all finetune得到的
CAT2=cat1  # 使用的测试数据的分区
FINETUNE_MODEL=$ROOT/weight/Bunny-lora-phi-2-Med-$CAT1/  # finetune model by Med-VQA19 all/C1/C2/C3


# 原始的--model-path参数取值为： $ROOT/weight/$VLM_NAME \


HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m bunny.eval.model_vqa_med \
    --model-path $FINETUNE_MODEL \
    --model-type $MODEL_TYPE \
    --output-file $BUNNY_DATASET/Med-VQA19/test/output-$CAT1-$CAT2.txt\
    --data-floder $BUNNY_DATASET/Med-VQA19/test \
    --image-floder $BUNNY_DATASET/Med-VQA19/test/images \
    --category $CAT2 \
    --dataType test \
    --lang $LANG \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode bunny \
    --dataType test
