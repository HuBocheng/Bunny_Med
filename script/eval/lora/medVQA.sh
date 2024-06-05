#!/bin/bash

SPLIT="MED_VQA19"
LANG=en
MODEL_TYPE=phi-2
MODEL_BASE=/path/to/base_llm_model
TARGET_DIR=bunny-lora-phi-2


python -m bunny.eval.model_vqa_med \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-base $MODEL_BASE \
    --model-type $MODEL_TYPE \
    --question-file ./eval/mmbench/$SPLIT.tsv \
    --answers-file $BUNNY_DATASET/Med-VQA19/test/answer.txt\
    --image-floder $BUNNY_DATASET/Med-VQA19/test/images \
    --category all \
    --dataType test \
    --lang $LANG \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode bunny \
    --dataType test

mkdir -p eval/mmbench/answers_upload/$SPLIT

python eval/mmbench/convert_mmbench_for_submission.py \
    --annotation-file ./eval/mmbench/$SPLIT.tsv \
    --result-dir ./eval/mmbench/answers/$SPLIT \
    --upload-dir ./eval/mmbench/answers_upload/$SPLIT \
    --experiment $TARGET_DIR
