#!/bin/bash

ROOT=/zhuzixuan/Bunny
MODEL_TYPE=phi-2

OUTPUT_DIR=Bunny-lora-$MODEL_TYPE-EndoVis


mkdir -p ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR

deepspeed bunny/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path $ROOT/weight/phi-2/ \
    --model_type $MODEL_TYPE \
    --version bunny \
    --data_floder $BUNNY_DATASET/EndoVis-18 \
    --image_folder left_frames \
    --dataType train \
    --vision_tower $ROOT/weight/siglip/ \
    --pretrain_mm_mlp_adapter $ROOT/weight/bunny-pretrain-phi-2-siglip/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt


# merge Lora weight
python -c 'print("merging lora weight......")'
python script/merge_lora_weights.py \
	--model-path /zhuzixuan/Bunny/checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
	--model-base /zhuzixuan/Bunny/weight/phi-2 \
	--model-type phi-2 \
	--save-model-path /zhuzixuan/Bunny/weight/$OUTPUT_DIR