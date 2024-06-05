#!/bin/bash

export ROOT=/zhuzixuan/Bunny
export MODEL_TYPE=phi-2
export PRETRAIN_DIR=bunny-$MODEL_TYPE-pretrain
export OUTPUT_DIR=bunny-lora-$MODEL_TYPE

exec "$@"
