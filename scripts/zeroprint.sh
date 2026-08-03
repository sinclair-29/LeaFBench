#!/usr/bin/env bash

export NLTK_DATA=/home/chj/LLMJailbreak/models/nltk_data
export GENSIM_DATA_DIR=/home/chj/LLMJailbreak/models

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 python main.py \
    --benchmark_config 'config/benchmark_zeroprint_smoke.yaml' \
    --fingerprint_config 'config/zeroprint_smoke.yaml' \
    --log_path 'logs/'
