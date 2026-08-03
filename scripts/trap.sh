HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 python main.py \
    --benchmark_config 'config/benchmark_trap_smoke.yaml' \
    --fingerprint_config 'config/trap_smoke.yaml' \
    --log_path 'logs/'
