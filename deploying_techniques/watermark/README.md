# LLM watermarks

LeaFBench implements only embedding and extraction for:

- KGW, using the soft `simple_1` watermark;
- the optimized bang-bang watermark from Wouters (2024);
- MorphMark `linear`, `exp`, and `log` variants.

The implementation reuses LeaFBench's model pool and model interface. It does
not vendor the official experiment repositories, quality metrics, attacks, or
dataset pipelines.

## Offline smoke tests

Each smoke configuration points to a committed five-record JSONL corpus under
`data/watermark/`. Edit its `model_path`, or override it at the command line:

```bash
python scripts/run_watermark_smoke.py \
  --config config/watermark_kgw_smoke.yaml \
  --model-path /absolute/path/to/opt-1.3b

python scripts/run_watermark_smoke.py \
  --config config/watermark_opt_smoke.yaml \
  --model-path /absolute/path/to/opt-1.3b

python scripts/run_watermark_smoke.py \
  --config config/watermark_morphmark_smoke.yaml \
  --model-path /absolute/path/to/opt-1.3b
```

The runner resets the RNG before matched unwatermarked and watermarked
generation. It prints detector statistics for every sample and fails only for
integration errors, empty generations, or invalid detector statistics. A
detection-threshold miss does not fail the smoke test.

The JSONL files are loaded locally; smoke execution never downloads C4. To
recreate them deliberately, with network access and the pinned source and
tokenizer revisions:

```bash
python -m pip install datasets==4.0.0 transformers==4.54.1
python scripts/prepare_watermark_smoke_data.py --force
```
