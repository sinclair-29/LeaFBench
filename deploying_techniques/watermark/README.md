# LLM watermarks

LeaFBench implements only embedding and extraction for:

- KGW, using the soft `simple_1` watermark;
- the optimized bang-bang watermark from Wouters (2024);
- MorphMark `linear`, `exp`, and `log` variants;
- WaterMod's official zero-bit `SAM` variant.

The implementation reuses LeaFBench's model pool and model interface. It does
not vendor the official experiment repositories, quality metrics, attacks, or
dataset pipelines.

## Offline smoke tests

Each smoke configuration points to a small committed JSONL corpus under
`data/watermark/` and to LeaFBench's existing local Llama-2-7B-chat checkpoint.
No model or dataset download is performed:

```bash
python scripts/run_watermark_smoke.py \
  --config config/watermark_kgw_smoke.yaml

python scripts/run_watermark_smoke.py \
  --config config/watermark_opt_smoke.yaml

python scripts/run_watermark_smoke.py \
  --config config/watermark_morphmark_smoke.yaml

python scripts/run_watermark_smoke.py \
  --config config/watermark_watermod_smoke.yaml
```

On the configured remote server, the WaterMod smoke experiment can be run
directly from the LeaFBench repository root:

```bash
bash scripts/watermod.sh
```

The script uses the existing local checkpoint at
`/home/chj/LLMJailbreak/models/Llama-2-7b-chat-hf`, runs fully offline on GPU
0, and writes matched generation/detection records to
`outputs/watermark/watermod_smoke.jsonl`. Set `MODEL_PATH` or
`CUDA_VISIBLE_DEVICES` to override either default.

The smoke backbone only validates integration. The transferred baseline smoke
configs historically support an OPT-1.3B override; the WaterMod paper instead
evaluated Qwen-2.5-1.5B. The WaterMod smoke command above intentionally uses
the same local Llama-2-7B-chat integration backbone as the other LeaFBench
smoke tests.

The runner resets the RNG before matched unwatermarked and watermarked
generation. It applies the same detector to both controls, prints their z-scores
side by side, and stores both detection records. It fails only for
integration errors, empty generations, or invalid detector statistics. A
detection-threshold miss does not fail the smoke test.

The JSONL files are loaded locally; smoke execution never downloads C4. To
recreate them deliberately, with network access and the pinned source and
tokenizer revisions:

```bash
python -m pip install datasets==4.0.0 transformers==4.54.1
python scripts/prepare_watermark_smoke_data.py --force
```

## WaterMod fidelity notes

The canonical LeaFBench method name is `watermod`; it implements only the
released `zero_bit/SAM` algorithm, not `SAM_MULTI`. Its defaults are
`delta=1.0`, Shannon entropy, `H_scale=1.2`, `prefix_length=1`,
`z_threshold=4.0`, `temperature=0.0`, and greedy decoding. The official SAM
configuration's keyed defaults (`hash_key=15485863`, `f_scheme=time`, and
`tau=1.0`) are retained as well.

The released code computes Shannon entropy with `torch.log` (natural log) but
normalizes it by `log2(vocab_size)`. At a uniform distribution, this makes the
entropy-derived odd-parity probability `(ln 2)^H_scale`, approximately 0.644
when `H_scale=1.2`, rather than 1. LeaFBench intentionally reproduces this
mixed-log behavior as the executable conference baseline. The paper's
consistent-log formulation is documented but is not exposed as a second mode.

The official code's group labels and `(rank + 1) % 2` expression are a
relabeling: together they select odd zero-based ranks when the keyed uniform
draw is below the entropy probability. LeaFBench uses explicit zero-based
parity internally and produces the same green mask. For cached generations,
the detector follows LeaFBench conventions by scoring only generated
completion tokens while using the prompt and already-generated completion
prefix to reconstruct each green list. Standalone text detection has no prompt
and therefore begins after the configured prefix length.

The fixed-input equivalence and CPU-only integration tests can be run with:

```bash
python -m unittest discover -s tests -p 'test_watermod.py' -v
```
