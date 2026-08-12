# Conference-paper fingerprint experiment plan (V100 × 16)

## Scope and claims

This run evaluates the correctness and empirical behavior of four model
fingerprinting methods: TRAP, PlugAE, ZeroPrint, and REEF. Model watermarking is
excluded. The available checkpoints support source-model effectiveness,
cross-model specificity, deployment robustness, and prompt stealthiness.

No locally available checkpoint is a verified LoRA/DPO/SFT descendant of either
instruction-tuned source. Therefore modification robustness is disabled for
TRAP, ZeroPrint, and REEF. PlugAE alone reports transfer from the pretrained
Qwen2.5-7B/Llama-2-7B checkpoint to its linked instruction-tuned checkpoint;
this result must be described specifically as base-to-instruction transfer, not
as general fine-tuning robustness.

## Experimental matrix

| Method | Source checkpoints | Independent construction seeds | Fingerprint size |
|---|---|---:|---:|
| TRAP | Qwen2.5-7B-Instruct, Llama-2-7B-Chat | 42, 43, 44 | 34 per seed (102/source) |
| PlugAE | Qwen2.5-7B, Llama-2-7B | 42, 43, 44 | 50 query/target pairs |
| ZeroPrint | Qwen2.5-7B-Instruct, Llama-2-7B-Chat | 1000 | 20 queries × 5 perturbations |
| REEF | Qwen2.5-7B-Instruct, Llama-2-7B-Chat | 42 | 200 seeded random statements, last 6 layers |

The 16 source/method/seed combinations run concurrently, one per GPU. TRAP and
PlugAE need independent optimization repeats. ZeroPrint and REEF are evaluated
once per source because their configured extraction is deterministic; decoding
variation is evaluated separately where the method supports sampling.

## Evaluations

1. Source effectiveness: report each method's native source score. For TRAP,
   additionally report target hit rate and invalid-output rate.
2. Model specificity: compare against eight or nine local negatives spanning
   unrelated families, family-related hard negatives, and Qwen scale variants.
   Primary statistics are ROC-AUC, source-score rank, and source margin over the
   best negative. The in-sample Youden threshold/FPR is descriptive only.
3. Deployment robustness: five system prompts plus temperatures
   {0.0, 0.2, 0.5, 0.7, 1.0} and top-p {1.0, 0.95, 0.9, 0.8}, using five
   decoding seeds. REEF supports system prompts but not sampling and records the
   unsupported component as not applicable.
4. Prompt stealthiness: OPT-1.3B log-perplexity, calibrated on a fixed random
   sample of 1,000 local TruthfulQA statements at the 99th percentile. REEF has
   no textual trigger and is not applicable.

## Statistical reporting

Treat construction seeds, not individual generations, as independent repeats.
For TRAP and PlugAE report mean, standard deviation, and a 95% bootstrap
confidence interval across the three construction seeds. For per-prompt hit
rates also report Wilson 95% intervals, clustered by construction seed. Report
all negative models individually; do not select or discard negative models
after inspecting scores. The primary specificity result is ROC-AUC because the
current Youden threshold is calibrated and evaluated on the same models.

## Runtime and reproducibility

The run is designed for a 16 × V100-32GB server and an approximately 12-hour
budget. Every artifact records its model path, configuration hash, optimization
seed, and benchmark-registry hash. Run the preflight before launch and archive
the complete `results/v100_fingerprint_paper`, `logs/v100_fingerprint_paper`,
git commit, environment lock file, and `nvidia-smi` output with the paper.

Commands:

```bash
cd /raid/chj/fingerprint/LeaFBench
bash scripts/v100/paper/preflight_paper.sh
nohup bash scripts/v100/paper/run_paper.sh > logs/v100_fingerprint_paper/launcher.log 2>&1 &
```
