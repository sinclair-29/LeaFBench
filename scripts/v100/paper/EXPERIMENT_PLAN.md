# Original-paper-aligned fingerprint experiment plan (V100 × 16)

## Scope and claims

This run evaluates the correctness and empirical behavior of four model
fingerprinting methods: TRAP, PlugAE, ZeroPrint, and REEF. Model watermarking is
excluded. Protocol details and known deviations are recorded in
`PAPER_PROTOCOL_AUDIT.md`. The available checkpoints support source-model effectiveness,
cross-model specificity, deployment robustness, and prompt stealthiness.

TRAP and ZeroPrint have no locally verified derivative checkpoint for their
instruction-tuned sources, so modification robustness is disabled for those
runs. PlugAE evaluates the paper's proactive embedding transfer into available
instruction-tuned descendants. REEF evaluates Llama-2-7B against the locally
available Llama-2-Chat and CodeLlama descendants. These subsets must not be
described as reproducing every derivative column in the original papers.

## Experimental matrix

| Method | Source checkpoints | Independent construction seeds | Fingerprint size |
|---|---|---:|---:|
| TRAP | Guanaco-7B, Llama-2-7B-Chat, Vicuna-7B-v1.3 | target seed 41; fixed optimizer seed 42 in each process | 100 targets/source, sharded 34/33/33 |
| PlugAE | Llama-7B, Llama-2-7B, Mistral-7B-v0.1 | 42 | one universal embedding trained on 50 ProFlingo pairs |
| ZeroPrint | Qwen2.5-7B-Instruct, Llama-2-7B-Chat | 1000 | 2 HumanEval queries × (1 base + 4 perturbations) × 20 repeats = 200 calls/new model; source artifact reused |
| REEF | Llama-2-7B and available comparison models | deterministic | first 200 TruthfulQA statements, layer 18 |

Fifteen jobs run concurrently; GPU 15 is reserved for failed/OOM retries. TRAP shards a single paper target
set; those shards are not statistical replicates. PlugAE optimization seeds are
not replicated in this source-model coverage run. ZeroPrint and REEF are
evaluated once per source; decoding variation is evaluated separately where the
method supports sampling.

## Evaluations

1. Source effectiveness: report each method's native source score. For TRAP,
   additionally report target hit rate and invalid-output rate.
2. Model specificity: compare against the available local negatives spanning
   unrelated families, family-related hard negatives, and scale variants.
   Primary statistics are ROC-AUC, source-score rank, and source margin over the
   best negative. The in-sample Youden threshold/FPR is descriptive only.
3. Deployment robustness: use method-specific configurations. TRAP's paper
   score uses 10 generations per suffix at temperature 0.6/top-p 0.9; broader
   prompt/sampling sweeps are secondary robustness analyses. REEF supports
   system prompts but not sampling and records sampling as not applicable.
4. Prompt stealthiness: OPT-1.3B log-perplexity, calibrated on a fixed random
   sample of 1,000 local TruthfulQA statements at the 99th percentile. REEF has
   no textual trigger and is not applicable.

## Statistical reporting

Treat construction seeds, not individual generations, as independent repeats.
Do not treat TRAP's three target shards as independent seeds; pool the 100
targets and bootstrap over target suffixes. The current PlugAE run has one
construction seed per source, so report query-level uncertainty but do not claim
seed-level variance. For per-prompt hit rates also report Wilson 95% intervals. Report
all negative models individually; do not select or discard negative models
after inspecting scores. The primary specificity result is ROC-AUC because the
current Youden threshold is calibrated and evaluated on the same models.

## Runtime and reproducibility

The full original-paper TRAP protocol is not expected to fit a 12-hour wall-clock
budget on the available V100 allocation. The 12-hour run is retained only as a
pilot; do not report it as the main experiment. Every artifact records its model path, configuration hash, optimization
seed, and benchmark-registry hash. Run the preflight before launch and archive
the complete `results/v100_fingerprint_paper`, `logs/v100_fingerprint_paper`,
git commit, environment lock file, and `nvidia-smi` output with the paper.

Commands:

```bash
cd /raid/chj/fingerprint/LeaFBench
bash scripts/v100/paper/preflight_paper.sh
bash scripts/v100/paper/run_paper.sh
```

`run_paper.sh` writes concise lifecycle events to
`logs/v100_fingerprint_paper/launcher.log`. Full training/progress output is
kept in one `gpu*_*.log` file per job, so concurrent progress bars never make
the master log unreadable.
