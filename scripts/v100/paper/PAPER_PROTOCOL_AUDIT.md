# Original-paper protocol audit

The earlier 12-hour configuration is a pilot and is not reportable as an
original-paper reproduction. The `config/v100/paper` configurations must pass
`preflight_paper.sh`, which now asserts the high-impact protocol fields below.

| Method | Original-paper protocol | Audited LeaFBench protocol | Previous high-impact deviation |
|---|---|---|---|
| TRAP | 100 suffixes; 4-digit main setting; 20-token suffix; 1500 GCG steps; 512 candidates; top-256; no early stop; number-token filtering; 10 completions/suffix | 100 targets shared by three disjoint shards (34/33/33); full target sentence; 1500/512/512/256; number-word filter; 10 completions | 100 steps, 128 candidates, top-128, 6-digit simple target, no number-token filter, one completion |
| PlugAE | ProFlingo's 50-query set; one universal adversarial embedding; copyright token `mkahg`; Adam, lr=0.1, 30 epochs; temperature=1, top-p=1 | Optimization implementation is universal across all 50 queries and two templates; paper optimizer fields asserted; base-model evaluation uses temperature=1/top-p=1 | Source layout used Qwen rather than prioritizing paper models; decoding was greedy |
| ZeroPrint | HumanEval code completion; n=2 base queries; m=4 perturbations; replace 3 words from top-10 GloVe neighbours; t=20 sampled generations/query; 512-token generation cap; MPNet; ridge alpha=0.001; 200 calls/model | fixed HumanEval seed-1000 cache; n=2, m=4, t=20; released-code 0.7/0.9 sampling and 512-token cap; no truncation; source artifact reused rather than re-queried | TruthfulQA n=20, m=5, t=1, greedy 48-token generation, output truncation 64; shared evaluation YAML would also re-query the source |
| REEF | first 200 TruthfulQA samples; last-token activations; per-feature centering/scaling; linear CKA; layer 18 default for efficient comparison | first 200 samples, layer 18, released-code standardization, linear CKA | Random 200-sample subset; concatenated last six layers; unstandardized activations |

Important protocol distinctions:

- Original GCG uses 500 steps. TRAP deliberately uses 1500 steps to make the
  suffix more model-specific.
- The ZeroPrint paper runs TRAP for 100 iterations as a query-budget baseline.
  That setting is not a reproduction of the original TRAP experiment and must
  be labelled separately if reported.
- The original TRAP code uses one fixed 100-target set and parallelizes it by
  offsets. Shards are not independent construction seeds.
- PlugAE is a proactive method: its learned embedding is assigned to a copyright
  token in candidate/derivative models. Comparisons to unrelated models do not
  install that embedding.

Remaining scope limitation: the local benchmark contains only a subset of the
derivative checkpoints used by the papers. Results on the available subset must
not be described as a complete reproduction of every paper table.
