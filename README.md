# LeaFBench

This is the official code for our paper entitled "[SoK: Large Language Model Copyright Auditing via Fingerprinting](http://arxiv.org/abs/2508.19843)". LeaFBench is a comprehensive benchmark designed to evaluate and compare various fingerprinting methods for large language models (LLMs). The project provides a unified interface for running, testing, and analyzing different models and algorithms, facilitating reproducible research and fair comparison.

## Project Structure

- `main.py`: Entry point for running the benchmark.
- `benchmark/`: Core benchmark logic, model interfaces, and implementations.
- `config/`: Configuration files for different experiments and benchmarks.
- `data/`: Datasets and auxiliary files used in experiments.
- `fingerprint/`: Fingerprinting methods and related utilities.
- `scripts/`: Shell scripts for running experiments and benchmarks.
- `utils/`: Utility functions and helpers.

## Environment Setup

**Install dependencies:**
Ensure you have Python 3.8+ installed. Install required packages:
```bash
pip install -r requirements.txt
```

**Note**: The pip may indicate that `gptqmodel` is not compatible with the `transformers` version. Please ignore this warning. We use `gptqmodel=2.2.0` and `transformers=4.54.1`.

## Reproducing Experiments

- All configuration files required to reproduce the experiments are located in the `config/` directory. Each YAML file corresponds to a specific experiment or benchmark setting.
- All scripts for running experiments are provided in the `scripts/` directory. Simply execute the corresponding shell script to run an experiment. For example:
  ```bash
  bash scripts/trap.sh
  ```
  The results will be generated according to the configuration specified in the related config file.

For further details, please refer to the comments in the scripts and configuration files.

## Artifact-first fingerprint evaluation

The semantic evaluation layer lives in the single top-level
`evaluation.py` file. It replaces neither the fingerprint algorithms nor the
legacy global similarity-matrix benchmark. Its job is to run the following
four reports from an already saved source fingerprint:

| Evaluation name | What it measures | Main outputs |
| --- | --- | --- |
| `model_modification_robustness` | Retention after fine-tuning, preference tuning, adapters, merging, quantization, or distillation | native method score and retention rate |
| `deployment_robustness` | Stability under system prompts and decoding settings | score, invalid rate when defined, and per-condition averages |
| `model_specificity` | False activation on unrelated, family-related, and scale-variant models | Youden-J threshold, model FPR, event FPR, and invalid rate |
| `prompt_stealthiness` | Whether textual fingerprint prompts pass a calibrated perplexity filter | mean GPT-2 log-PPL and filter pass rate |

Current applicability is explicit rather than inferred:

| Method | Modification | Deployment system prompt | Deployment sampling | Specificity | Prompt stealthiness |
| --- | --- | --- | --- | --- | --- |
| TRAP | yes | yes | yes | yes | yes |
| PlugAE | yes | yes | yes | yes | yes |
| ZeroPrint | yes | yes | yes | yes | yes |
| REEF | yes | yes | not applicable | yes | not applicable |

The old experiment numbers are deliberately not used as code identifiers.
The names above state what each evaluation actually does.

### Configuration boundaries

There are three inputs, with separate responsibilities:

- A benchmark config is only a model registry: checkpoint paths, model types,
  and model variants. `config/benchmark_evaluation_models.yaml` contains a DGX
  example using `/raid/chj/fingerprint/models` for locally available models.
- A fingerprint config such as `config/trap.yaml` or `config/reef.yaml` contains
  only method construction/extraction settings. Method capability declarations
  also live in the method implementation; for example, REEF declares sampling
  robustness and prompt stealthiness unsupported.
- One source-model Evaluation config selects the model groups, generation
  conditions, seeds, and any subset of the four reports. Examples are
  `config/evaluation_qwen25_7b_instruct.yaml` and
  `config/evaluation_llama2_7b_chat.yaml`. For a fingerprint embedded in a
  pretrained source checkpoint, `config/evaluation_qwen25_7b_base.yaml`
  explicitly selects the linked instruct checkpoint for deployment tests. The
  file is method-independent and is not read while generating a fingerprint.

Every model referenced by an Evaluation config must exist in the selected
benchmark config. Set an evaluation's `enabled` field to `false` to skip that
report. If it is enabled but unsupported by the fingerprint method, the fixed
result file is still written with `status: not_applicable`.

### Stage 1: generate and save one fingerprint batch

For example, generate a TRAP fingerprint for Qwen2.5-7B-Instruct:

```bash
python evaluation.py generate \
  --benchmark-config config/benchmark_evaluation_models.yaml \
  --fingerprint-config config/trap.yaml \
  --source-model Qwen2.5-7B-Instruct \
  --model-alias qwen25 \
  --results-root results
```

This stage calls the fingerprint method exactly once and writes immutable,
numbered JSON artifacts. It never runs any of the four evaluations.
For PlugAE, select `Qwen2.5-7B` with `--fingerprint-config config/plugae.yaml`
and later use `config/evaluation_qwen25_7b_base.yaml`; PlugAE embeds its source
fingerprint in a pretrained checkpoint and transfers it to linked derivatives.

### Stage 2: evaluate only the saved artifacts

Pass the exact fingerprint config and benchmark config recorded during stage 1:

```bash
python evaluation.py run \
  --benchmark-config config/benchmark_evaluation_models.yaml \
  --fingerprint-config config/trap.yaml \
  --evaluation-config config/evaluation_qwen25_7b_instruct.yaml \
  --batch-dir results/exp_qwen25_trap_seed_042_a
```

The run command fails if the numbered artifacts are missing, non-contiguous,
or inconsistent with `fingerprint_config.json`. It never regenerates a missing
fingerprint. Completed and `not_applicable` results are skipped. Failed results
run again only with `--retry-failed`; use `--overwrite` to intentionally rerun
all enabled result files.

### Result layout and versioning

One folder is one complete fingerprint batch and its reports:

```text
results/
├── EXPERIMENTS.md
└── exp_qwen25_trap_seed_042_a/
    ├── fingerprint_config.json
    ├── 001.json
    ├── 002.json
    ├── ...
    ├── model_modification_robustness.json
    ├── deployment_robustness.json
    ├── model_specificity.json
    └── prompt_stealthiness.json
```

`fingerprint_config.json` is the complete generation record: source checkpoint,
method parameters, fingerprint seed, benchmark-config hash, artifact count, and
experiment ID. Each numbered artifact has a globally unambiguous
`fingerprint_id` formed from that experiment ID and its item index.

The short folder name is only for navigation. When generation settings change,
a new whole folder receives the next letter (`_b`, `_c`, and so on). If an
Evaluation config changes after results already exist, the saved fingerprint
artifacts are copied unchanged into the next whole-folder variant and evaluated
there; old result files are preserved. `results/EXPERIMENTS.md` is regenerated
as a readable index of the real settings and statuses.

### Reproducibility rules

- Greedy decoding runs once with seed `0`; sampled conditions use every seed
  explicitly listed in the Evaluation config.
- The source fingerprint score is the denominator for modification retention.
- Model-score methods calibrate a finite decision threshold with Youden's
  statistic, `J = TPR - FPR`, using the configured positive and negative models.
- TRAP reports target hit rate and invalid rate; PlugAE reports keyword hit/TRR;
  ZeroPrint reports its configured similarity; REEF reports linear CKA.
- Prompt stealthiness uses mean negative log-likelihood (`log-PPL`), not
  exponentiated PPL. The provided configs calibrate the 99.9th percentile from
  1,000 seeded MMLU questions using GPT-2.

## Further Extensions

LeaFBench is designed to be easily extensible. You can add new models or fingerprinting methods by following the structure in the `benchmark/` and `fingerprint/` directories. Please refer to the [README for Benchmark](benchmark/README.md) and [README for Fingerprinting Methods](fingerprint/README.md) for guidance.

## Collections of Papers

We also provide a collection of papers about LLM fingerprinting in [Awesome-LLM-Fingerprinting](https://github.com/shaoshuo-ss/Awesome-LLM-Fingerprinting).

## Citation
If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{shao2025sok,
    title={SoK: Large Language Model Copyright Auditing via Fingerprinting},
    author={Shao, Shuo and Li, Yiming and He, Yu and Yao, Hongwei and Yang, Wenyuan and Tao, Dacheng and Qin, Zhan},
    journal={arXiv preprint arXiv:2508.19843},
    year={2025}
}
```
