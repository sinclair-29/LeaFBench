import copy
import hashlib
import os
import random
import re

import pandas as pd
from transformers import set_seed

from fingerprint.fingerprint_interface import FingerprintTestResult, LLMFingerprintInterface
from fingerprint.trap.gcg import GCGOptimizer
from fingerprint.trap.generate_prompts import generate_adversarial_suffix, generate_csv


class TRAPFingerprint(LLMFingerprintInterface):
    candidate_model_types = ("pretrained", "instruct", "instruction_tuning")
    requires_model_rendered_cross_model = True
    evaluation_capabilities = {
        "model_modification_robustness": True,
        "deployment_robustness": {"system_prompts": True, "sampling": True},
        "model_specificity": True,
        "prompt_stealthiness": True,
    }

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)
        self.n_goals = self.config.get("n_goals", 100)
        self.goal_offset = self.config.get("goal_offset", 0)
        self.goal_count = self.config.get("goal_count")
        self.prompt_seed = self.config.get("prompt_seed", self.config.get("seed", 42))
        self.string_type = self.config.get("string_type", "number")
        self.string_length = self.config.get("string_length", 3)
        self.prompt_path = self.config.get("prompt_path")
        self.gcg_config = self.config.get("gcg_config", {})
        self.test_n_times = self.config.get("test_n_times", 5)
        self.batch_size = self.config.get("batch_size", 16)
        self.source_self_test_max_new_tokens = self.config.get(
            "source_self_test_max_new_tokens", 18
        )

    def prepare(self, train_models=None):
        del train_models
        if os.path.exists(self.prompt_path) and not self.config.get(
            "regenerate_prompts", False
        ):
            frame = pd.read_csv(
                self.prompt_path,
                dtype={"prompt": str, "target": str, "string_target": str},
            )
        else:
            random.seed(self.prompt_seed)
            frame = generate_csv(
                self.n_goals,
                self.string_type,
                self.string_length,
                self.prompt_path,
            )

        end = None if self.goal_count is None else self.goal_offset + self.goal_count
        frame = frame.iloc[self.goal_offset:end]
        if frame.empty:
            raise ValueError("TRAP goal shard is empty")
        self.prompts = frame["prompt"].tolist()
        self.targets = frame["target"].tolist()
        self.string_target = frame["string_target"].tolist()

    def prepare_evaluation(self, records, train_models=None):
        del train_models
        if not records or any(
            record.get("payload", {}).get("kind") != "trap" for record in records
        ):
            raise ValueError("TRAP evaluation requires saved TRAP artifacts.")

    def expected_artifact_count(self):
        return len(getattr(self, "prompts", [])) or self.goal_count or self.n_goals

    def validate_partial_records(self, records):
        for index, record in enumerate(records, start=1):
            payload = record.get("payload", {})
            if record.get("item_index") != index:
                raise ValueError("TRAP partial records are not contiguous.")
            if payload.get("kind") != "trap":
                raise ValueError("TRAP partial batch contains a non-TRAP artifact.")
            if payload.get("instruction") != self.prompts[index - 1]:
                raise ValueError(f"TRAP prompt changed for checkpoint {index:03d}.")
            if str(payload.get("target")) != str(self.string_target[index - 1]):
                raise ValueError(f"TRAP target changed for checkpoint {index:03d}.")

    def _item_seed(self, local_index):
        base_seed = self.gcg_config.get("seed", self.config.get("seed", 42))
        absolute_index = self.goal_offset + local_index
        material = f"{base_seed}:{self.prompt_seed}:{absolute_index}".encode()
        return int(hashlib.sha256(material).hexdigest()[:8], 16) % (2**31)

    @staticmethod
    def _model_context(model):
        torch_model, tokenizer = model.load_model()
        render = lambda prompt: model.render_prompts([prompt], tokenizer)[0]
        max_input_length = (model.params or {}).get("max_input_length", 512)
        return torch_model, tokenizer, render, max_input_length

    def _payload(self, local_index, raw_prompt, rendered_prompt, **extra):
        instruction = self.prompts[local_index]
        payload = {
            "kind": "trap",
            "instruction": instruction,
            "raw_user_prompt": raw_prompt,
            "rendered_prompt": rendered_prompt,
            "optimized_text": (
                raw_prompt[len(instruction) :]
                if raw_prompt.startswith(instruction)
                else raw_prompt
            ),
            "target": str(self.string_target[local_index]),
        }
        payload.update(extra)
        return payload

    def iter_fingerprint_records(self, source_model, experiment_id, start_index=1):
        torch_model, tokenizer, render, max_input_length = self._model_context(
            source_model
        )
        for item_index in range(start_index, len(self.prompts) + 1):
            local_index = item_index - 1
            item_seed = self._item_seed(local_index)
            item_config = {**copy.deepcopy(self.gcg_config), "seed": item_seed}
            optimizer = GCGOptimizer(
                torch_model,
                tokenizer,
                render,
                item_config,
                max_input_length=max_input_length,
            )
            final_loss, raw_prompt = optimizer.optimize(
                self.prompts[local_index], self.targets[local_index]
            )
            rendered_prompt = render(raw_prompt)
            outputs = source_model.generate(
                [raw_prompt],
                do_sample=False,
                max_new_tokens=self.source_self_test_max_new_tokens,
            )
            output = outputs[0] if outputs else ""
            target = str(self.string_target[local_index])
            parsed = self._parse_target(output, target)
            success = parsed == target
            warnings = [] if success else [f"trap_source_self_test_failed:{item_index:03d}"]

            payload = self._payload(
                local_index,
                raw_prompt,
                rendered_prompt,
                optimization={
                    "final_loss": float(final_loss),
                    "num_steps": int(self.gcg_config.get("num_steps", 250)),
                    "base_seed": self.gcg_config.get(
                        "seed", self.config.get("seed", 42)
                    ),
                    "item_seed": item_seed,
                },
                source_self_test={
                    "decoding": {
                        "do_sample": False,
                        "max_new_tokens": self.source_self_test_max_new_tokens,
                    },
                    "output": output,
                    "parsed_target": parsed,
                    "success": int(success),
                    "invalid": int(parsed is None),
                },
            )
            yield self._record(
                experiment_id,
                item_index,
                source_model,
                payload,
                metadata={"quality_warnings": warnings},
            )

    def get_fingerprint(self, model):
        if model.model_name not in {model.pretrained_model, model.instruct_model}:
            return 0
        torch_model, tokenizer, render, max_input_length = self._model_context(model)
        return generate_adversarial_suffix(
            torch_model,
            tokenizer,
            self.prompts,
            self.targets,
            self.gcg_config,
            render,
            max_input_length=max_input_length,
        )

    def _run_trials(
        self,
        testing_model,
        entries,
        generation=None,
        *,
        prompts_are_rendered=False,
        seed=None,
        ignore_batch_errors=False,
        metadata=None,
    ):
        if seed is not None:
            set_seed(seed)
        generation = dict(generation or {})
        trials = []

        for start in range(0, len(entries), self.batch_size):
            batch = entries[start : start + self.batch_size]
            try:
                outputs = testing_model.generate(
                    [entry["prompt"] for entry in batch],
                    prompts_are_rendered=prompts_are_rendered,
                    **generation,
                )
            except Exception:
                if not ignore_batch_errors:
                    raise
                outputs = [""] * len(batch)

            for entry, output in zip(batch, outputs):
                if isinstance(output, list):
                    output = output[0] if output else ""
                target = entry["target"]
                parsed = self._parse_target(output, target)
                trials.append(
                    {
                        "fingerprint_id": entry["fingerprint_id"],
                        "target": target,
                        "parsed_target": parsed,
                        "output": output,
                        "success": int(parsed == target),
                        "invalid": int(parsed is None),
                        **({"seed": seed} if seed is not None else {}),
                    }
                )

        total = len(trials)
        hit_rate = sum(row["success"] for row in trials) / total if total else 0.0
        invalid_rate = sum(row["invalid"] for row in trials) / total if total else 0.0
        return FingerprintTestResult(
            score=hit_rate,
            metrics={"target_hit_rate": hit_rate, "invalid_rate": invalid_rate},
            trials=trials,
            metadata=metadata or {},
        )

    def compare_fingerprints(self, base_model, testing_model):
        fingerprint = base_model.get_fingerprint()
        if not fingerprint:
            return 0.0
        entries = [
            {
                "fingerprint_id": f"legacy:{index + 1:03d}",
                "prompt": prompt,
                "target": str(self.string_target[index]),
            }
            for index, prompt in enumerate(fingerprint)
            for _ in range(self.test_n_times)
        ]
        return self._run_trials(
            testing_model,
            entries,
            ignore_batch_errors=True,
        ).score

    def fingerprint_to_records(self, fingerprint, source_model, experiment_id):
        if len(fingerprint) != len(self.string_target):
            raise ValueError(
                "TRAP fingerprint count does not match its prepared target count."
            )
        _, tokenizer = source_model.load_model()
        rendered = source_model.render_prompts(fingerprint, tokenizer)
        return [
            self._record(
                experiment_id,
                index,
                source_model,
                self._payload(index - 1, raw_prompt, rendered[index - 1]),
            )
            for index, raw_prompt in enumerate(fingerprint, start=1)
        ]

    def fingerprint_from_records(self, records):
        return [record["payload"]["raw_user_prompt"] for record in records]

    def verify_fingerprint(self, source_model, testing_model, generation=None):
        records = getattr(source_model, "fingerprint_records", None)
        if not records:
            raise ValueError("TRAP verification requires loaded fingerprint records.")

        generation = dict(generation or {})
        seed = int(generation.pop("seed", 0))
        input_mode = generation.pop("input_mode", "source_rendered")
        if input_mode not in {"source_rendered", "model_rendered"}:
            raise ValueError(f"Unsupported TRAP input_mode: {input_mode}")
        if input_mode == "source_rendered" and (
            testing_model.model_name != source_model.model_name
        ):
            raise ValueError(
                "TRAP source_rendered prompts may only be reused on the source "
                "checkpoint; cross-model verification must use model_rendered."
            )

        prompt_key = (
            "rendered_prompt" if input_mode == "source_rendered" else "raw_user_prompt"
        )
        entries = [
            {
                "fingerprint_id": record["fingerprint_id"],
                "prompt": record["payload"][prompt_key],
                "target": record["payload"]["target"],
            }
            for record in records
        ]
        return self._run_trials(
            testing_model,
            entries,
            generation,
            prompts_are_rendered=input_mode == "source_rendered",
            seed=seed,
            metadata={"input_mode": input_mode, "seed": seed},
        )

    def stealth_texts(self, records):
        return [
            {
                "fingerprint_id": record["fingerprint_id"],
                "kind": kind,
                "text": record["payload"][field],
            }
            for record in records
            for kind, field in (
                ("full_user_prompt", "raw_user_prompt"),
                ("optimized_suffix", "optimized_text"),
            )
        ]

    @staticmethod
    def _parse_target(output, target):
        output = "" if output is None else str(output)
        if target.isdigit():
            match = re.search(rf"(?<!\d)\d{{{len(target)}}}(?!\d)", output)
        elif target.isalpha():
            match = re.search(
                rf"(?<![A-Za-z])[A-Za-z]{{{len(target)}}}(?![A-Za-z])", output
            )
        else:
            return target if target in output else None
        return match.group(0) if match else None
