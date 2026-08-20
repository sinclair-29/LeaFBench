import csv
import hashlib
import logging
import random
import re

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed

from benchmark.model_pool import TokenEmbeddingOverrideError
from fingerprint.fingerprint_interface import (
    FingerprintTestResult,
    LLMFingerprintInterface,
)


PROFLINGO_TEMPLATES = (
    (
        "alpaca",
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request. ### Instruction: ",
        " ### Response:",
    ),
    (
        "zero_shot",
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's "
        "questions.\nHuman: ",
        "\nAssistant:",
    ),
)


class PlugAEFingerprint(LLMFingerprintInterface):
    """PlugAE using the paper's transferred-embedding evaluation protocol."""

    requires_suspect_fingerprints = False
    candidate_model_types = ("pretrained",)
    evaluation_capabilities = {
        "model_modification_robustness": True,
        "deployment_robustness": {"system_prompts": True, "sampling": True},
        "model_specificity": True,
        "prompt_stealthiness": True,
    }

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config or {}, accelerator=accelerator)
        self.query_path = self.config.get("query_path", "data/plugae_questions.csv")
        self.num_queries = self.config.get("num_queries", 50)
        self.copyright_token = self.config.get("copyright_token", "mkahg")
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.epochs = self.config.get("epochs", 30)
        self.optimization_batch_size = self.config.get("optimization_batch_size", 8)
        self.generation_batch_size = self.config.get("generation_batch_size", 16)
        self.diagnostic_interval = self.config.get("diagnostic_interval", 5)
        self.source_self_test_max_new_tokens = self.config.get(
            "source_self_test_max_new_tokens", 48
        )
        self.seed = self.config.get("seed", 42)
        if not self.copyright_token:
            raise ValueError("PlugAE copyright_token must not be empty.")
        if self.num_queries is not None and self.num_queries < 1:
            raise ValueError("PlugAE num_queries must be at least 1.")
        if self.epochs < 1:
            raise ValueError("PlugAE epochs must be at least 1.")
        if self.optimization_batch_size < 1 or self.generation_batch_size < 1:
            raise ValueError("PlugAE batch sizes must be at least 1.")
        if self.diagnostic_interval < 1:
            raise ValueError("PlugAE diagnostic_interval must be at least 1.")
        self.queries = []
        self.targets = []
        self.keywords = []
        self.query_sha256 = None
        self.output_parser = None

    def expected_artifact_count(self):
        return 1

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def prepare(self, train_models=None):
        """Load the ProFlingo query, target, and keyword triples."""
        del train_models
        with open(self.query_path, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))

        required = {"question", "answer", "keyword"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(
                f"PlugAE query file {self.query_path} must contain columns "
                f"{sorted(required)}."
            )

        if self.num_queries is not None:
            rows = rows[: self.num_queries]
        if not rows:
            raise ValueError("PlugAE requires at least one query.")
        self.queries = [row["question"] for row in rows]
        self.targets = [row["answer"] for row in rows]
        self.keywords = [row["keyword"] for row in rows]
        self.output_parser = self._infer_output_parser(self.keywords)
        query_payload = "\n".join(
            "\0".join((row["question"], row["answer"], row["keyword"]))
            for row in rows
        )
        self.query_sha256 = hashlib.sha256(query_payload.encode("utf-8")).hexdigest()
        self.logger.info("Loaded %d PlugAE query-target pairs", len(rows))

    def get_fingerprint(self, model):
        """Optimize one universal adversarial embedding for a candidate model."""
        if model.type not in self.candidate_model_types:
            return None
        if not self.queries:
            raise ValueError("PlugAE.prepare() must be called before fingerprinting.")

        torch_model, tokenizer = model.load_model()
        embedding_layer = torch_model.get_input_embeddings()
        if embedding_layer is None or not hasattr(embedding_layer, "weight"):
            raise ValueError(f"Candidate model {model.model_name} has no input embeddings.")

        adversarial_embedding = self._initialize_embedding(embedding_layer.weight)
        initial_embedding = adversarial_embedding.detach().clone()
        optimizer = torch.optim.Adam([adversarial_embedding], lr=self.learning_rate)
        indices = list(range(len(self.queries)))
        rng = random.Random(self.seed)
        was_training = torch_model.training
        requires_grad = [parameter.requires_grad for parameter in torch_model.parameters()]

        self.logger.info(
            "Optimizing PlugAE for %s: %d queries, %d epochs, batch size %d",
            model.model_name,
            len(indices),
            self.epochs,
            self.optimization_batch_size,
        )

        diagnostic_epochs = {0, self.epochs}
        diagnostic_epochs.update(
            range(self.diagnostic_interval, self.epochs + 1, self.diagnostic_interval)
        )
        checkpoints = []
        epoch_losses = []

        try:
            torch_model.eval()
            torch_model.requires_grad_(False)
            checkpoints.append(
                self._source_diagnostic(
                    torch_model,
                    tokenizer,
                    embedding_layer,
                    initial_embedding,
                    epoch=0,
                )
            )
            for epoch in range(self.epochs):
                rng.shuffle(indices)
                epoch_loss = 0.0
                updates = 0
                for start in range(0, len(indices), self.optimization_batch_size):
                    batch_indices = indices[start : start + self.optimization_batch_size]
                    inputs_embeds, attention_mask, labels = self._build_batch(
                        tokenizer,
                        embedding_layer,
                        adversarial_embedding,
                        batch_indices,
                    )

                    optimizer.zero_grad(set_to_none=True)
                    logits = torch_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    ).logits
                    token_losses = F.cross_entropy(
                        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(labels.shape[0], -1)
                    loss = token_losses.sum(dim=1).mean()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    updates += 1

                self.logger.info(
                    "PlugAE %s epoch %d/%d loss %.4f",
                    model.model_name,
                    epoch + 1,
                    self.epochs,
                    epoch_loss / max(updates, 1),
                )
                epoch_number = epoch + 1
                epoch_losses.append(
                    {
                        "epoch": epoch_number,
                        "mean_batch_target_loss": epoch_loss / max(updates, 1),
                    }
                )
                if epoch_number in diagnostic_epochs:
                    checkpoints.append(
                        self._source_diagnostic(
                            torch_model,
                            tokenizer,
                            embedding_layer,
                            adversarial_embedding.detach(),
                            epoch=epoch_number,
                        )
                    )
        finally:
            for parameter, flag in zip(torch_model.parameters(), requires_grad):
                parameter.requires_grad_(flag)
            torch_model.train(was_training)

        baseline = checkpoints[0]
        final = checkpoints[-1]
        quality_warnings = []
        if final["metrics"]["transfer_response_rate"] == 0:
            quality_warnings.append("plugae_source_self_test_zero_hits")
        if final["mean_target_nll"] >= baseline["mean_target_nll"]:
            quality_warnings.append("plugae_target_nll_did_not_improve")
        if (
            final["metrics"]["transfer_response_rate"]
            <= baseline["metrics"]["transfer_response_rate"]
        ):
            quality_warnings.append("plugae_trr_did_not_beat_random_initialization")

        return {
            "embedding": adversarial_embedding.detach().cpu(),
            "copyright_token": self.copyright_token,
            "query_sha256": self.query_sha256,
            "source_model": model.model_name,
            "optimization": {
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.optimization_batch_size,
                "seed": self.seed,
                "templates": [template[0] for template in PROFLINGO_TEMPLATES],
            },
            "training_diagnostics": {
                "epoch_losses": epoch_losses,
                "checkpoints": checkpoints,
                "random_initialization_epoch": 0,
                "quality_warnings": quality_warnings,
            },
        }

    def compare_fingerprints(self, base_model, testing_model):
        """Transfer the candidate embedding when appropriate and return TRR."""
        fingerprint = base_model.get_fingerprint()
        if not isinstance(fingerprint, dict) or "embedding" not in fingerprint:
            raise ValueError(f"Missing PlugAE fingerprint for {base_model.model_name}.")
        if fingerprint.get("query_sha256") != self.query_sha256:
            raise ValueError(
                f"Cached PlugAE queries for {base_model.model_name} do not match "
                f"{self.query_path}; regenerate its fingerprint."
            )
        if fingerprint.get("copyright_token") != self.copyright_token:
            raise ValueError(
                f"Cached PlugAE token for {base_model.model_name} does not match "
                "the current configuration; regenerate its fingerprint."
            )

        try:
            result = self._evaluate_transferred_embedding(
                base_model, testing_model, fingerprint, generation={}
            )
        except TokenEmbeddingOverrideError as exc:
            self.logger.warning(
                "Ignoring incompatible PlugAE comparison %s -> %s: %s",
                base_model.model_name,
                testing_model.model_name,
                exc,
            )
            return float("nan")
        score = result.score
        self.logger.info(
            "PlugAE TRR %s -> %s: %.4f (%d/%d)",
            base_model.model_name,
            testing_model.model_name,
            score,
            sum(item["success"] for item in result.trials),
            len(result.trials),
        )
        return score

    def fingerprint_to_records(self, fingerprint, source_model, experiment_id):
        if not isinstance(fingerprint, dict) or "embedding" not in fingerprint:
            raise ValueError("PlugAE fingerprint must contain an embedding.")
        payload = {
            "kind": "plugae",
            "embedding": fingerprint["embedding"].detach().cpu().tolist(),
            "copyright_token": fingerprint["copyright_token"],
            "query_sha256": fingerprint["query_sha256"],
            "queries": list(self.queries),
            "targets": list(self.targets),
            "keywords": list(self.keywords),
            "source_model": fingerprint.get("source_model", source_model.model_name),
            "optimization": fingerprint.get("optimization", {}),
            "output_parser": dict(self.output_parser or {}),
            "training_diagnostics": fingerprint.get("training_diagnostics", {}),
        }
        warnings = payload["training_diagnostics"].get("quality_warnings", [])
        return [
            self._record(
                experiment_id,
                1,
                source_model,
                payload,
                metadata={"quality_warnings": list(warnings)},
            )
        ]

    def fingerprint_from_records(self, records):
        if len(records) != 1 or records[0]["payload"].get("kind") != "plugae":
            raise ValueError("PlugAE batches must contain exactly one embedding artifact.")
        payload = records[0]["payload"]
        self._restore_evaluation_queries(payload)
        return {
            "embedding": torch.tensor(payload["embedding"]),
            "copyright_token": payload["copyright_token"],
            "query_sha256": payload["query_sha256"],
            "source_model": payload.get("source_model"),
            "optimization": payload.get("optimization", {}),
            "training_diagnostics": payload.get("training_diagnostics", {}),
        }

    def prepare_evaluation(self, records, train_models=None):
        del train_models
        if len(records) != 1 or records[0].get("payload", {}).get("kind") != "plugae":
            raise ValueError("PlugAE evaluation requires one embedding artifact.")
        self._restore_evaluation_queries(records[0]["payload"])

    def _restore_evaluation_queries(self, payload):
        queries = payload.get("queries")
        targets = payload.get("targets")
        keywords = payload.get("keywords")
        if not all(isinstance(values, list) for values in (queries, targets, keywords)):
            raise ValueError("PlugAE artifact does not contain its evaluation query set.")
        if not queries or not (len(queries) == len(targets) == len(keywords)):
            raise ValueError("PlugAE artifact query, target, and keyword counts differ.")
        query_payload = "\n".join(
            "\0".join((question, target, keyword))
            for question, target, keyword in zip(queries, targets, keywords)
        )
        query_sha256 = hashlib.sha256(query_payload.encode("utf-8")).hexdigest()
        if query_sha256 != payload.get("query_sha256"):
            raise ValueError("PlugAE artifact query hash is invalid.")
        self.queries = queries
        self.targets = targets
        self.keywords = keywords
        self.query_sha256 = query_sha256
        self.output_parser = payload.get("output_parser") or self._infer_output_parser(
            keywords
        )

    def verify_fingerprint(self, source_model, testing_model, generation=None):
        fingerprint = source_model.get_fingerprint()
        if not isinstance(fingerprint, dict) or "embedding" not in fingerprint:
            raise ValueError(f"Missing PlugAE fingerprint for {source_model.model_name}.")
        if fingerprint.get("query_sha256") != self.query_sha256:
            raise ValueError(
                "Saved PlugAE fingerprint was generated from a different query set."
            )
        if fingerprint.get("copyright_token") != self.copyright_token:
            raise ValueError(
                "Saved PlugAE fingerprint uses a different copyright token."
            )
        generation = dict(generation or {})
        seed = int(generation.pop("seed", 0))
        generation.pop("input_mode", None)
        set_seed(seed)

        generation["seed"] = seed
        return self._evaluate_transferred_embedding(
            source_model, testing_model, fingerprint, generation
        )

    def stealth_texts(self, records):
        del records
        return [
            {
                "fingerprint_id": f"query:{index:03d}:{template_id}",
                "kind": "trigger_prompt",
                "text": self._prompt_text(
                    template_id, prefix, suffix, self.copyright_token, question
                ),
            }
            for index, question in enumerate(self.queries, start=1)
            for template_id, prefix, suffix in PROFLINGO_TEMPLATES
        ]

    @staticmethod
    def _prompt_text(template_id, prefix, suffix, token, question):
        del template_id
        return f"{prefix}{token} simply answer: {question}{suffix}"

    def _prompt_specs(self, token):
        return [
            {
                "query_index": query_index,
                "template_id": template_id,
                "question": question,
                "keyword": keyword,
                "text": self._prompt_text(
                    template_id, prefix, suffix, token, question
                ),
                "prefix": prefix,
                "suffix_text": f" simply answer: {question}{suffix}",
            }
            for query_index, (question, keyword) in enumerate(
                zip(self.queries, self.keywords), start=1
            )
            for template_id, prefix, suffix in PROFLINGO_TEMPLATES
        ]

    @staticmethod
    def _infer_output_parser(keywords):
        digit_lengths = {
            len(str(keyword)) for keyword in keywords if str(keyword).isdigit()
        }
        if len(digit_lengths) == 1 and len(digit_lengths) == len(
            {len(str(keyword)) for keyword in keywords}
        ) and all(str(keyword).isdigit() for keyword in keywords):
            return {"kind": "fixed_digits", "length": next(iter(digit_lengths))}
        return {"kind": "nonempty_text"}

    def _parse_output(self, output, keyword):
        output = "" if output is None else str(output)
        parser = self.output_parser or self._infer_output_parser(self.keywords)
        if parser.get("kind") == "fixed_digits":
            length = int(parser["length"])
            match = re.search(rf"(?<!\d)\d{{{length}}}(?!\d)", output)
            parsed = match.group(0) if match else None
            return parsed, parsed == str(keyword), parsed is None
        invalid = not output.strip()
        return output.strip() or None, self._keyword_matches(output, keyword), invalid

    def _trials_and_metrics(self, outputs, *, seed, extra=None):
        specs = self._prompt_specs(self.copyright_token)
        if len(outputs) != len(specs):
            raise RuntimeError(
                f"PlugAE generated {len(outputs)} outputs for {len(specs)} prompts."
            )
        trials = []
        for spec, output in zip(specs, outputs):
            parsed, success, invalid = self._parse_output(output, spec["keyword"])
            trial = {
                "query_index": spec["query_index"],
                "template_id": spec["template_id"],
                "question": spec["question"],
                "keyword": spec["keyword"],
                "parsed_output": parsed,
                "output": "" if output is None else str(output),
                "success": int(success),
                "invalid": int(invalid),
                "seed": int(seed),
            }
            if extra:
                trial.update(extra)
            trials.append(trial)

        total = len(trials)
        rate = sum(item["success"] for item in trials) / total if total else 0.0
        invalid_rate = (
            sum(item["invalid"] for item in trials) / total if total else 0.0
        )
        by_template = {}
        for template_id, _, _ in PROFLINGO_TEMPLATES:
            selected = [item for item in trials if item["template_id"] == template_id]
            by_template[template_id] = (
                sum(item["success"] for item in selected) / len(selected)
                if selected
                else 0.0
            )
        query_groups = {}
        for item in trials:
            query_groups.setdefault(item["query_index"], []).append(item["success"])
        any_rate = (
            sum(any(values) for values in query_groups.values()) / len(query_groups)
            if query_groups
            else 0.0
        )
        all_rate = (
            sum(all(values) for values in query_groups.values()) / len(query_groups)
            if query_groups
            else 0.0
        )
        metrics = {
            "keyword_hit_rate": rate,
            "transfer_response_rate": rate,
            "invalid_rate": invalid_rate,
            "query_any_template_hit_rate": any_rate,
            "query_all_templates_hit_rate": all_rate,
            **{
                f"template_{template_id}_hit_rate": value
                for template_id, value in by_template.items()
            },
        }
        return trials, metrics

    def _validate_template_round_trip(self, tokenizer, token, specs):
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise TokenEmbeddingOverrideError(
                f"PlugAE copyright token {token!r} is not one tokenizer token."
            )
        for spec in specs:
            expected = (
                tokenizer.encode(spec["prefix"], add_special_tokens=True)
                + token_ids
                + tokenizer.encode(spec["suffix_text"], add_special_tokens=False)
            )
            actual = tokenizer.encode(spec["text"], add_special_tokens=True)
            if actual != expected:
                raise TokenEmbeddingOverrideError(
                    "PlugAE validation prompt does not reproduce the embedding "
                    f"position optimized by template {spec['template_id']}."
                )

    def _evaluate_transferred_embedding(
        self, source_model, testing_model, fingerprint, generation
    ):
        generation = dict(generation or {})
        seed = int(generation.pop("seed", 0))
        generation.pop("input_mode", None)
        set_seed(seed)
        token = fingerprint["copyright_token"]
        specs = self._prompt_specs(token)
        prompts = [spec["text"] for spec in specs]
        is_derivative = testing_model.pretrained_model == source_model.model_name
        override_name = testing_model.base_model
        override_registered = False
        try:
            if is_derivative:
                testing_model.model_pool.register_token_embedding_override(
                    override_name, token, fingerprint["embedding"]
                )
                override_registered = True
                tokenizer = testing_model.model_pool.get_tokenizer(override_name)
                self._validate_template_round_trip(tokenizer, token, specs)
            outputs = []
            for start in range(0, len(prompts), self.generation_batch_size):
                outputs.extend(
                    testing_model.generate(
                        prompts[start : start + self.generation_batch_size],
                        prompts_are_rendered=True,
                        **generation,
                    )
                )
        finally:
            if override_registered:
                testing_model.model_pool.clear_token_embedding_override(override_name)
        trials, metrics = self._trials_and_metrics(outputs, seed=seed)
        return FingerprintTestResult(
            score=metrics["transfer_response_rate"],
            metrics=metrics,
            trials=trials,
            metadata={
                "seed": seed,
                "embedding_transferred": is_derivative,
                "prompt_templates": [item[0] for item in PROFLINGO_TEMPLATES],
                "output_parser": dict(self.output_parser or {}),
            },
        )

    def _prompt_embedding_examples(self, tokenizer, embedding_layer, embedding):
        device = embedding_layer.weight.device
        dtype = embedding_layer.weight.dtype
        examples = []
        for question in self.queries:
            user_text = f" simply answer: {question}"
            for _, prefix, suffix in PROFLINGO_TEMPLATES:
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
                suffix_ids = tokenizer.encode(
                    user_text + suffix, add_special_tokens=False
                )
                with torch.no_grad():
                    prefix_embeds = embedding_layer(
                        torch.tensor(prefix_ids, device=device)
                    )
                    suffix_embeds = embedding_layer(
                        torch.tensor(suffix_ids, device=device)
                    )
                examples.append(
                    torch.cat(
                        [
                            prefix_embeds,
                            embedding.to(device=device, dtype=dtype).unsqueeze(0),
                            suffix_embeds,
                        ],
                        dim=0,
                    )
                )
        return examples

    def _generate_from_embedding(
        self, torch_model, tokenizer, embedding_layer, embedding
    ):
        examples = self._prompt_embedding_examples(
            tokenizer, embedding_layer, embedding
        )
        outputs = []
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        for start in range(0, len(examples), self.generation_batch_size):
            batch = examples[start : start + self.generation_batch_size]
            max_length = max(item.shape[0] for item in batch)
            width = batch[0].shape[1]
            padded = torch.zeros(
                (len(batch), max_length, width),
                device=batch[0].device,
                dtype=batch[0].dtype,
            )
            attention_mask = torch.zeros(
                (len(batch), max_length),
                device=batch[0].device,
                dtype=torch.long,
            )
            for row, item in enumerate(batch):
                padded[row, -item.shape[0] :] = item
                attention_mask[row, -item.shape[0] :] = 1
            with torch.no_grad():
                generated = torch_model.generate(
                    inputs_embeds=padded,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=self.source_self_test_max_new_tokens,
                    pad_token_id=pad_id,
                )
            outputs.extend(
                tokenizer.decode(item, skip_special_tokens=True)
                for item in generated
            )
        return outputs

    def _mean_target_nll(
        self, torch_model, tokenizer, embedding_layer, embedding
    ):
        loss_sum = 0.0
        token_count = 0
        indices = list(range(len(self.queries)))
        for start in range(0, len(indices), self.optimization_batch_size):
            batch_indices = indices[start : start + self.optimization_batch_size]
            inputs_embeds, attention_mask, labels = self._build_batch(
                tokenizer, embedding_layer, embedding, batch_indices
            )
            with torch.no_grad():
                logits = torch_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits
                losses = F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            loss_sum += float(losses.item())
            token_count += int((labels[:, 1:] != -100).sum().item())
        return loss_sum / max(token_count, 1)

    def _source_diagnostic(
        self, torch_model, tokenizer, embedding_layer, embedding, *, epoch
    ):
        outputs = self._generate_from_embedding(
            torch_model, tokenizer, embedding_layer, embedding
        )
        trials, metrics = self._trials_and_metrics(
            outputs, seed=self.seed, extra={"epoch": int(epoch)}
        )
        return {
            "epoch": int(epoch),
            "mean_target_nll": self._mean_target_nll(
                torch_model, tokenizer, embedding_layer, embedding
            ),
            "metrics": metrics,
            "trials": trials,
        }

    def _initialize_embedding(self, embedding_weights):
        weights = embedding_weights.detach().float()
        mean = weights.mean(dim=0)
        std = weights.std(dim=0).clamp_min(1e-6)
        generator = torch.Generator(device=weights.device)
        generator.manual_seed(self.seed)
        noise = torch.randn(
            mean.shape,
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        )
        return torch.nn.Parameter(mean + noise * std)

    def _build_batch(
        self, tokenizer, embedding_layer, adversarial_embedding, batch_indices
    ):
        examples = []
        example_labels = []
        device = embedding_layer.weight.device
        dtype = embedding_layer.weight.dtype

        for index in batch_indices:
            user_text = f" simply answer: {self.queries[index]}"
            target_ids = tokenizer.encode(
                f" {self.targets[index]}", add_special_tokens=False
            )
            if not target_ids:
                raise ValueError(f"Empty target tokenization for query {index}.")

            for _, prefix, suffix in PROFLINGO_TEMPLATES:
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
                suffix_ids = tokenizer.encode(
                    user_text + suffix, add_special_tokens=False
                )
                with torch.no_grad():
                    prefix_embeds = embedding_layer(
                        torch.tensor(prefix_ids, device=device)
                    ).detach()
                    suffix_embeds = embedding_layer(
                        torch.tensor(suffix_ids, device=device)
                    ).detach()
                    target_embeds = embedding_layer(
                        torch.tensor(target_ids, device=device)
                    ).detach()

                prompt_embeds = torch.cat(
                    [
                        prefix_embeds,
                        adversarial_embedding.to(dtype=dtype).unsqueeze(0),
                        suffix_embeds,
                    ],
                    dim=0,
                )
                full_embeds = torch.cat([prompt_embeds, target_embeds], dim=0)
                labels = torch.full(
                    (full_embeds.shape[0],), -100, device=device, dtype=torch.long
                )
                labels[prompt_embeds.shape[0] :] = torch.tensor(
                    target_ids, device=device
                )
                examples.append(full_embeds)
                example_labels.append(labels)

        inputs_embeds = pad_sequence(examples, batch_first=True)
        attention_mask = pad_sequence(
            [
                torch.ones(item.shape[0], device=device, dtype=torch.long)
                for item in examples
            ],
            batch_first=True,
        )
        labels = pad_sequence(example_labels, batch_first=True, padding_value=-100)
        return inputs_embeds, attention_mask, labels

    @staticmethod
    def _keyword_matches(output, keyword):
        if isinstance(output, list):
            output = output[0] if output else ""
        normalize = lambda value: re.sub(r"\s+", "", str(value)).casefold()
        return normalize(keyword) in normalize(output)
