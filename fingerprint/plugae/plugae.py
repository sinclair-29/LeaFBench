import csv
import hashlib
import logging
import random
import re

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from benchmark.model_pool import TokenEmbeddingOverrideError
from fingerprint.fingerprint_interface import LLMFingerprintInterface


PROFLINGO_TEMPLATES = (
    (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request. ### Instruction: ",
        " ### Response:",
    ),
    (
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

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config or {}, accelerator=accelerator)
        self.query_path = self.config.get("query_path", "data/plugae_questions.csv")
        self.num_queries = self.config.get("num_queries", 50)
        self.copyright_token = self.config.get("copyright_token", "mkahg")
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.epochs = self.config.get("epochs", 30)
        self.optimization_batch_size = self.config.get("optimization_batch_size", 8)
        self.generation_batch_size = self.config.get("generation_batch_size", 16)
        self.seed = self.config.get("seed", 42)
        if not self.copyright_token:
            raise ValueError("PlugAE copyright_token must not be empty.")
        if self.num_queries is not None and self.num_queries < 1:
            raise ValueError("PlugAE num_queries must be at least 1.")
        if self.epochs < 1:
            raise ValueError("PlugAE epochs must be at least 1.")
        if self.optimization_batch_size < 1 or self.generation_batch_size < 1:
            raise ValueError("PlugAE batch sizes must be at least 1.")
        self.queries = []
        self.targets = []
        self.keywords = []
        self.query_sha256 = None

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

        try:
            torch_model.eval()
            torch_model.requires_grad_(False)
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
        finally:
            for parameter, flag in zip(torch_model.parameters(), requires_grad):
                parameter.requires_grad_(flag)
            torch_model.train(was_training)

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
                "templates": ["alpaca", "zero_shot"],
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

        prompts = [
            f"{fingerprint['copyright_token']} simply answer: {question}"
            for question in self.queries
        ]
        is_derivative = testing_model.pretrained_model == base_model.model_name
        override_name = testing_model.base_model
        override_registered = False

        try:
            if is_derivative:
                testing_model.model_pool.register_token_embedding_override(
                    override_name,
                    fingerprint["copyright_token"],
                    fingerprint["embedding"],
                )
                override_registered = True

            outputs = []
            for start in range(0, len(prompts), self.generation_batch_size):
                outputs.extend(
                    testing_model.generate(
                        prompts[start : start + self.generation_batch_size]
                    )
                )
        except TokenEmbeddingOverrideError as exc:
            self.logger.warning(
                "Ignoring incompatible PlugAE comparison %s -> %s: %s",
                base_model.model_name,
                testing_model.model_name,
                exc,
            )
            return float("nan")
        finally:
            if override_registered:
                testing_model.model_pool.clear_token_embedding_override(override_name)

        matches = sum(
            self._keyword_matches(output, keyword)
            for output, keyword in zip(outputs, self.keywords)
        )
        score = matches / len(self.keywords) if self.keywords else 0.0
        self.logger.info(
            "PlugAE TRR %s -> %s: %.4f (%d/%d)",
            base_model.model_name,
            testing_model.model_name,
            score,
            matches,
            len(self.keywords),
        )
        return score

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

            for prefix, suffix in PROFLINGO_TEMPLATES:
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
