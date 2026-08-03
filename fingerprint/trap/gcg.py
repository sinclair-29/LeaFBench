import logging
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DynamicCache


logger = logging.getLogger(__name__)


class GCGOptimizer:
    """Reference-style GCG with strict prompt round-trip validation."""

    _SUPPORTED_CONFIG = {
        "num_steps",
        "search_width",
        "batch_size",
        "topk",
        "n_replace",
        "optim_str_init",
        "seed",
        "use_prefix_cache",
        "allow_non_ascii",
        "filter_ids",
        "early_stop",
        "add_space_before_target",
    }

    def __init__(
        self,
        model,
        tokenizer,
        render_prompt,
        config,
        max_input_length=None,
    ):
        unknown = set(config) - self._SUPPORTED_CONFIG
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported TRAP GCG configuration option(s): {names}")

        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.render_prompt = render_prompt
        self.max_input_length = max_input_length
        self.embedding_layer = model.get_input_embeddings()
        self.device = self.embedding_layer.weight.device
        self.vocab_size = self.embedding_layer.num_embeddings

        self.num_steps = config.get("num_steps", 250)
        self.search_width = config.get("search_width", 512)
        self.batch_size = config.get("batch_size") or self.search_width
        self.topk = config.get("topk", 256)
        self.n_replace = config.get("n_replace", 1)
        self.optim_str_init = config.get(
            "optim_str_init",
            "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        )
        self.seed = config.get("seed")
        self.use_prefix_cache = config.get("use_prefix_cache", False)
        self.allow_non_ascii = config.get("allow_non_ascii", False)
        self.add_space_before_target = config.get("add_space_before_target", False)

        self._validate_config(config)
        self.not_allowed_ids = (
            None if self.allow_non_ascii else self._get_nonascii_token_ids()
        )

        self.before_ids = None
        self.after_ids = None
        self.target_ids = None
        self.prefix_cache = None
        self.instruction = None
        self.separator = None

    def _validate_config(self, config):
        integer_options = {
            "num_steps": self.num_steps,
            "search_width": self.search_width,
            "batch_size": self.batch_size,
            "topk": self.topk,
        }
        for name, value in integer_options.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"gcg_config.{name} must be a positive integer")
        if self.topk > self.vocab_size:
            raise ValueError(
                f"gcg_config.topk ({self.topk}) exceeds vocabulary size "
                f"({self.vocab_size})"
            )
        if self.n_replace != 1:
            raise ValueError("The reference GCG implementation requires n_replace: 1")
        if not isinstance(self.optim_str_init, str) or not self.optim_str_init:
            raise ValueError("gcg_config.optim_str_init must be a non-empty string")
        if config.get("filter_ids", True) is not True:
            raise ValueError(
                "gcg_config.filter_ids cannot be disabled because full-prompt "
                "round-trip validation is required"
            )
        if config.get("early_stop", False) is not False:
            raise ValueError("gcg_config.early_stop is not supported by the reference GCG")

    def _get_nonascii_token_ids(self):
        special_ids = set(self.tokenizer.all_special_ids)
        disallowed = []
        for token_id in range(self.vocab_size):
            try:
                token = self.tokenizer.decode([token_id])
            except (IndexError, KeyError, ValueError):
                disallowed.append(token_id)
                continue
            if (
                token_id in special_ids
                or not token.isascii()
                or not token.isprintable()
            ):
                disallowed.append(token_id)
        return torch.tensor(disallowed, device=self.device, dtype=torch.long)

    def _encode(self, text, *, add_special_tokens):
        encoded = self.tokenizer(text, add_special_tokens=add_special_tokens)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return input_ids

    def _decode_control(self, control_ids):
        prefix_ids = self.before_ids[0].tolist()
        prefix_text = self.tokenizer.decode(
            prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        decoded = self.tokenizer.decode(
            prefix_ids + control_ids.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not decoded.startswith(prefix_text):
            raise RuntimeError("Tokenizer could not decode the GCG suffix in context")
        return decoded[len(prefix_text):]

    def _fingerprint_for_ids(self, control_ids):
        return self.instruction + self._decode_control(control_ids)

    def _expected_context_ids(self, control_ids):
        return (
            self.before_ids[0].tolist()
            + control_ids.tolist()
            + self.after_ids[0].tolist()
        )

    def _actual_context_ids(self, control_ids):
        fingerprint = self._fingerprint_for_ids(control_ids)
        rendered = self.render_prompt(fingerprint)
        context_ids = self._encode(rendered, add_special_tokens=True)
        if (
            self.max_input_length is not None
            and len(context_ids) > self.max_input_length
        ):
            raise RuntimeError(
                "TRAP optimization context exceeds the source model's "
                f"max_input_length ({self.max_input_length})"
            )
        return context_ids

    def _is_losslessly_serializable(self, control_ids):
        return self._actual_context_ids(control_ids) == self._expected_context_ids(
            control_ids
        )

    def _prepare_prompt(self, instruction, target):
        self.instruction = instruction
        self.separator = " " if instruction else ""

        marker = "{optim_str}"
        if marker in instruction:
            raise ValueError(f"TRAP prompt cannot contain the reserved marker {marker!r}")
        rendered_template = self.render_prompt(instruction + marker)
        if rendered_template.count(marker) != 1:
            raise RuntimeError("Prompt renderer did not preserve the TRAP suffix marker")
        before_text, after_text = rendered_template.split(marker)

        initial_control = self.separator + self.optim_str_init
        fingerprint = instruction + initial_control
        rendered_context = self.render_prompt(fingerprint)
        expected_context = before_text + initial_control + after_text
        if rendered_context != expected_context:
            raise RuntimeError(
                "Prompt renderer changed content while locating the GCG suffix"
            )

        try:
            encoded_context = self.tokenizer(
                rendered_context,
                add_special_tokens=True,
                return_offsets_mapping=True,
            )
            context_ids = encoded_context["input_ids"]
            offsets = encoded_context["offset_mapping"]
        except (KeyError, NotImplementedError, TypeError, ValueError) as error:
            raise RuntimeError(
                "TRAP GCG requires a fast tokenizer with offset mappings so the "
                "optimization slice can be derived from the complete prompt"
            ) from error

        control_start = len(before_text)
        control_end = control_start + len(initial_control)
        control_token_indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > control_start and start < control_end
        ]
        if not control_token_indices:
            raise RuntimeError("Tokenizer produced no tokens for the GCG suffix")
        first_control = control_token_indices[0]
        last_control = control_token_indices[-1] + 1
        if control_token_indices != list(range(first_control, last_control)):
            raise RuntimeError("GCG suffix tokens are not a contiguous prompt slice")

        first_start = offsets[first_control][0]
        last_end = offsets[last_control - 1][1]
        if first_start < control_start or last_end > control_end:
            raise RuntimeError(
                "A tokenizer token crosses the fixed prompt/GCG suffix boundary"
            )

        self.before_ids = torch.tensor(
            [context_ids[:first_control]], device=self.device, dtype=torch.long
        )
        current_ids = torch.tensor(
            [context_ids[first_control:last_control]],
            device=self.device,
            dtype=torch.long,
        )
        self.after_ids = torch.tensor(
            [context_ids[last_control:]], device=self.device, dtype=torch.long
        )
        target_text = " " + target if self.add_space_before_target else target
        self.target_ids = torch.tensor(
            [self._encode(target_text, add_special_tokens=False)],
            device=self.device,
            dtype=torch.long,
        )
        if self.target_ids.shape[1] == 0:
            raise ValueError("TRAP target must contain at least one token")

        if not self._is_losslessly_serializable(current_ids[0]):
            raise RuntimeError(
                "The initial GCG suffix cannot be reconstructed as the exact prompt "
                "token sequence used for optimization"
            )

        self.prefix_cache = None
        if self.use_prefix_cache:
            with torch.no_grad():
                output = self.model(input_ids=self.before_ids, use_cache=True)
            cache = output.past_key_values
            if isinstance(cache, DynamicCache):
                self.prefix_cache = cache
            else:
                self.prefix_cache = DynamicCache.from_legacy_cache(cache)

        return current_ids

    def _sample_ids(self, current_ids, gradient):
        gradient = gradient.clone()
        if self.not_allowed_ids is not None:
            gradient[:, self.not_allowed_ids] = float("inf")

        topk_ids = (-gradient).topk(self.topk, dim=1).indices
        suffix_length = current_ids.shape[0]
        positions = torch.floor(
            torch.arange(self.search_width, device=self.device)
            * suffix_length
            / self.search_width
        ).to(torch.long)
        candidates = current_ids.repeat(self.search_width, 1)

        sampled_ranks = torch.empty(
            self.search_width, device=self.device, dtype=torch.long
        )
        for position in range(suffix_length):
            indices = torch.nonzero(positions == position, as_tuple=False).flatten()
            count = indices.numel()
            if count == 0:
                continue
            ranks = torch.cat(
                [
                    torch.randperm(self.topk, device=self.device)
                    for _ in range((count + self.topk - 1) // self.topk)
                ]
            )[:count]
            sampled_ranks[indices] = ranks

        replacements = topk_ids[positions, sampled_ranks]
        candidates.scatter_(1, positions.unsqueeze(1), replacements.unsqueeze(1))
        return candidates

    def _filter_candidates(self, current_ids, sampled_ids):
        # The unchanged current suffix is valid by induction and prevents an
        # empty candidate set if every newly sampled mutation changes a boundary.
        candidate_rows = torch.cat([current_ids, sampled_ids], dim=0)
        valid = []
        seen = set()
        for row in candidate_rows:
            key = tuple(row.tolist())
            if key in seen:
                continue
            seen.add(key)
            if self._is_losslessly_serializable(row):
                valid.append(row)

        if not valid:
            raise RuntimeError("Internal error: the validated current suffix was lost")
        rejected = len(seen) - len(valid)
        if rejected:
            logger.debug(
                "Rejected %d GCG candidate(s) whose verification prompt would "
                "not match optimization",
                rejected,
            )
        return torch.stack(valid)

    def _expanded_prefix_cache(self, batch_size):
        if self.prefix_cache is None:
            return None
        expanded = DynamicCache()
        for layer_idx in range(len(self.prefix_cache)):
            key = self.prefix_cache.key_cache[layer_idx]
            value = self.prefix_cache.value_cache[layer_idx]
            expanded.update(
                key.expand(batch_size, -1, -1, -1),
                value.expand(batch_size, -1, -1, -1),
                layer_idx,
            )
        return expanded

    def _build_embeddings(self, control_ids):
        batch_size = control_ids.shape[0]
        pieces = []
        if not self.use_prefix_cache:
            pieces.append(
                self.embedding_layer(self.before_ids).expand(batch_size, -1, -1)
            )
        pieces.extend(
            [
                self.embedding_layer(control_ids),
                self.embedding_layer(self.after_ids).expand(batch_size, -1, -1),
                self.embedding_layer(self.target_ids).expand(batch_size, -1, -1),
            ]
        )
        return torch.cat(pieces, dim=1)

    def _target_loss(self, logits, input_length, batch_size):
        target_length = self.target_ids.shape[1]
        target_logits = logits[..., input_length - target_length - 1 : -1, :]
        labels = self.target_ids.expand(batch_size, -1)
        loss = F.cross_entropy(
            target_logits.reshape(-1, target_logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        )
        return loss.view(batch_size, -1).mean(dim=1)

    def _compute_gradient(self, current_ids):
        one_hot = F.one_hot(current_ids, num_classes=self.vocab_size).to(
            self.embedding_layer.weight.dtype
        )
        one_hot.requires_grad_(True)
        control_embeds = one_hot @ self.embedding_layer.weight

        pieces = []
        if not self.use_prefix_cache:
            pieces.append(self.embedding_layer(self.before_ids))
        pieces.extend(
            [
                control_embeds,
                self.embedding_layer(self.after_ids),
                self.embedding_layer(self.target_ids),
            ]
        )
        input_embeds = torch.cat(pieces, dim=1)
        output = self.model(
            inputs_embeds=input_embeds,
            past_key_values=self._expanded_prefix_cache(1),
            use_cache=self.use_prefix_cache,
        )
        loss = self._target_loss(output.logits, input_embeds.shape[1], 1).mean()
        return torch.autograd.grad(loss, one_hot)[0].squeeze(0)

    def _calculate_losses(self, control_ids):
        losses = []
        for start in range(0, control_ids.shape[0], self.batch_size):
            batch_ids = control_ids[start : start + self.batch_size]
            input_embeds = self._build_embeddings(batch_ids)
            with torch.no_grad():
                output = self.model(
                    inputs_embeds=input_embeds,
                    past_key_values=self._expanded_prefix_cache(batch_ids.shape[0]),
                    use_cache=self.use_prefix_cache,
                )
            losses.append(
                self._target_loss(
                    output.logits,
                    input_embeds.shape[1],
                    batch_ids.shape[0],
                )
            )
        return torch.cat(losses)

    def optimize(self, instruction, target):
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

        current_ids = self._prepare_prompt(instruction, target)
        best_ids = current_ids[0].clone()
        best_loss = float("inf")

        for step in tqdm(range(self.num_steps), desc="TRAP GCG"):
            gradient = self._compute_gradient(current_ids)
            with torch.no_grad():
                sampled_ids = self._sample_ids(current_ids[0], gradient)
                candidates = self._filter_candidates(current_ids, sampled_ids)
                losses = self._calculate_losses(candidates)
                best_index = losses.argmin()
                current_ids = candidates[best_index].unsqueeze(0)
                current_loss = losses[best_index].item()

            if current_loss < best_loss:
                best_loss = current_loss
                best_ids = current_ids[0].clone()
            logger.debug(
                "TRAP GCG step %d/%d: current loss %.6f, best loss %.6f",
                step + 1,
                self.num_steps,
                current_loss,
                best_loss,
            )

        if not self._is_losslessly_serializable(best_ids):
            raise RuntimeError(
                "Final TRAP fingerprint differs between optimization and verification"
            )
        fingerprint = self._fingerprint_for_ids(best_ids)
        self.before_ids = None
        self.after_ids = None
        self.target_ids = None
        self.prefix_cache = None
        return best_loss, fingerprint
