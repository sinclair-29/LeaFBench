from __future__ import annotations

from typing import Any, Sequence

import torch
from transformers import LogitsProcessorList

from benchmark.model_interface import ModelInterface
from deploying_techniques.watermark.config import WatermarkConfig
from deploying_techniques.watermark.detector import WatermarkDetector
from deploying_techniques.watermark.processor import WatermarkLogitsProcessor


class WatermarkedModel(ModelInterface):
    """A LeaFBench model wrapper that changes only generation logits."""

    def __init__(self, config, source_model: ModelInterface, model_pool=None, accelerator=None):
        super().__init__(config, model_pool=model_pool, accelerator=accelerator)
        self.source_model = source_model
        self.watermark_config = WatermarkConfig.from_mapping(config["watermark"])
        self.last_generation: list[dict[str, Any]] = []
        self.last_unwatermarked_generation: list[dict[str, Any]] = []

    def render_prompts(self, prompts, tokenizer):
        return self.source_model.render_prompts(prompts, tokenizer)

    def _device(self, model):
        return self.accelerator.device if self.accelerator is not None else model.device

    def _generation_params(self, tokenizer, overrides: dict[str, Any]) -> dict[str, Any]:
        do_sample = self.params.get("do_sample", True)
        params = {
            "max_new_tokens": self.params.get("max_new_tokens", 200),
            "do_sample": do_sample,
            "num_beams": self.params.get("num_beams", 1),
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            params.update(
                {
                    "temperature": self.params.get("temperature", 1.0),
                    "top_p": self.params.get("top_p", 1.0),
                    "top_k": self.params.get("top_k", 0),
                }
            )
        for optional in ("min_new_tokens", "min_length", "no_repeat_ngram_size"):
            if optional in self.params:
                params[optional] = self.params[optional]
        params.update(overrides)
        if self.params.get("suppress_eos", False) and tokenizer.eos_token_id is not None:
            params["suppress_tokens"] = [tokenizer.eos_token_id]
        return params

    def _generate(self, prompts: Sequence[str], apply_watermark: bool, **kwargs) -> list[str]:
        model, tokenizer = self.load_model()
        rendered = self.render_prompts(prompts, tokenizer)
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            list(rendered),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.params.get("max_input_length", 2048),
        )
        device = self._device(model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        vocab_size = min(len(tokenizer), int(model.config.vocab_size))
        generation_params = self._generation_params(tokenizer, kwargs)
        if apply_watermark:
            processor = WatermarkLogitsProcessor(self.watermark_config, vocab_size)
            generation_params["logits_processor"] = LogitsProcessorList([processor])

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_params)

        input_width = inputs["input_ids"].shape[1]
        generation_records = []
        generated_texts = []
        for index, output in enumerate(output_ids):
            generated_ids = output[input_width:].detach().cpu().tolist()
            prompt_ids = inputs["input_ids"][index]
            prompt_ids = prompt_ids[inputs["attention_mask"][index].bool()].detach().cpu().tolist()
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(text)
            generation_records.append(
                {
                    "text": text,
                    "prompt_ids": prompt_ids,
                    "generated_ids": generated_ids,
                }
            )
        if apply_watermark:
            self.last_generation = generation_records
        else:
            self.last_unwatermarked_generation = generation_records
        return generated_texts

    def generate(self, prompts: Sequence[str], **kwargs) -> list[str]:
        return self._generate(prompts, apply_watermark=True, **kwargs)

    def generate_unwatermarked(self, prompts: Sequence[str], **kwargs) -> list[str]:
        return self._generate(prompts, apply_watermark=False, **kwargs)

    def detect(self, outputs: Sequence[str]) -> list[dict[str, Any]]:
        model, tokenizer = self.load_model()
        vocab_size = min(len(tokenizer), int(model.config.vocab_size))
        detector = WatermarkDetector(
            self.watermark_config,
            vocab_size,
            self._device(model),
            model=model,
        )
        output_list = list(outputs)
        cached_texts = [record["text"] for record in self.last_generation]
        cached_unwatermarked_texts = [
            record["text"] for record in self.last_unwatermarked_generation
        ]
        cached_records = None
        if output_list == cached_texts:
            cached_records = self.last_generation
        elif output_list == cached_unwatermarked_texts:
            cached_records = self.last_unwatermarked_generation
        if cached_records is not None:
            terminal_ids = {
                token_id
                for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
                if token_id is not None
            }
            results = []
            for record in cached_records:
                generated_ids = list(record["generated_ids"])
                while generated_ids and generated_ids[-1] in terminal_ids:
                    generated_ids.pop()
                results.append(detector.detect_token_ids(generated_ids, record["prompt_ids"]))
            return results
        return [detector.detect_text(text, tokenizer) for text in output_list]
