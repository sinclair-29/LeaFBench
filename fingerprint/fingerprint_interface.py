import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class FingerprintTestResult:
    """Structured result returned by method-specific fingerprint verification."""

    score: float
    metrics: Dict[str, float]
    trials: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)



class LLMFingerprintInterface:
    """
    Interface for LLM fingerprinting.
    """

    # Symmetric methods fingerprint every model before comparison. Asymmetric
    # methods can override these properties to fingerprint candidates only.
    requires_suspect_fingerprints = True
    candidate_model_types = ("pretrained", "instruct")
    evaluation_capabilities = {
        "model_modification_robustness": True,
        "deployment_robustness": {
            "system_prompts": True,
            "sampling": True,
        },
        "model_specificity": True,
        "prompt_stealthiness": True,
    }

    def __init__(self, config=None, accelerator=None):
        self.config = config
        self.accelerator = accelerator

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods. For example, this could involve training fingerprinting classifiers.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        pass

    def prepare_evaluation(self, records, train_models=None):
        """Prepare candidate verification from a saved source artifact batch."""
        del records
        self.prepare(train_models=train_models)
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given text.

        Args:
            text (str): The input text to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def fingerprint_to_records(self, fingerprint, source_model, experiment_id):
        """Convert a native fingerprint into numbered JSON-compatible records."""
        if isinstance(fingerprint, torch.Tensor):
            tensor = fingerprint.detach().cpu()
            rows = tensor if tensor.ndim > 1 else tensor.unsqueeze(0)
            return [
                self._record(
                    experiment_id,
                    index,
                    source_model,
                    {
                        "kind": "tensor",
                        "values": row.tolist(),
                        "original_shape": list(tensor.shape),
                    },
                )
                for index, row in enumerate(rows, start=1)
            ]
        if isinstance(fingerprint, list):
            return [
                self._record(
                    experiment_id,
                    index,
                    source_model,
                    {"kind": "value", "value": value},
                )
                for index, value in enumerate(fingerprint, start=1)
            ]
        raise TypeError(
            f"{type(self).__name__} cannot serialize fingerprint type "
            f"{type(fingerprint).__name__}"
        )

    def fingerprint_from_records(self, records):
        """Reconstruct a native fingerprint from numbered artifact records."""
        if not records:
            raise ValueError("A fingerprint batch must contain at least one artifact.")
        payloads = [record["payload"] for record in records]
        kinds = {payload.get("kind") for payload in payloads}
        if kinds == {"tensor"}:
            rows = [torch.tensor(payload["values"]) for payload in payloads]
            tensor = torch.stack(rows)
            original_shape = payloads[0].get("original_shape")
            if original_shape is not None:
                if any(
                    payload.get("original_shape") != original_shape
                    for payload in payloads
                ):
                    raise ValueError("Tensor artifact shape metadata is inconsistent.")
                tensor = tensor.reshape(original_shape)
            return tensor
        if kinds == {"value"}:
            return [payload["value"] for payload in payloads]
        raise ValueError(f"Unsupported or mixed artifact payload kinds: {sorted(kinds)}")

    def verify_fingerprint(self, source_model, testing_model, generation=None):
        """Verify a saved fingerprint against one model and return structured data."""
        from transformers import set_seed

        generation = dict(generation or {})
        set_seed(int(generation.get("seed", 0)))
        candidate_fingerprint = self.get_fingerprint(testing_model)
        if source_model is testing_model:
            testing_view = copy.copy(testing_model)
            testing_view.set_fingerprint(candidate_fingerprint)
            score = float(self.compare_fingerprints(source_model, testing_view))
            return FingerprintTestResult(score=score, metrics={"similarity": score})

        previous = testing_model.get_fingerprint()
        try:
            testing_model.set_fingerprint(candidate_fingerprint)
            score = float(self.compare_fingerprints(source_model, testing_model))
        finally:
            testing_model.set_fingerprint(previous)
        return FingerprintTestResult(score=score, metrics={"similarity": score})

    def stealth_texts(self, records):
        """Return named natural-language inputs for prompt stealth evaluation."""
        texts = []
        for record in records:
            payload = record.get("payload", {})
            if payload.get("kind") == "value" and isinstance(payload.get("value"), str):
                texts.append({"fingerprint_id": record["fingerprint_id"], "kind": "prompt", "text": payload["value"]})
        return texts

    def supports_evaluation(self, evaluation_name, component=None):
        capability = self.evaluation_capabilities.get(evaluation_name, False)
        if component is None:
            return bool(capability)
        return bool(capability.get(component, False)) if isinstance(capability, dict) else False

    @staticmethod
    def _record(experiment_id, index, source_model, payload, metadata=None):
        return {
            "schema_version": 1,
            "fingerprint_id": f"{experiment_id}:{index:03d}",
            "item_index": index,
            "source_model": source_model.model_name,
            "payload": payload,
            "metadata": metadata or {},
        }
