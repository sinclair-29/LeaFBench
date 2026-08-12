import torch
import numpy as np

from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.reef.generate_activation import load_statements, get_acts
from fingerprint.reef.compute_cka import CKA


class REEFFingerprint(LLMFingerprintInterface):
    evaluation_capabilities = {
        "model_modification_robustness": True,
        "deployment_robustness": {"system_prompts": True, "sampling": False},
        "model_specificity": True,
        "prompt_stealthiness": False,
    }

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)

    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods. For example, this could involve training fingerprinting classifiers.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        dataset_path = self.config.get('dataset_path', None)
        num_samples = self.config.get('num_samples', 200)
        self.layers = self.config.get('layers', 18)
        statements = load_statements(dataset_path)
        if num_samples > len(statements):
            raise ValueError(
                f"REEF requested {num_samples} statements but only "
                f"{len(statements)} are available."
            )
        rng = np.random.default_rng(int(self.config.get('seed', 42)))
        indices = rng.choice(len(statements), size=num_samples, replace=False)
        self.statements = [statements[index] for index in indices]
        self.batch_size = self.config.get('batch_size', 1)

    def prepare_evaluation(self, records, train_models=None):
        del train_models
        if not records:
            raise ValueError("REEF evaluation requires saved activation artifacts.")
        self.layers = self.config.get('layers', 18)
        self.batch_size = self.config.get('batch_size', 1)
        self.statements = [record["payload"]["statement"] for record in records]
    
    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given text.

        Args:
            text (str): The input text to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        torch_model, tokenizer = model.load_model()
        statements = model.render_prompts(self.statements, tokenizer)
        device = (
            self.accelerator.device
            if self.accelerator is not None
            else next(torch_model.parameters()).device
        )
        fingerprint = get_acts(
            statements, tokenizer, torch_model,
            model.model_family, 
            self.layers,
            device,
            batch_size=self.batch_size
        )
        return fingerprint
        
    
    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        device = (
            self.accelerator.device
            if self.accelerator is not None
            else base_model.get_fingerprint().device
        )
        cka = CKA(device)
        base_fingerprint = base_model.get_fingerprint()
        print(f"Base fingerprint shape: {base_fingerprint.shape}")
        testing_fingerprint = testing_model.get_fingerprint()
        base_fingerprint = base_fingerprint.to(device)
        testing_fingerprint = testing_fingerprint.to(device)
        cka_value = cka.linear_CKA(base_fingerprint, testing_fingerprint)
        return cka_value.item()

    def fingerprint_to_records(self, fingerprint, source_model, experiment_id):
        tensor = fingerprint.detach().cpu()
        if tensor.ndim != 2 or tensor.shape[0] != len(self.statements):
            raise ValueError(
                "REEF activations must have one matrix row per prepared statement."
            )
        return [
            self._record(
                experiment_id,
                index,
                source_model,
                {
                    "kind": "reef_activation",
                    "statement": self.statements[index - 1],
                    "values": row.tolist(),
                    "layers": self.layers,
                },
            )
            for index, row in enumerate(tensor, start=1)
        ]

    def fingerprint_from_records(self, records):
        if not records or any(
            record["payload"].get("kind") != "reef_activation"
            for record in records
        ):
            raise ValueError("REEF batches must contain activation artifacts.")
        layer_specs = {str(record["payload"].get("layers")) for record in records}
        if len(layer_specs) != 1:
            raise ValueError("REEF artifact layer specifications are inconsistent.")
        self.statements = [record["payload"]["statement"] for record in records]
        return torch.stack(
            [torch.tensor(record["payload"]["values"]) for record in records]
        )
