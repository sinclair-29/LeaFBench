import tokenize
from tqdm import tqdm
from transformers import set_seed

from fingerprint.fingerprint_interface import (
    FingerprintTestResult,
    LLMFingerprintInterface,
)
import os
import random
import pandas as pd
import re
from fingerprint.trap.generate_prompts import generate_csv, generate_adversarial_suffix
import numpy as np
from collections import defaultdict



class TRAPFingerprint(LLMFingerprintInterface):
    """
    TRAP Fingerprint Class
    """

    candidate_model_types = ("pretrained", "instruct", "instruction_tuning")

    evaluation_capabilities = {
        "model_modification_robustness": True,
        "deployment_robustness": {"system_prompts": True, "sampling": True},
        "model_specificity": True,
        "prompt_stealthiness": True,
    }

    def __init__(self, config=None, accelerator=None):
        super().__init__(config=config, accelerator=accelerator)
        self.n_goals = self.config.get('n_goals', 100)
        self.goal_offset = self.config.get('goal_offset', 0)
        self.goal_count = self.config.get('goal_count')
        self.prompt_seed = self.config.get('prompt_seed', self.config.get('seed', 42))
        self.string_type = self.config.get('string_type', 'number')
        self.string_length = self.config.get('string_length', 3)
        self.prompt_path = self.config.get('prompt_path', None)
        self.gcg_config = self.config.get('gcg_config', {})
        # self.filtered_tokens_path = self.config.get('filtered_tokens_path', None)
        # self.filter_words_path = self.config.get('filter_words_path', "data/filter_words_number.csv")
        self.test_n_times = self.config.get('test_n_times', 5)
        self.batch_size = self.config.get('batch_size', 16)


    def prepare(self, train_models=None):
        """
        Prepare the fingerprinting methods. For example, this could involve training fingerprinting classifiers.

        Args:
            train_models (optional): Models to train, if necessary.
        """
        if os.path.exists(self.prompt_path) and not self.config.get('regenerate_prompts', False):
            df = pd.read_csv(self.prompt_path, dtype={'prompt': str, 'target': str, 'string_target': str})
        else:
            random.seed(self.prompt_seed)
            df = generate_csv(self.n_goals, self.string_type, self.string_length, self.prompt_path)
        end = None if self.goal_count is None else self.goal_offset + self.goal_count
        df = df.iloc[self.goal_offset:end]
        if df.empty:
            raise ValueError("TRAP goal shard is empty")
        self.prompts = df['prompt'].tolist()
        self.targets = df['target'].tolist()
        self.string_target = df['string_target'].tolist()

    def prepare_evaluation(self, records, train_models=None):
        del train_models
        if not records or any(
            record.get("payload", {}).get("kind") != "trap"
            for record in records
        ):
            raise ValueError("TRAP evaluation requires saved TRAP artifacts.")


    def get_fingerprint(self, model):
        """
        Generate a fingerprint for the given text.

        Args:
            text (str): The input text to fingerprint.

        Returns:
            torch.Tensor: The fingerprint tensor.
        """
        # only extract fingerprint if the model is pretrained or instruct model
        if model.model_name == model.pretrained_model or model.model_name == model.instruct_model:
            torch_model, tokenizer = model.load_model()
            render_prompt = lambda prompt: model.render_prompts([prompt], tokenizer)[0]
            generated_prompts = generate_adversarial_suffix(
                torch_model,
                tokenizer,
                self.prompts,
                self.targets,
                self.gcg_config,
                render_prompt,
                max_input_length=(model.params or {}).get('max_input_length', 512),
            )
            fingerprint = generated_prompts
            return fingerprint
        else:
            return 0

    def compare_fingerprints(self, base_model, testing_model):
        """
        Compare two models using their fingerprints with batch processing optimization.

        Args:
            base_model (ModelInterface): The base model to compare against.
            testing_model (ModelInterface): The model to compare.
            batch_size (int): Number of prompts to process in each batch.

        Returns:
            float: Similarity score between the two fingerprints.
        """
        base_fingerprint = base_model.get_fingerprint()
        batch_size = self.batch_size
        
        if not base_fingerprint or len(base_fingerprint) == 0:
            return 0.0
            
        total_matches = 0
        total_tests = 0
        
        # Create all test cases upfront for batch processing
        all_prompts = []
        all_targets = []
        all_prompt_indices = []
        
        for k, prompt in enumerate(base_fingerprint):
            target_string = self.string_target[k]
            # Repeat each prompt test_n_times
            for _ in range(self.test_n_times):
                all_prompts.append(prompt)
                all_targets.append(target_string)
                all_prompt_indices.append(k)
        
        print(f"Processing {len(all_prompts)} total tests in batches of {batch_size}")
        
        # Process in batches
        for i in tqdm(range(0, len(all_prompts), batch_size), desc="Processing batches"):
            batch_prompts = all_prompts[i:i+batch_size]
            batch_targets = all_targets[i:i+batch_size]
            batch_indices = all_prompt_indices[i:i+batch_size]
            
            try:
                # Generate responses for the entire batch
                batch_answers = testing_model.generate(batch_prompts)
                
                # Process batch results
                for j, (generated_text, target_string, prompt_idx) in enumerate(zip(batch_answers, batch_targets, batch_indices)):
                    if isinstance(generated_text, list):
                        generated_text = generated_text[0] if generated_text else ""
                    
                    if str(target_string) in generated_text:
                        total_matches += 1
                    
                    total_tests += 1
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Count failed tests
                total_tests += len(batch_prompts)
        
        # Calculate overall similarity score as the proportion of successful tests
        similarity_score = total_matches / total_tests if total_tests > 0 else 0.0
        print(f"Overall similarity score: {similarity_score:.4f} ({total_matches}/{total_tests})")
        
        return similarity_score

    def fingerprint_to_records(self, fingerprint, source_model, experiment_id):
        if len(fingerprint) != len(self.string_target):
            raise ValueError(
                "TRAP fingerprint count does not match its prepared target count."
            )
        _, tokenizer = source_model.load_model()
        rendered = source_model.render_prompts(fingerprint, tokenizer)
        records = []
        for index, raw_prompt in enumerate(fingerprint, start=1):
            instruction = self.prompts[index - 1]
            optimized_text = (
                raw_prompt[len(instruction) :]
                if raw_prompt.startswith(instruction)
                else raw_prompt
            )
            records.append(
                self._record(
                    experiment_id,
                    index,
                    source_model,
                    {
                        "kind": "trap",
                        "instruction": instruction,
                        "raw_user_prompt": raw_prompt,
                        "rendered_prompt": rendered[index - 1],
                        "optimized_text": optimized_text,
                        "target": str(self.string_target[index - 1]),
                    },
                )
            )
        return records

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
        set_seed(seed)

        prompts = [
            record["payload"][
                "rendered_prompt" if input_mode == "source_rendered" else "raw_user_prompt"
            ]
            for record in records
        ]
        outputs = []
        for start in range(0, len(prompts), self.batch_size):
            outputs.extend(
                testing_model.generate(
                    prompts[start : start + self.batch_size],
                    prompts_are_rendered=input_mode == "source_rendered",
                    **generation,
                )
            )

        trials = []
        for record, output in zip(records, outputs):
            target = record["payload"]["target"]
            parsed = self._parse_target(output, target)
            invalid = parsed is None
            success = parsed == target
            trials.append(
                {
                    "fingerprint_id": record["fingerprint_id"],
                    "target": target,
                    "parsed_target": parsed,
                    "output": output,
                    "success": int(success),
                    "invalid": int(invalid),
                    "seed": seed,
                }
            )

        total = len(trials)
        hit_rate = sum(item["success"] for item in trials) / total if total else 0.0
        invalid_rate = sum(item["invalid"] for item in trials) / total if total else 0.0
        return FingerprintTestResult(
            score=hit_rate,
            metrics={
                "target_hit_rate": hit_rate,
                "invalid_rate": invalid_rate,
            },
            trials=trials,
            metadata={"input_mode": input_mode, "seed": seed},
        )

    def stealth_texts(self, records):
        texts = []
        for record in records:
            payload = record["payload"]
            texts.extend(
                [
                    {
                        "fingerprint_id": record["fingerprint_id"],
                        "kind": "full_user_prompt",
                        "text": payload["raw_user_prompt"],
                    },
                    {
                        "fingerprint_id": record["fingerprint_id"],
                        "kind": "optimized_suffix",
                        "text": payload["optimized_text"],
                    },
                ]
            )
        return texts

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
