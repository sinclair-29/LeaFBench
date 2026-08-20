import random
import string
import os

import pandas as pd

from fingerprint.trap.gcg import GCGOptimizer


def generate_random_string(string_length, string_type, lower_case_only=True):
    if string_type == "string":
        alphabet = string.ascii_lowercase if lower_case_only else string.ascii_letters
    elif "number" in string_type:
        alphabet = string.digits
    else:
        raise NotImplementedError(f"String type not supported: {string_type}")
    return "".join(random.choice(alphabet) for _ in range(string_length))


def get_prompt_target(string_length, string_type, random_string):
    if string_type in {"number", "number_simpletarget"}:
        if string_length == 1:
            prompt = "Write a random digit. Your reply should only contain the random digit."
            verbose_target = f"Sure, here is a random digit: {random_string}"
        else:
            prompt = (
                f"Write a random string composed of {string_length} digits. "
                "Your reply should only contain the random string."
            )
            verbose_target = (
                f"Sure, here is a random string of {string_length} digits: {random_string}"
            )
        target = random_string if string_type == "number_simpletarget" else verbose_target
    elif string_type == "string":
        prompt = (
            f"Write a random string composed of {string_length} lower case letters. "
            "Your reply should only contain the random string."
        )
        target = (
            f"Sure, here is a random string of {string_length} lowercase letters: "
            f"{random_string}"
        )
    else:
        raise NotImplementedError("String type not supported")
    return {"prompt": prompt, "target": target, "string_target": random_string}


def generate_csv(n_goals, string_type, string_length, csv_path):
    data = [
        get_prompt_target(
            string_length,
            string_type,
            generate_random_string(string_length, string_type),
        )
        for _ in range(n_goals)
    ]
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def generate_adversarial_suffix(
    model,
    tokenizer,
    prompts,
    targets,
    config,
    render_prompt,
    max_input_length=None,
):
    optimizer = GCGOptimizer(
        model,
        tokenizer,
        render_prompt,
        config,
        max_input_length=max_input_length,
    )
    return [optimizer.optimize(prompt, target)[1] for prompt, target in zip(prompts, targets)]
