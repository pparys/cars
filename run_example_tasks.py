import json
import os

import torch
from tqdm import tqdm

import utils
import lib
import mcmc

def load_gad_tasks(split, subset):
    assert split in ["SLIA", "CP", "BV4"] 
    slia_tasks_path = f"datasets/GAD-dataset/{split}.jsonl"
    slia_tasks = []
    with open(slia_tasks_path, "r") as f:
        for line in f:
            task = json.loads(line)
            slia_tasks.append(task)
    if subset is not None:
        slia_tasks = [slia_tasks[i] for i in subset]
    return slia_tasks


def run_mcmc_gad_tasks(styles):
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    if not torch.cuda.is_available():
        model_id = "hsultanbey/codegen350multi_finetuned"

    model = lib.ConstrainedModel(model_id, None, torch_dtype=torch.float32)

    root_log_dir = "gad_dataset_runs"

    split_log_dir = f"{root_log_dir}/{utils.timestamp()}-example"
    # make sure the directory exists
    os.makedirs(split_log_dir, exist_ok=True)

    n_samples = 1 #100
    n_steps = 1000 # 11
    max_new_tokens = 5

    task_id = "example"
    task_prompt = "My favorite car is"#"Generate a binary string of length 5"
    with open("examples/test/binary_len_5_new.ebnf", "r") as f:
        task_grammar = f.read()

    model._set_grammar_constraint(task_grammar)
    for sample_style in styles:
        print(f"Task ID: {task_id}, Sample style: {sample_style}")

        mcmc_runner = mcmc.MCMC(
            model=model,
            prompt=task_prompt,
            sample_style=sample_style,
            name_prefix=task_id,
            root_log_dir=split_log_dir,
        )
        mcmc_runner.get_samples(
            n_samples=n_samples,
            n_steps=n_steps,
            max_new_tokens=max_new_tokens,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--styles", default=None)

    args = parser.parse_args()

    run_mcmc_gad_tasks(mcmc.parse_styles_arg(args.styles))
