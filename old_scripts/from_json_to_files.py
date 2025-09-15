import json
import os

import torch
from tqdm import tqdm

import utils

def load_gad_tasks(split):
    assert split in ["SLIA", "CP", "BV4"] 
    slia_tasks_path = f"datasets/GAD-dataset/{split}.jsonl"
    slia_tasks = []
    with open(slia_tasks_path, "r") as f:
        for line in f:
            task = json.loads(line)
            slia_tasks.append(task)
    return slia_tasks


def run_mcmc_gad_tasks(split):

    root_log_dir = "gad_dataset_runs"

    split_log_dir = f"{root_log_dir}/{utils.timestamp()}-{split}"
    # make sure the directory exists
    os.makedirs(split_log_dir, exist_ok=True)

    # Load the GAD tasks
    slia_tasks = load_gad_tasks(split)
    print(f"Loaded {len(slia_tasks)} tasks")
    print([task["id"] for task in slia_tasks])

    for task in slia_tasks:
        task_id = task["id"]
        dir = f"datasets/{split}"
        os.makedirs(dir, exist_ok=True)
        with open(f"{dir}/{task_id}.ebnf", "w", encoding="utf-8") as f:
            f.write(task["grammar"])
        with open(f"{dir}/{task_id}.txt", "w", encoding="utf-8") as f:
            f.write(task["prompt"])



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--split", required=True, choices=["SLIA", "CP", "BV4"])

    args = parser.parse_args()

    run_mcmc_gad_tasks(args.split)
