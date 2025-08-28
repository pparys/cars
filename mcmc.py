import os
import json
import time
import gc
from dataclasses import dataclass
from transformers_gad.oracle.oracle_trie import Trie

import torch
import numpy as np
from tqdm import tqdm

import lib
import utils

def is_valid_propose_style(propose_style):
    if propose_style in ["prefix", "priority", "restart"]:
        return True
    mix, p = propose_style.split("-")
    p = float(p)
    if mix == "mix" and 0 <= p <= 1:
        return True
    return False

class MCMC:
    def __init__(
        self, 
        model: lib.ConstrainedModel, 
        prompt: str, 
        propose_style: str,
        name_prefix: str,
        root_log_dir: str, 
    ):
        self.model = model
        prompt = model._format_prompt(prompt)
        self.prompt_ids = model.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.model.device)
        # assert propose_style in ["prefix", "priority", "restart"]
        # assert is_valid_propose_style(propose_style)
        assert (propose_style == "ars")
        self.propose_style = propose_style
        self.root_log_dir = root_log_dir
        os.makedirs(root_log_dir, exist_ok=True)
        self.log_dir = f"{root_log_dir}/{utils.timestamp()}-{name_prefix}-{propose_style}"
        os.makedirs(self.log_dir, exist_ok=True)

    def get_sample(self, n_steps: int, max_new_tokens: int):
        # hopefully this works

        steps = []
        successes = []
        sample_file = f"{self.log_dir}/{utils.timestamp(millis=True)}-n{n_steps}.json"
        oracle_trie = Trie()
        

        for i in range(n_steps):
            sample_start_time = time.time()
            try:
                current_ids, current_scores = self.model._generate(
                    self.prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    constrain=True,
                    prefix_ids=None,
                    oracle_trie = oracle_trie
                )
                sample_end_time = time.time()
                tokens = [self.model.tokenizer.decode(token_id) for token_id in current_ids[0]]
                token_ids = [int(id) for id in current_ids[0]]
                current_raw_logprob = self.model._get_seq_logprob(self.prompt_ids, current_ids, constrain=False).item()
                current_cons_logprob = self.model._get_seq_logprob_from_scores(current_scores, current_ids).item()
                print(f"Sample {i} success: {token_ids} / {tokens}, raw_logprob: {current_raw_logprob}, cons_logprob: {current_cons_logprob}", end='')
            
                # save to steps
                step = {
                    "tokens": tokens,
                    "token_ids": token_ids,
                    "raw_logprob": current_raw_logprob,
                    "cons_logprob": current_cons_logprob,
                }
                steps.append(step)
                successes.append(True)

            except ValueError as e:
                sample_end_time = time.time()
                tokens = [self.model.tokenizer.decode(token_id) for token_id in e.args[1]]
                print(f"Sample {i} failed, tokens: {e.args[1]} / {tokens}", end='')
                successes.append(False)

            sample_time = sample_end_time - sample_start_time
            logits_time = self.model.gcd_logits_processor.logits_process_time
            print(f", time: {sample_time:.2f} ({logits_time:.2f})", flush=True)
            steps_dump = {"steps": steps, "successes": successes}
            with open(sample_file, "w") as f:
                json.dump(steps_dump, f, indent=4)
            gc.collect()
            torch.cuda.empty_cache()
            
        #return current_ids

    def get_samples(self, n_samples: int, n_steps: int, max_new_tokens: int):
        for i in tqdm(range(n_samples)):
            print(f"Sample {i}")
            sample_start_time = time.time()
            self.get_sample(n_steps, max_new_tokens)
            sample_end_time = time.time()
            sample_time = sample_end_time - sample_start_time
            print(f"Sample time: {sample_time:.2f} s")
            #sample_str = self.model.tokenizer.decode(sample[0])
            #print(f"Sample: {sample_str}")
