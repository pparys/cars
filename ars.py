import os
import json
import time
import gc
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

import lib_ars
import utils

def all_sample_styles():
    #return ["ars0", "ars1", "ars2", "ars3", "ars0f", "ars1f", "ars2f", "ars3f"]
    return ["rs", "rsft", "ars", "cars"]

class ARS:
    def __init__(self, model : lib_ars.ConstrainedModel, prompt : str, sample_style : str, log_dir : str):
        self.model = model
        prompt = model._format_prompt(prompt)
        self.prompt_ids = model.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.model.device)
        assert sample_style in all_sample_styles()
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        learn_level = 3 if sample_style=="cars" else (2 if sample_style=="ars" else 0)
        self.model.reset_sampling(learn_level = learn_level, constrain_first = (sample_style=="rsft"))


    def get_sample(self, n_steps : int, max_new_tokens : int):
    
        steps = []
        successes = []
        sample_file = f"{self.log_dir}/{utils.timestamp(millis=True)}-n{n_steps}.json"
        sample_file_tmp = f"{sample_file}.tmp"
        
        for i in range(n_steps):
            sample_start_time = time.time()
            try:
                current_ids, current_scores, current_raw_logprob = self.model._generate(self.prompt_ids, max_new_tokens=max_new_tokens)
                sample_end_time = time.time()
                tokens = [self.model.tokenizer.decode(token_id) for token_id in current_ids[0]]
                token_ids = [int(id) for id in current_ids[0]]
                #current_raw_logprob2 = self.model._get_seq_logprob(self.prompt_ids, current_ids, constrain=False).item()
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
                token_ids = e.args[0]
                tokens = [self.model.tokenizer.decode(token_id) for token_id in token_ids]
                print(f"Sample {i} failed, tokens: {e.args[0]} / {tokens}", end='')
                successes.append(False)

            sample_time = sample_end_time - sample_start_time
            logits_time = self.model.gcd_logits_processor.logits_process_time
            print(f", time: {sample_time:.2f} ({logits_time:.2f})", flush=True)
            print(self.model.tokenizer.decode(token_ids))
            steps_dump = {"steps": steps, "successes": successes}
            with open(sample_file_tmp, "w") as f:
                json.dump(steps_dump, f, indent=4)
            #gc.collect()
            #torch.cuda.empty_cache()
        os.rename(sample_file_tmp, sample_file)
        print(f"Total suceeses: {len(steps)}/{len(successes)}")

    def get_samples(self, n_samples : int, n_steps : int, max_new_tokens : int):
        for i in tqdm(range(n_samples)):
            print(f"Sample {i}")
            sample_start_time = time.time()
            self.get_sample(n_steps, max_new_tokens)
            sample_end_time = time.time()
            sample_time = sample_end_time - sample_start_time
            print(f"Sample time: {sample_time:.2f} s")
            #sample_str = self.model.tokenizer.decode(sample[0])
            #print(f"Sample: {sample_str}")
