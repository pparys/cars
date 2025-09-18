import os
import json
import time
import gc
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

import lib_mcmc
import utils

def all_sample_styles():
    return ["prefix", "priority", "restart"]

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
        model: lib_mcmc.ConstrainedModel, 
        prompt: str, 
        propose_style: str,
        log_dir: str, 
    ):
        self.model = model
        prompt = model._format_prompt(prompt)
        self.prompt_ids = model.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.model.device)
        # assert propose_style in ["prefix", "priority", "restart"]
        assert is_valid_propose_style(propose_style)
        self.propose_style = propose_style
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def get_sample(self, n_steps: int, max_new_tokens: int):
        # hopefully this works
        gc.collect()
        torch.cuda.empty_cache()

        current_ids, current_scores = self.model._generate(
            self.prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            constrain=True,
            prefix_ids=None,
        )
        current_cons_logprob = self.model._get_seq_logprob_from_scores(current_scores, current_ids).item()
        current_raw_logprob = self.model._get_seq_logprob(self.prompt_ids, current_ids, constrain=False).item()
        print(f"Initial: {[self.model.tokenizer.decode(token_id) for token_id in current_ids[0]]}")

        steps = []
        sample_file = f"{self.log_dir}/{utils.timestamp(millis=True)}-n{n_steps}.json"
        sample_file_tmp = f"{sample_file}.tmp"

        for i in range(n_steps):
            step_propose_style = self.propose_style
            if step_propose_style.startswith("mix"):
                _, p = step_propose_style.split("-")
                p = float(p)
                step_propose_style = "restart" if np.random.rand() < p else "priority"
            print(f"Step {i} ({step_propose_style})")

            print(f"Current: {[self.model.tokenizer.decode(token_id) for token_id in current_ids[0]]}")
            print(f"Current raw logprob: {current_raw_logprob}")
            proposal_ids, proposal_scores, _ = self.model._propose_next_sequence(
                prompt_ids=self.prompt_ids,
                current_ids=current_ids,
                max_new_tokens=max_new_tokens,
                constrain=True,
                current_scores=current_scores,
                propose_style=step_propose_style,
            )
            proposal_raw_logprob = self.model._get_seq_logprob(self.prompt_ids, proposal_ids, constrain=False).item()
            proposal_cons_logprob = self.model._get_seq_logprob_from_scores(proposal_scores, proposal_ids).item()
            print(f"Proposal: {[self.model.tokenizer.decode(token_id) for token_id in proposal_ids[0]]}")
            print(f"Proposal raw logprob: {proposal_raw_logprob}")

            acceptance_prob = None
            if torch.equal(current_ids, proposal_ids):
                acceptance_prob = 1
            else:
                prop_logprob_cur_to_next = self.model._propose_next_sequence_logprob(
                    current_ids=current_ids,
                    current_scores=current_scores,
                    next_ids=proposal_ids,
                    next_scores=proposal_scores,
                    propose_style=step_propose_style,
                )

                prop_logprob_next_to_cur = self.model._propose_next_sequence_logprob(
                    current_ids=proposal_ids,
                    current_scores=proposal_scores,
                    next_ids=current_ids,
                    next_scores=current_scores,
                    propose_style=step_propose_style,
                )

                log_acc_ratio = proposal_raw_logprob + prop_logprob_next_to_cur - \
                    current_raw_logprob - prop_logprob_cur_to_next

                acceptance_prob = min(1, np.exp(log_acc_ratio))
            print(f"Acceptance prob: {acceptance_prob}")
    
            accepted = bool(np.random.rand() < acceptance_prob)

            # save to steps
            step = {
                "current": {
                    "tokens": [self.model.tokenizer.decode(token_id) for token_id in current_ids[0]],
                    "token_ids": [int(id) for id in current_ids[0]],
                    "raw_logprob": current_raw_logprob,
                    "cons_logprob": current_cons_logprob,
                },
                "proposal": {
                    "tokens": [self.model.tokenizer.decode(token_id) for token_id in proposal_ids[0]],
                    "token_ids": [int(id) for id in proposal_ids[0]],
                    "raw_logprob": proposal_raw_logprob,
                    "cons_logprob": proposal_cons_logprob,
                },
                "acceptance_prob": acceptance_prob,
                "accepted": accepted,
            }
            steps.append(step)
            steps_dump = {"steps": steps}
            with open(sample_file_tmp, "w") as f:
                json.dump(steps_dump, f, indent=4)

            if accepted:
            # if np.random.rand() < acceptance_prob:
                current_ids = proposal_ids
                current_scores = proposal_scores
                current_cons_logprob = proposal_cons_logprob
                current_raw_logprob = proposal_raw_logprob
                print(f"Accepted")
            
            print("\n\n")
            
        os.rename(sample_file_tmp, sample_file)
        return current_ids

    def get_samples(self, n_samples: int, n_steps: int, max_new_tokens: int):
        for i in tqdm(range(n_samples)):
            print(f"Sample {i}")
            sample_start_time = time.time()
            sample = self.get_sample(n_steps, max_new_tokens)
            sample_end_time = time.time()
            sample_time = sample_end_time - sample_start_time
            print(f"Sample time: {sample_time:.2f} s")
            sample_str = self.model.tokenizer.decode(sample[0])
            print(f"Sample: {sample_str}", flush=True)

    
def run_mcmc():
    # from collections import Counter

    # # prompt_file, grammar_file, log_dir = "prompts/bin_rand.md", "grammars/set_bin3_skew.ebnf", "mcmc_runs/bin3_skew-llama3-8b"
    # # prompt_file, grammar_file, log_dir = "prompts/bin_rand.md", "grammars/set_bin3_skew.ebnf", "mcmc_runs/bin3_skew-mistral-7b"

    # # prompt_file, grammar_file, log_dir = "prompts/bin_rand.md", "grammars/set_bin3_skew.ebnf", None
    # # prompt_file, grammar_file, log_dir = "prompts/int_list_50.md", "grammars/int_list.ebnf", "mcmc_runs/int_list_50"

    # # prompt_file, grammar_file, log_dir = "prompts/name-combine-4_short.md", "grammars/name-combine-4_short.ebnf", "mcmc_runs/name-combine-4_short-prefix"
    # # prompt_file, grammar_file, log_dir = "prompts/name-combine-4_short-short.md", "grammars/name-combine-4_short.ebnf", "mcmc_runs/name-combine-4_short-prefix"

    # # prompt_file, grammar_file, log_dir = "prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mcmc_runs/qm_max3-priority"
    # # prompt_file, grammar_file, log_dir = "prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mcmc_runs/qm_max3-prefix"
    # prompt_file, grammar_file, log_dir = "prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mcmc_runs/qm_max3-prefix-DUMMY"

    # model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # model_id = "meta-llama/Llama-3.2-1B-Instruct"
    # model_id = "google/gemma-2-2b-it"

    # model = lib_mcmc.ConstrainedModel(model_id, None)
    model = lib_mcmc.ConstrainedModel(model_id, None, torch_dtype=torch.bfloat16)

    root_log_dir = "mcmc_runs"

    # runs_params = [
    #     # ("prompts/name-combine-2_short.txt", "grammars/name-combine-2_short.ebnf", "priority", "name-combine-2_short"),
    #     # ("prompts/name-combine-2_short.txt", "grammars/name-combine-2_short.ebnf", "prefix", "name-combine-2_short"),
    #     # ("prompts/name-combine-2_short.txt", "grammars/name-combine-2_short.ebnf", "restart", "name-combine-2_short"),
    # ]

    runs_params = [
        ("prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "restart", "qm_max3"),

        # ("prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mix-0.5", "qm_max3"), # 1/2 restart, 1/2 priority
        # ("prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mix-0.75", "qm_max3"), # 3/4 restart, 1/4 priority
        # ("prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mix-0.9", "qm_max3"), # 9/10 restart, 1/10 priority
        # ("prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mix-0.66", "qm_max3"), # 2/3 restart, 1/3 priority
        # ("prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mix-0.8", "qm_max3"), # 4/5 restart, 1/5 priority

        # ("prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mix-0.25", "qm_max3"), # 1/4 restart, 3/4 priority
        # ("prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mix-0.1", "qm_max3"), # 1/10 restart, 9/10 priority
        # ("prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mix-0.33", "qm_max3"), # 1/3 restart, 2/3 priority
        # ("prompts/qm_max3.txt", "grammars/qm_max3.ebnf", "mix-0.2", "qm_max3"), # 1/5 restart, 4/5 priority
    ]

    # runs_params = [
    #     ("prompts/name-combine-4_short.txt", "grammars/name-combine-4_short.ebnf", "restart", "name-combine-4_short"),
    #     ("prompts/name-combine-4_short.txt", "grammars/name-combine-4_short.ebnf", "priority", "name-combine-4_short"),
    #     ("prompts/name-combine-4_short.txt", "grammars/name-combine-4_short.ebnf", "prefix", "name-combine-4_short"),
    # ]

    # runs_params = [
    #     ("prompts/dr-name.txt", "grammars/dr-name.ebnf", "restart", "dr-name"),
    #     ("prompts/dr-name.txt", "grammars/dr-name.ebnf", "priority", "dr-name"),
    #     ("prompts/dr-name.txt", "grammars/dr-name.ebnf", "prefix", "dr-name"),
    # ]

    # runs_params = [
    #     ("prompts/cp_re_ptb_1434.txt", "grammars/cp_re_ptb_1434.ebnf", "restart", "cp_re_ptb_1434"),
    #     ("prompts/cp_re_ptb_1434.txt", "grammars/cp_re_ptb_1434.ebnf", "priority", "cp_re_ptb_1434"),
    #     ("prompts/cp_re_ptb_1434.txt", "grammars/cp_re_ptb_1434.ebnf", "prefix", "cp_re_ptb_1434"),
    # ]

    # runs_params = [
    #     ("prompts/bin_rand_short.md", "grammars/bin_skew.ebnf", "restart", "bin_skew"),
    #     ("prompts/bin_rand_short.md", "grammars/bin_skew.ebnf", "priority", "bin_skew"),
    #     ("prompts/bin_rand_short.md", "grammars/bin_skew.ebnf", "prefix", "bin_skew"),
    # ]

    n_samples = 500
    n_steps = 22
    max_new_tokens = 128

    for prompt_file, grammar_file, propose_style, name_prefix in runs_params:
        prompt = open(prompt_file).read()
        grammar_str = open(grammar_file).read()

        model._set_grammar_constraint(grammar_str)
        mcmc = MCMC(model, prompt, propose_style, name_prefix, root_log_dir)
        mcmc.get_samples(n_samples, n_steps, max_new_tokens)
            
if __name__ == "__main__":
    run_mcmc()