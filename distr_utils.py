import json
import os
import pprint

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.special import softmax, kl_div
from scipy.stats import chisquare
from collections import defaultdict
import math


def get_all_task_dirs():
    data_dir = "runs_log"
    return [(task_dir[:task_dir.rfind('-')], f"{data_dir}/{task_dir}") for task_dir in sorted(os.listdir(data_dir))]


def get_all_style_dirs(task_dir : str):
    return [(style_dir[:style_dir.find('-')], f"{task_dir}/{style_dir}") for style_dir in sorted(os.listdir(task_dir))]


def load_runs_log_from_dir(dir: str) -> list[dict]:
    runs = []
    for run in os.listdir(dir):
        if run.endswith(".json"):
            with open(os.path.join(dir, run), "r") as f:
                run_data = json.load(f)
                runs.append(run_data)
    #print(f"Loaded {len(runs)} samples from {dir}")
    return runs


def get_success_rates(dir : str):
    res = []
    for data in load_runs_log_from_dir(dir):
        assert len(data["successes"]) == 1000
        res.append(data["successes"].count(True))
    return res


def extract_samples(samples):
    result = []
    for data in samples:
        for d in data:
            for s in d["steps"]:
                if "token_ids" in s:
                    result.append((tuple(s["token_ids"]), s["raw_logprob"]))
                else:
                    result.append((tuple(s["current"]["token_ids"]), s["current"]["raw_logprob"]))
                    result.append((tuple(s["proposal"]["token_ids"]), s["proposal"]["raw_logprob"]))
    return result
    

def get_kl_divergence(main_style : str, dir : str):
    all_dirs = [subdir for style, subdir in get_all_style_dirs(dir)]
    kl_dirs = [subdir for style, subdir in get_all_style_dirs(dir) if style == main_style]
    all_data = [load_runs_log_from_dir(subdir) for subdir in all_dirs]
    kl_data = [load_runs_log_from_dir(subdir) for subdir in kl_dirs]
    all_data = extract_samples(all_data)
    kl_data = extract_samples(kl_data)
    
    new_distr = defaultdict(int)
    total = 0
    for x, _ in kl_data:
        new_distr[x] += 1
        total += 1
    if total==0:
        return None

    orig_distr = {}
    for x, v in all_data:
        if x in orig_distr:
            if not math.isclose(v, orig_distr[x]):
                print(x, v, orig_distr[x])
            assert math.isclose(v, orig_distr[x])
        else:
            orig_distr[x] = v

    # KL-divergence:
    keys = list(orig_distr.keys())
    log_probs = np.array([orig_distr[k] for k in keys])
    orig_probs = softmax(log_probs)
    new_probs = np.array([new_distr[k]/total for k in keys])
    kl = np.sum(kl_div(new_probs, orig_probs))
    return kl
    
    # For chi2 - all keys:
    #f_exp = orig_probs*total
    #f_obs = np.array([new_distr[k] for k in keys])
    #for i in range(0, len(f_obs), 20):
    #    linia = f_obs[i:i+20]
    #    print(' '.join(f"{liczba:5.0f}   " for liczba in linia))
    #for i in range(0, len(f_exp), 20):
    #    linia = f_exp[i:i+20]
    #    print(' '.join(f"{liczba:8.2f}" for liczba in linia))
    #chi2_stat, p_value = chisquare(f_obs=f_obs, f_exp=f_exp)
    #print("Chi2:", chi2_stat)
    #print("p-value:", p_value)


def plot_success_rates(big_task : str, tasks : list[tuple[str, str]], style : str, output_dir : str, cut : int = 1000):

    all_runs_data = []
    for task, task_dir in tasks:
        for s, subdir in get_all_style_dirs(task_dir):
            if s==style:
                for data in load_runs_log_from_dir(subdir):
                    assert len(data["successes"]) == 1000
                    all_runs_data.append((task, data["successes"][:cut]))

    plt.figure(figsize=(12, 6))
    cmap = get_cmap('tab20')
    j = 0

    #llast=[]
    for task, successes in all_runs_data:
        rate = []
        ok = 0
        for i in range(len(successes)):
            if successes[i]:
                ok = ok+1
            rate.append(ok/(i+1))
        #llast.append(mcmc_runs['steps'][-1])

        x = range(1, len(rate)+1)
        y = rate
        plt.plot(x, y, marker='o', linestyle='-', color=cmap(j), label=task)
        j = j+1

    plt.title("Success rate, " + big_task)
    plt.xlabel("number of samples")
    plt.ylabel("success rate")

    plt.grid(True)
    plt.legend(loc='lower right')

    # Display plot
    #plt.show()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{big_task}-{style}-{cut}.png"), dpi=200)
    plt.close()

    #print("Improvements on probabilities (in the last sample):")
    #for last in llast:
    #    print(f"{np.exp(last['raw_logprob']):.10f} -> {np.exp(last['cons_logprob']):.10f}")

    #############################
    plt.figure(figsize=(12, 6))
    cmap = get_cmap('tab20')
    j = 0

    #llast=[]
    for task, successes in all_runs_data:
        x = []
        y = []
        ok = 0
        for i in range(len(successes)):
            if successes[i]:
                ok = ok+1
                x.append(ok)
                y.append(i+1)

        plt.plot(y, x, marker='o', linestyle='-', color=cmap(j), label=task)
        j = j+1

    plt.title("Samples needed, " + big_task)
    plt.ylabel("correct samples")
    plt.xlabel("tried samples")

    plt.grid(True)
    plt.legend(loc='lower right')

    # Display plot
    #plt.show()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"v-{big_task}-{style}-{cut}.png"), dpi=200)
    plt.close()

    #print("Improvements on probabilities (in the last sample):")
    #for last in llast:
    #    print(f"{np.exp(last['raw_logprob']):.10f} -> {np.exp(last['cons_logprob']):.10f}")

