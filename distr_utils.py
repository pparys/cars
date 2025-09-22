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


def load_runs_log_from_dir(dir_or_file: str) -> list[dict]:
    files = []
    if os.path.isdir(dir_or_file):
        for file in os.listdir(dir_or_file):
            if file.endswith(".json"):
                files.append(os.path.join(dir_or_file, file))
    else:
        files.append(dir_or_file)
    runs = []
    for file in files:
        with open(file, "r") as f:
            run_data = json.load(f)
            runs.append(run_data)
    #print(f"Loaded {len(runs)} samples from {dir}")
    return runs


def get_success_rates(dir : str):
    res = []
    for data in load_runs_log_from_dir(dir):
        #assert len(data["successes"]) == 1000
        #res.append(data["successes"].count(True))
        s = 0
        t = 0
        for a in data["successes"]:
            if a:
                s += 1
            t += 1
            if s==100:
                break;
        res.append((s, t))
    return res


def extract_samples(all_data, remove_unfinished = False, only_final = False):
    result = []
    
    def maybe_append(s):
        if remove_unfinished and s["tokens"][-1] != '<|eot_id|>':
            assert len(s['tokens']) >= 512
        else:
            result.append((tuple(s["token_ids"]), s["tokens"], s["raw_logprob"]))
    
    def process_step(s):
        if "token_ids" in s:
            maybe_append(s)
        else:
            maybe_append(s["current"])
            maybe_append(s["proposal"])
    
    for data in all_data:
        for d in data:
            if only_final and len(d["steps"])>0 and "current" in d["steps"][0]:
                maybe_append(d["steps"][9]["current"])
            else:
                for s in d["steps"]:
                    process_step(s)
                
    return result
    

def get_kl_divergence(main_style : str, dir : str):
    bkgr_dirs = [subdir for style, subdir in get_all_style_dirs(dir)]
    my_dirs = [subdir for style, subdir in get_all_style_dirs(dir) if style == main_style]
    bkgr_data = [load_runs_log_from_dir(subdir) for subdir in bkgr_dirs]
    my_data = [load_runs_log_from_dir(subdir) for subdir in my_dirs]
    bkgr_data = extract_samples(bkgr_data, remove_unfinished = True)
    my_data = extract_samples(my_data, only_final = True)[:100]
    
    my_distr = defaultdict(int)
    for x,_,_ in my_data:
        my_distr[x] += 1
    if len(my_data)==0:
        return None, 0

    bkgr_distr = {}
    for x,_,v in bkgr_data:
        if x in bkgr_distr:
            if not math.isclose(v, bkgr_distr[x]):
                print(x, v, bkgr_distr[x])
            assert math.isclose(v, bkgr_distr[x])
        else:
            bkgr_distr[x] = v

    # KL-divergence:
    keys = list(bkgr_distr.keys())
    log_probs = np.array([bkgr_distr[k] for k in keys])
    bkgr_probs = softmax(log_probs)
    my_probs = np.array([my_distr[k]/len(my_data) for k in keys])
    kl = np.sum(kl_div(my_probs, bkgr_probs))
    #for a in range(len(my_probs)):
    #    print(f"{my_probs[a]:.2f} {bkgr_probs[a]:.4f}")
    return kl, len(my_data)


def get_num_unfinished(main_style : str, dir : str):
    subdirs = [subdir for style, subdir in get_all_style_dirs(dir) if style == main_style]
    data = [load_runs_log_from_dir(subdir) for subdir in subdirs]
    data_all = extract_samples(data)
    data_good = extract_samples(data, remove_unfinished = True)

    return len(data_all)-len(data_good), len(data_all)


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

