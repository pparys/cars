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


def extract_samples(all_data, remove_unfinished = False, only_final = False, mcmc_len = 10):
    result = []
    
    def maybe_append(s):
        if remove_unfinished and (s["tokens"][-1] not in ['<|eot_id|>', '<|im_end|>']):
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
                maybe_append(d["steps"][mcmc_len-1]["current"])
            else:
                for s in d["steps"]:
                    process_step(s)
                
    return result
    

def get_kl_divergence_from_data(keys, my_data, bkgr_probs):
    my_distr = defaultdict(int)
    for x,_,_ in my_data:
        my_distr[x] += 1
    my_probs = np.array([my_distr[k]/len(my_data) for k in keys])
    kl = np.sum(kl_div(my_probs, bkgr_probs))
    #for a in range(len(my_probs)):
    #    print(f"{my_probs[a]:.2f} {bkgr_probs[a]:.4f}")
    return kl


def bootstrap_kl_divergence_from_dir(main_style : str, dir : str, mcmc_len : int = 10):
    bkgr_dirs = [subdir for style, subdir in get_all_style_dirs(dir)]
    my_dirs = [subdir for style, subdir in get_all_style_dirs(dir) if style == main_style]
    bkgr_data = [load_runs_log_from_dir(subdir) for subdir in bkgr_dirs]
    my_data = [load_runs_log_from_dir(subdir) for subdir in my_dirs]
    bkgr_data = extract_samples(bkgr_data, remove_unfinished = True)
    my_data = extract_samples(my_data, only_final = True, mcmc_len = mcmc_len)[:300]

    def isclose(x,y):
        return abs(x-y)<0.6

    bkgr_distr = {}
    for x,_,v in bkgr_data:
        if x in bkgr_distr:
            if not isclose(v, bkgr_distr[x]):
                print(x, v, bkgr_distr[x])
            assert isclose(v, bkgr_distr[x])
        else:
            bkgr_distr[x] = v

    keys = list(bkgr_distr.keys())
    log_probs = np.array([bkgr_distr[k] for k in keys])
    bkgr_probs = softmax(log_probs)
    
    """Perform bootstrap resampling to get confidence intervals for KL divergence."""
    n_samples = len(my_data)
    bootstrap_kls = []
    if n_samples==0:
        return None, None, None, None, 0
    
    n_bootstrap = 500
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        resampled = [my_data[i] for i in resampled_indices]
            
        kl = get_kl_divergence_from_data(keys, resampled, bkgr_probs)
        bootstrap_kls.append(kl)
        
    # Calculate confidence intervals
    lower_ci = np.percentile(bootstrap_kls, 2.5)
    upper_ci = np.percentile(bootstrap_kls, 97.5)
    mean_kl = np.mean(bootstrap_kls)

    return mean_kl, lower_ci, upper_ci, get_kl_divergence_from_data(keys, my_data, bkgr_probs), n_samples


def get_num_unfinished(main_style : str, dir : str):
    subdirs = [subdir for style, subdir in get_all_style_dirs(dir) if style == main_style]
    data = [load_runs_log_from_dir(subdir) for subdir in subdirs]
    data_all = extract_samples(data)
    data_good = extract_samples(data, remove_unfinished = True)

    return len(data_all)-len(data_good), len(data_all)


def swap_pairs(my_list):
    for i in range(0, len(my_list) - 1, 2):
        my_list[i], my_list[i + 1] = my_list[i + 1], my_list[i]
        x,y = my_list[i]
        x = x.replace('_with_', "+")
        my_list[i] = (x+", CARS", y)
        x,y = my_list[i+1]
        x = x.replace('_with_', "+")
        my_list[i+1] = (x+", ARS", y)
    return my_list

def plot_success_rates(big_task : str, tasks : list[tuple[str, str]], style : str, output_dir : str, cut : int = 1000):

    all_runs_data = []
    for task, task_dir in tasks:
        for s, subdir in get_all_style_dirs(task_dir):
            #print(subdir)
            if (big_task=="BV4" and (s==style or s==f"old{style}")) or (big_task=="fuzzing" and (subdir in ["runs_log/fuzzing-json-generate_json-42128c9b-1/cars-2025-09-17_07-16-44", "runs_log/fuzzing-sql-generate_sql-71d4ccd4-1/cars-2025-09-19_23-15-50", "runs_log/fuzzing-sql-generate_sql_with_grammar-88960849-1/cars-2025-09-21_03-38-41", "runs_log/fuzzing-json-generate_json-42128c9b-1/ars-2025-09-22_11-23-38", "runs_log/fuzzing-sql-generate_sql-71d4ccd4-1/ars-2025-09-20_06-05-52", "runs_log/fuzzing-sql-generate_sql_with_grammar-88960849-1/ars-2025-09-20_06-13-36", "runs_log/fuzzing-xml-generate_xml_with_grammar-55b3824b-1/cars-2025-09-17_14-16-50", "runs_log/fuzzing-xml-generate_xml_with_grammar-55b3824b-1/ars-2025-09-20_07-06-53", "runs_log/fuzzing-xml-generate_xml-dfa28a53-1/cars-2025-09-17_12-32-37", "runs_log/fuzzing-xml-generate_xml-dfa28a53-1/ars-2025-09-20_06-26-59"])):
                for data in load_runs_log_from_dir(subdir):
                    if len(data["successes"]) >= 1000:
                        print(s, subdir)
                        all_runs_data.append((task, data["successes"][:cut]))
    if big_task=="fuzzing":
        all_runs_data = swap_pairs(all_runs_data)
        
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
    plt.xlabel("Number of Calls")
    plt.ylabel("Success Rate")

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

    plt.title("Production of samples in time, " + big_task)
    plt.ylabel("Produced Samples")
    plt.xlabel("Number of Calls")

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

