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
    all_dirs = [subdir for _, subdir in get_all_style_dirs(dir)]
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

    orig_distr = {}
    for x, v in all_data:
        if x in orig_distr:
            if not math.isclose(v, orig_distr[x]):
                print(x, v, orig_distr[x])
            assert math.isclose(v, orig_distr[x])
        else:
            orig_distr[x] = v
    #pprint.pprint(orig_distr)
    #pprint.pprint(new_distr)

    # KL-divergence:
    keys = list(orig_distr.keys())
    log_probs = np.array([orig_distr[k] for k in keys])
    orig_probs = softmax(log_probs)
    new_probs = np.array([new_distr[k]/total for k in keys])
    print("KL:", np.sum(kl_div(new_probs, orig_probs)))
    
    # For chi2 - all keys:
    f_exp = orig_probs*total
    f_obs = np.array([new_distr[k] for k in keys])
    for i in range(0, len(f_obs), 20):
        linia = f_obs[i:i+20]
        print(' '.join(f"{liczba:5.0f}   " for liczba in linia))
    for i in range(0, len(f_exp), 20):
        linia = f_exp[i:i+20]
        print(' '.join(f"{liczba:8.2f}" for liczba in linia))
    chi2_stat, p_value = chisquare(f_obs=f_obs, f_exp=f_exp)
    print("Chi2:", chi2_stat)
    #print("p-value:", p_value)

    #daaffafaf





def plot_success_rates(all_mcmc_run_data: list[tuple[list[str], str]], split: str, output_dir: str, cut = 1000):

    print(f"Drawing success rates for length {cut}")
    # Create a figure with equal aspect ratio (square)
    plt.figure(figsize=(12, 6))
    # Get a colormap with distinct colors
    cmap = get_cmap('tab20')

    j = 0
    llast=[]
    for mcmc_run_dirs, task_id in all_mcmc_run_data:
        mcmc_runs = [load_mcmc_run_data(run_dir, min_steps=cut) for run_dir in mcmc_run_dirs]
        assert len(mcmc_runs)==1
        mcmc_runs = mcmc_runs[0]
        if len(mcmc_runs)==0:
            continue
        assert len(mcmc_runs)==1
        mcmc_runs = mcmc_runs[0]
        successes = mcmc_runs['successes'][:cut]
        rate = []
        ok = 0
        for i in range(len(successes)):
            if successes[i]:
                ok = ok+1
            rate.append(ok/(i+1))
        llast.append(mcmc_runs['steps'][-1])

        x = range(1, len(rate)+1)
        y = rate
        plt.plot(x, y, marker='o', linestyle='-', color=cmap(j), label=task_id)
        j = j+1

    plt.title("Success rate")
    plt.xlabel("number of samples")
    plt.ylabel("success rate")

    plt.grid(True)
    plt.legend(loc='lower right')

    # Wyświetlenie wykresu
    #plt.show()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{split}-sr-{cut}.png"), dpi=200)

    print("Improvements on probabilities (in the last sample):")
    for last in llast:
        print(f"{np.exp(last['raw_logprob']):.10f} -> {np.exp(last['cons_logprob']):.10f}")

'''
def plot_kl_scatter(all_mcmc_run_data: list[tuple[list[str], str]], split: str, output_dir: str):
    # for each run, plot KL divergence as a scatter
    # x axis is kl at step 0, y axis is kl at step 10
    # each run is a different color
    steps_total = 10

    task_kls_at_0 = {
        "prefix": [],
        "priority": [],
        "restart": []
    }
    task_kls_at_10 = {
        "prefix": [],
        "priority": [],
        "restart": []
    }

    for mcmc_run_dirs, task_id in all_mcmc_run_data:

        mcmc_runs = [load_mcmc_run_data(run_dir, min_steps=steps_total) for run_dir in mcmc_run_dirs]

        # flatten mcmc_runs
        all_samples = [sample for run in mcmc_runs for sample in run]
        true_distr_est = estimate_full_distribution(all_samples, "raw_logprob")
        print(len(true_distr_est))
        # run_kls = {}
        for r, run in enumerate(mcmc_runs):
            # Extract just the directory name for cleaner labels
            label = os.path.basename(mcmc_run_dirs[r]).rsplit("-", 1)[-1]
            if label == "asap":
                continue

            samples_step_0 = [tuple(step["steps"][0]["current"]["token_ids"]) for step in run]
            samples_step_10 = [tuple(step["steps"][steps_total - 1]["current"]["token_ids"]) for step in run]
            mean_kl_step_0, _, _ = bootstrap_kl(samples_step_0, true_distr_est, n_bootstrap=500)
            mean_kl_step_10, _, _ = bootstrap_kl(samples_step_10, true_distr_est, n_bootstrap=500)
            # run_kls[label] = (mean_kl_step_0, mean_kl_step_10)
            task_kls_at_0[label].append(mean_kl_step_0)
            task_kls_at_10[label].append(mean_kl_step_10)

            # Convert to numpy arrays for scatter plot
            kl_step_0 = np.array([mean_kl_step_0])
            kl_step_10 = np.array([mean_kl_step_10])
            # Remove NaN values
            kl_step_0 = kl_step_0[~np.isnan(kl_step_0)]
            kl_step_10 = kl_step_10[~np.isnan(kl_step_10)]
            # Scatter plot: KL divergence at step 1 vs KL divergence at step 10

    # Create a figure with equal aspect ratio (square)
    plt.figure(figsize=(6, 6))
    # Get a colormap with distinct colors
    cmap = get_cmap('tab10')

    # Find min and max values for both axes to set equal scale
    min_val = min(min(task_kls_at_0["restart"]), min(task_kls_at_10["prefix"]), min(task_kls_at_10["priority"]), min(task_kls_at_10["restart"]))
    max_val = max(max(task_kls_at_0["restart"]), max(task_kls_at_10["prefix"]), max(task_kls_at_10["priority"]), max(task_kls_at_10["restart"]))

    # Add some padding
    padding = (max_val - min_val) * 0.05
    plot_range = [min_val - padding, max_val + padding]

    # Plot the scatter points
    plt.scatter(task_kls_at_0["restart"], task_kls_at_10["prefix"], s=100, color=cmap(0), edgecolors="none", label="uniform", alpha=0.7)
    plt.scatter(task_kls_at_0["restart"], task_kls_at_10["priority"], s=100, color=cmap(1), edgecolors="none", label="priority", alpha=0.7)
    plt.scatter(task_kls_at_0["restart"], task_kls_at_10["restart"], s=100, color=cmap(2), edgecolors="none", label="restart", alpha=0.7)

    # Add diagonal line (y=x)
    plt.plot(plot_range, plot_range, 'k:', alpha=0.7)

    # Set the same limits for both axes
    plt.xlim(plot_range)
    plt.ylim(plot_range)

    # Add decorations
    plt.xlabel('GCD KL Divergence', fontsize=18)
    plt.ylabel(f'MCMC(k=10) KL Divergence', fontsize=18)
    plt.title(f'{split}', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='best', framealpha=0.7)
    plt.tight_layout()
    plt.gca().set_aspect('equal')  # Ensure perfect square aspect ratio

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{split}-kl-gcd_scatter.png"), dpi=200)

    print_kl_stats(task_kls_at_0, task_kls_at_10)

def plot_kl_scatter_asap(all_mcmc_run_data: list[tuple[list[str], str]], split: str, output_dir: str):
    # for each run, plot KL divergence as a scatter
    # x axis is kl at step 0, y axis is kl at step 10
    # each run is a different color
    steps_total = 10

    task_kls_at_0 = {
        "prefix": [],
        "priority": [],
        "restart": [],
        "asap": [],
    }
    task_kls_at_10 = {
        "prefix": [],
        "priority": [],
        "restart": [],
        "asap": [],
    }

    for mcmc_run_dirs, task_id in all_mcmc_run_data:

        mcmc_runs = [load_mcmc_run_data(run_dir, min_steps=steps_total) for run_dir in mcmc_run_dirs]

        # flatten mcmc_runs
        all_samples = [sample for run in mcmc_runs for sample in run]
        true_distr_est = estimate_full_distribution(all_samples, "raw_logprob")
        print(len(true_distr_est))
        # run_kls = {}
        for r, run in enumerate(mcmc_runs):
            # Extract just the directory name for cleaner labels
            label = os.path.basename(mcmc_run_dirs[r]).rsplit("-", 1)[-1]

            samples_step_0 = [tuple(step["steps"][0]["current"]["token_ids"]) for step in run]
            samples_step_10 = [tuple(step["steps"][steps_total - 1]["current"]["token_ids"]) for step in run]
            mean_kl_step_0, _, _ = bootstrap_kl(samples_step_0, true_distr_est, n_bootstrap=500)
            mean_kl_step_10, _, _ = bootstrap_kl(samples_step_10, true_distr_est, n_bootstrap=500)
            # run_kls[label] = (mean_kl_step_0, mean_kl_step_10)
            task_kls_at_10[label].append(mean_kl_step_10)
            task_kls_at_0[label].append(mean_kl_step_0)

            # Convert to numpy arrays for scatter plot
            kl_step_0 = np.array([mean_kl_step_0])
            kl_step_10 = np.array([mean_kl_step_10])
            # Remove NaN values
            kl_step_0 = kl_step_0[~np.isnan(kl_step_0)]
            kl_step_10 = kl_step_10[~np.isnan(kl_step_10)]
            # Scatter plot: KL divergence at step 1 vs KL divergence at step 10

    # Create a figure with equal aspect ratio (square)
    plt.figure(figsize=(6, 6))
    # Get a colormap with distinct colors
    cmap = get_cmap('tab10')

    # Find min and max values for both axes to set equal scale
    # min_val = min(min(task_kls_at_0["prefix"]), min(task_kls_at_10["prefix"]))
    min_val = min(min(task_kls_at_10["prefix"]), min(task_kls_at_10["priority"]), min(task_kls_at_10["restart"]), min(task_kls_at_10["asap"]))
    # max_val = max(max(task_kls_at_0["prefix"]), max(task_kls_at_10["prefix"]))
    max_val = max(max(task_kls_at_10["prefix"]), max(task_kls_at_10["priority"]), max(task_kls_at_10["restart"]), max(task_kls_at_10["asap"]))

    # Add some padding
    padding = (max_val - min_val) * 0.05
    plot_range = [min_val - padding, max_val + padding]

    # Plot the scatter points
    plt.scatter(task_kls_at_10["asap"], task_kls_at_10["prefix"], s=100, color=cmap(0), edgecolors='none', label="uniform", alpha=0.7)
    plt.scatter(task_kls_at_10["asap"], task_kls_at_10["priority"], s=100, color=cmap(1), edgecolors='none', label="priority", alpha=0.7)
    plt.scatter(task_kls_at_10["asap"], task_kls_at_10["restart"], s=100, color=cmap(2), edgecolors='none', label="restart", alpha=0.7)

    # Add diagonal line (y=x)
    plt.plot(plot_range, plot_range, 'k:', alpha=0.7)

    # Set the same limits for both axes
    plt.xlim(plot_range)
    plt.ylim(plot_range)

    # Add decorations
    plt.xlabel('ASAP(k=10) KL Divergence', fontsize=18)
    plt.ylabel(f'MCMC(k=10) KL Divergence', fontsize=18)
    # plt.title(f'ASAP@10 vs MCMC@10 KL Divergence, {split}', fontsize=20)
    plt.title(f'{split}', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='best', framealpha=0.7)
    plt.tight_layout()
    plt.gca().set_aspect('equal')  # Ensure perfect square aspect ratio

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{split}-kl-asap_scatter.png"), dpi=200)

    # print_kl_stats(task_kls_at_0, task_kls_at_10)
    print_asap_kl_stats(task_kls_at_10)


def plot_sampled_mass(mcmc_run_dirs: list[str]):
    steps_total = 20
    mcmc_runs = [load_mcmc_run_data(run_dir, min_steps=steps_total) for run_dir in mcmc_run_dirs]

    runs_mass_stats = []
    steps_range = list(range(1, steps_total+1))
    for run in mcmc_runs:
        # print(f"Processing run with {len(run)} samples")
        # sampled_mass = []
        avg_logprobs = []
        avg_logprobs_wor = []
        for n_steps in steps_range:
            logprobs = [
                (tuple(step["steps"][n_steps-1]["current"]["token_ids"]),
                 step["steps"][n_steps-1]["current"]["raw_logprob"])
                for step in run
            ]
            avg_logprob = np.mean([logprob for _, logprob in logprobs])

            samples_wor = {}
            for seq, logprob in logprobs:
                if seq not in samples_wor:
                    samples_wor[seq] = logprob
                else:
                    assert samples_wor[seq] == logprob, f"Logprob mismatch for sample {seq}: {samples_wor[seq]} vs {logprob}"

            avg_logprob_wor = np.mean(list(samples_wor.values()))

            avg_logprobs.append(avg_logprob)
            avg_logprobs_wor.append(avg_logprob_wor)
        runs_mass_stats.append((avg_logprobs, avg_logprobs_wor))

    # Create a figure and axis
    plt.figure(figsize=(10, 6))

    # Get a colormap with distinct colors
    cmap = get_cmap('tab10')

    # Plot each run with a different color
    for i, ((avg_logprobs, avg_logprobs_wor), run_dir) in enumerate(zip(runs_mass_stats, mcmc_run_dirs)):
        # Extract just the directory name for cleaner labels
        label = os.path.basename(run_dir)

        # Plot average logprobs with solid line
        plt.plot(steps_range, avg_logprobs, marker='o', linestyle='-', linewidth=2,
                 color=cmap(i), label=f"{label} (with rep.)")

        # Plot average logprobs without repetition with dashed line
        plt.plot(steps_range, avg_logprobs_wor, marker='o', linestyle='--', linewidth=2,
                 color=cmap(i), label=f"{label} (w/o rep.)")

    # Add decorations
    plt.xlabel('Number of MCMC Steps', fontsize=14)
    plt.ylabel('Average Log Probability', fontsize=14)
    plt.title('Average Log Probability vs. Steps', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best', framealpha=0.7)
    plt.tight_layout()

    plt.show()

def get_mcmc_samples(mcmc_runs: list[dict], n_steps: int) -> list[tuple]:
    samples = []
    for mcmc_run in mcmc_runs:
        sample_at_step = tuple(mcmc_run["steps"][n_steps-1]["current"]["token_ids"])
        logprob_at_step = mcmc_run["steps"][n_steps-1]["current"]["raw_logprob"]
        samples.append((sample_at_step, logprob_at_step))
    return samples
'''