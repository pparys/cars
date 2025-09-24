import sys
from distr_utils import *

def print_kl_divergence_dir(task : str, dir : str):
    steps_total = 10
    #mcmc_runs = [load_mcmc_run_data(run_dir, min_steps=steps_total) for run_dir in mcmc_run_dirs]

    # flatten mcmc_runs
    #all_samples = [sample for run in mcmc_runs for sample in run]
    #true_distr_est = estimate_full_distribution(all_samples, "raw_logprob")
    #print(len(true_distr_est))

    runs_kls = []
    steps_range = list(range(1, steps_total+1))
    
    for run in [1]:
        run_kls = []
        run_lower_cis = []
        run_upper_cis = []
        n_samples_in_run = len(run)
        
        for n_steps in steps_range:
            # Get samples at this step
            samples = [tuple(step["steps"][n_steps-1]["current"]["token_ids"]) 
                       for step in run]
            
            # Compute KL divergence with bootstrapping
            # mean_kl, lower_ci, upper_ci = bootstrap_kl(samples, true_distr_est, n_bootstrap=n_samples_in_run)
            mean_kl, lower_ci, upper_ci = bootstrap_kl(samples, true_distr_est, n_bootstrap=500)
            
            run_kls.append(mean_kl)
            run_lower_cis.append(lower_ci)
            run_upper_cis.append(upper_ci)
        
        runs_kls.append((run_kls, run_lower_cis, run_upper_cis))

    # Create a figure and axis
    plt.figure(figsize=(6, 6))

    # Get a colormap with distinct colors
    cmap = get_cmap('tab10')



def print_kl_divergence_all(main_style : str, model : str):
    print(f"KL-divergence for {main_style}, model {model}")
    for task, dir in get_all_task_dirs():
        if dir.endswith(f"-{model}"):
            print_kl_divergence_dir(task, dir)
#            klmean, kllow, klhigh, kl, count = bootstrap_kl_divergence_from_dir(main_style, dir)
#            if count>0:
#                print(f"{task} --> {kllow:.5f}-{klmean:.5f}-{klhigh:.5f} {kl:.5f}", f" ({count} samples)" if count<300 else "")


if __name__ == "__main__":
    print_kl_divergence_all(sys.argv[1] if len(sys.argv)>=2 else "1")
