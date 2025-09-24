import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from pathlib import Path
from collections import defaultdict
import warnings
import sys

plt.rcParams.update({
	'font.size': 18,
	'axes.labelsize': 20,
	'axes.titlesize': 20,
	'xtick.labelsize': 16,
	'ytick.labelsize': 16,
	'legend.fontsize': 18,
	'figure.titlesize': 24,
	'font.family': 'serif',
	'mathtext.fontset': 'cm',
	'axes.linewidth': 1.5,
	'axes.spines.top': False,
	'axes.spines.right': False,
	'xtick.major.size': 6,
	'ytick.major.size': 6,
	'xtick.major.width': 1.5,
	'ytick.major.width': 1.5,
	'xtick.direction': 'in',
	'ytick.direction': 'in',
	'xtick.minor.visible': False,
	'ytick.minor.visible': False,
	'legend.frameon': True,
	'axes.grid': True,
	'grid.linestyle': ':',
	'grid.alpha': 0.5,
	'figure.dpi': 300,
	'savefig.dpi': 300,
	'savefig.bbox': 'tight',
	'savefig.pad_inches': 0.1
})

colors = {
	'ars': '#1f77b4',      # Blue
	'cars': '#ff7f0e',     # Orange  
	'gcd': '#2ca02c',      # Green
	'restart': '#d62728',  # Red
	'rsft': '#9467bd',     # Purple
	'rs': '#8c564b',       # Brown
	'awrs': '#e377c2',     # Pink
}

# Use this to control
LABEL_ORDER = ['rs', 'ars', 'rsft', 'cars', 'gcd', 'restart', 'awrs']
LABEL_MAPPING = {
    'restart': 'mcmc'
    }
SKIP_LIST = ['oldrs', 'oldars', 'oldcars', 'oldrsft', 'priority', 'prefix']
MIN_STEPS = 10
def kl_divergence(p, q):
	for key in p.keys():
		if key not in q:
			raise ValueError(f"Key {key} not found in q")
	for key in q.keys():
		if key not in p:
			raise ValueError(f"Key {key} not found in p")
	
	kl = 0.0
	for key in p.keys():
		if p[key] > 0 and q[key] > 0:
			kl += p[key] * np.log(p[key] / q[key])
	return kl

def parse_directory_name(dir_name: str) -> str:
	parts = dir_name.split('-')
	return parts[0] if parts else dir_name

def load_runs_data(base_path: str, min_steps: int = 1) -> dict:
	base_path = Path(base_path)
	current_dir = Path.cwd()
	full_path = current_dir / 'runs_log' / base_path
	method_runs = defaultdict(list)
	
	for run_dir in full_path.iterdir():
		if not run_dir.is_dir():
			continue
			
		method_name = parse_directory_name(run_dir.name)
		if method_name in SKIP_LIST:
			continue
		print(f"Processing {run_dir.name} -> method: {method_name}")
		
		for json_file in run_dir.glob("*.json"):
			try:
				with open(json_file, 'r') as f:
					run_data = json.load(f)
					if "steps" in run_data and len(run_data["steps"]) >= min_steps:
						method_runs[method_name].append(run_data)
			except Exception as e:
				print(f"Error loading {json_file}: {e}")
				continue
	
	for method, runs in method_runs.items():
		print(f"Loaded {len(runs)} samples for method {method}")
	
	return dict(method_runs)

def estimate_full_distribution(all_samples: list, distr_type: str) -> dict:
	if distr_type not in ["raw_logprob", "cons_logprob"]:
		raise ValueError("Invalid distribution type. Choose 'raw_logprob' or 'cons_logprob'.")
	
	logprobs = {}
	mismatches = 0
	
	for sample in all_samples:
		steps = sample.get("steps", [])
		for step in steps:
			if "token_ids" in step and isinstance(step["token_ids"], list):
				sample_ids = tuple(step["token_ids"])
				sample_logprob = step.get(distr_type, 0.0)
				
				if len(sample_ids) > 0:
					if sample_ids not in logprobs:
						logprobs[sample_ids] = sample_logprob
					else:
						if abs(logprobs[sample_ids] - sample_logprob) > 1e-6:
							mismatches += 1
			
			else:
				for side in ["current", "proposal"]:
					if side in step:
						sample_data = step[side]
						sample_ids = tuple(sample_data.get("token_ids", []))
						sample_logprob = sample_data.get(distr_type, 0.0)
						
						if len(sample_ids) > 0:
							if sample_ids not in logprobs:
								logprobs[sample_ids] = sample_logprob
							else:
								if abs(logprobs[sample_ids] - sample_logprob) > 1e-6:
									mismatches += 1
	
	print(f"Number of mismatches: {mismatches}")
	
	if not logprobs:
		return {}
	
	total_logprob = -np.inf
	for logprob in logprobs.values():
		total_logprob = np.logaddexp(total_logprob, logprob)
	
	normalized_logprobs = {sample_ids: logprob - total_logprob for sample_ids, logprob in logprobs.items()}
	distribution = {sample_ids: float(np.exp(logprob)) for sample_ids, logprob in normalized_logprobs.items()}
	
	total_prob = sum(distribution.values())
	print(f"Total probability sums to: {total_prob}")
	
	if not np.isclose(total_prob, 1.0):
		print(f"WARNING!!!!! Probabilities sum to {total_prob}, renormalizing...")
		distribution = {k: v/total_prob for k, v in distribution.items()}
	
	return distribution

def get_empirical_distribution(samples: list, n_steps: int = None) -> dict:
	counts = {}
	
	for sample in samples:
		steps = sample.get("steps", [])
		
		if n_steps is not None:
			if len(steps) > n_steps:
				step = steps[n_steps]
				if "current" in step and "token_ids" in step["current"]:
					sample_ids = tuple(step["current"]["token_ids"])
					if len(sample_ids) > 0:
						if sample_ids not in counts:
							counts[sample_ids] = 0
						counts[sample_ids] += 1
		else:
			for step in steps:
				if "token_ids" in step and isinstance(step["token_ids"], list):
					sample_ids = tuple(step["token_ids"])
					if len(sample_ids) > 0:
						if sample_ids not in counts:
							counts[sample_ids] = 0
						counts[sample_ids] += 1
				elif "current" in step and "token_ids" in step["current"]:
					sample_ids = tuple(step["current"]["token_ids"])
					if len(sample_ids) > 0:
						if sample_ids not in counts:
							counts[sample_ids] = 0
						counts[sample_ids] += 1
	
	if not counts:
		return {}
	
	total_count = sum(counts.values())
	empirical_distribution = {sample_ids: count / total_count for sample_ids, count in counts.items()}
	
	return empirical_distribution

def match_supports(method_distr: dict, target_distr: dict, keep_support: str = "target") -> tuple:
	res_method_distr = method_distr.copy()
	res_target_distr = target_distr.copy()
	
	if keep_support == "target":
		for sample in target_distr:
			if sample not in res_method_distr:
				res_method_distr[sample] = 0.0
	
	method_sum = sum(res_method_distr.values())
	target_sum = sum(res_target_distr.values())
	
	if method_sum > 0:
		res_method_distr = {k: v/method_sum for k, v in res_method_distr.items()}
	if target_sum > 0:
		res_target_distr = {k: v/target_sum for k, v in res_target_distr.items()}
	
	return res_method_distr, res_target_distr

def bootstrap_kl(samples: list, target_distr: dict, n_steps: int = None, n_bootstrap: int = 500) -> tuple:
	if not samples or not target_distr:
		return float('inf'), float('inf'), float('inf')
	
	n_samples = len(samples)
	bootstrap_kls = []
	
	for _ in range(n_bootstrap):
		# Resample with replacement
		resampled_indices = np.random.choice(n_samples, size=n_samples, replace=True)
		resampled = [samples[i] for i in resampled_indices]
		
		# Get empirical distribution from resampled data
		empirical_distr = get_empirical_distribution(resampled, n_steps)
		
		if not empirical_distr:
			bootstrap_kls.append(float('inf'))
			continue
		
		# Match supports and compute KL
		try:
			matched_method, matched_target = match_supports(empirical_distr, target_distr, keep_support="target")
			kl = kl_divergence(matched_method, matched_target)
			bootstrap_kls.append(kl)
		except:
			bootstrap_kls.append(float('inf'))
	
	# Remove infinite values
	bootstrap_kls = [kl for kl in bootstrap_kls if not np.isinf(kl)]
	
	if not bootstrap_kls:
		return float('inf'), float('inf'), float('inf')
	
	# Calculate CI
	lower_ci = np.percentile(bootstrap_kls, 2.5)
	upper_ci = np.percentile(bootstrap_kls, 97.5)
	mean_kl = np.mean(bootstrap_kls)
	
	return mean_kl, lower_ci, upper_ci

def plot_kl_runs(base_path: str, task_id: str, output_dir: str, distr_type: str = "raw_logprob"):

	method_runs = load_runs_data(base_path, min_steps=MIN_STEPS)
	
	if not method_runs:
		print("No data found!")
		return

	all_samples = []
	for samples in method_runs.values():
		all_samples.extend(samples)
	
	print("Estimating full distribution...")
	true_distribution = estimate_full_distribution(all_samples, distr_type)
	print(f"True distribution has {len(true_distribution)} unique sequences")
	
	if not true_distribution:
		print("Could not estimate true distribution!")
		return
	
	fig, ax = plt.subplots(1, 1, figsize=(8, 6))
	
	steps_total = 10
	steps_range = list(range(steps_total))
	
	sorted_methods = []
	for label in LABEL_ORDER:
		if label in method_runs:
			sorted_methods.append(label)
	
	for method in method_runs:
		if method not in sorted_methods:
			sorted_methods.append(method)
	
	if "restart" in method_runs:
		restart_samples = method_runs["restart"]
		empirical_distr = get_empirical_distribution(restart_samples, 0)
		if empirical_distr:
			matched_method, matched_target = match_supports(empirical_distr, true_distribution, keep_support="target")
			gcd_mean_kl = kl_divergence(matched_method, matched_target)
		else:
			gcd_mean_kl = float('inf')
		
		if not np.isinf(gcd_mean_kl):
			ax.axhline(y=gcd_mean_kl, color=colors.get('gcd', '#2ca02c'), linestyle=':', linewidth=3,
					  label='GCD', alpha=0.8)
			
			marker_positions = [0, 2, 4, 6, 8]
			ax.scatter(marker_positions, [gcd_mean_kl] * len(marker_positions), 
					  color=colors.get('gcd', '#2ca02c'), s=40, alpha=0.8, zorder=5)
			
			print(f"GCD KL divergence: {gcd_mean_kl:.4f}")
	
	for method in sorted_methods:
		samples = method_runs[method]
		color = colors.get(method, '#333333')
		
		if method == "restart":
			method_kls = []
			method_lower_cis = []
			method_upper_cis = []
			
			for n_steps in steps_range:
				mean_kl, lower_ci, upper_ci = bootstrap_kl(samples, true_distribution, n_steps, n_bootstrap=500)
				
				if np.isinf(mean_kl):
					mean_kl = 0.0
					lower_ci = 0.0
					upper_ci = 0.0
				
				method_kls.append(mean_kl)
				method_lower_cis.append(lower_ci)
				method_upper_cis.append(upper_ci)
				
				print(f"Restart step {n_steps} KL divergence: {mean_kl:.4f}")
			
			ax.plot(steps_range, method_kls, marker='o', linestyle='-', linewidth=2.5,
				   color=color, label=LABEL_MAPPING.get(method, method).upper(), alpha=0.8)
			ax.fill_between(steps_range, method_lower_cis, method_upper_cis,
						   alpha=0.2, color=color)
		
		else:
			empirical_distr = get_empirical_distribution(samples, None)
			if empirical_distr:
				matched_method, matched_target = match_supports(empirical_distr, true_distribution, keep_support="target")
				mean_kl = kl_divergence(matched_method, matched_target)
			else:
				mean_kl = float('inf')
			
			ax.axhline(y=mean_kl, color=color, linestyle=':', linewidth=3,
					label=LABEL_MAPPING.get(method, method).upper(), alpha=0.9)
			
			marker_positions = [0, 2, 4, 6, 8]
			ax.scatter(marker_positions, [mean_kl] * len(marker_positions), 
					color=color, s=40, alpha=0.8, zorder=5)
	
	ax.set_xlabel('Steps')
	# ax.set_ylabel(f'KL Divergence ({distr_type.replace("_", " ").title()})')
	ax.set_ylabel(f'KL Divergence')
	ax.set_title(task_id)
	ax.set_xlim(-0.5, steps_total - 0.5)
	ax.set_xticks(steps_range)
	ax.grid(True, alpha=0.3)
	ax.legend(loc='best', frameon=True)
	
	# Set y-axis to start from 0
	ax.set_ylim(bottom=0)
	
	plt.tight_layout()
	
	# Save the plot
	os.makedirs(output_dir, exist_ok=True)
	save_path = os.path.join(output_dir, f"{task_id.replace(' ', '_')}-kl_divergence.png")
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	print(f"Plot saved to {save_path}")
	
	plt.show()

if __name__ == "__main__":
	base_path = sys.argv[1]
	print(f"Got: {base_path}")
	task_name = base_path.split('-')[0]
	output_dir = "plots/kl-div"
	
	# Create the plot
	print("Creating KL divergence plot...")
	plot_kl_runs(
		base_path=base_path,
		task_id=task_name,
		output_dir=output_dir,
		distr_type='raw_logprob'
	)
	
	print("Plot completed successfully!")