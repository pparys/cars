import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict
import glob
import re

@dataclass
class SampledPlans:
    """A data structure to hold a list of plans and their sample count."""
    plans: List[str]
    valid_samples_count: int
    total_samples_count: int

def collect_plans(input_dir: str) -> Dict[str, SampledPlans]:
    """
    Collects plans from each run directory, keeping each run separate.
    The key in the returned dictionary is the full run directory name.
    """
    results_per_run = {}

    for run_dir_name in sorted(os.listdir(input_dir)):
        full_run_path = os.path.join(input_dir, run_dir_name)

        if os.path.isdir(full_run_path):
            current_run_plans = []
            current_run_successes = []

            for filename in sorted(os.listdir(full_run_path)):
                if not filename.endswith('.json'):
                    continue

                json_path = os.path.join(full_run_path, filename)
                try:
                    with open(json_path, 'r') as f:
                        sample_data = json.load(f)

                    current_run_successes = sample_data.get('successes', [])

                    for step in sample_data.get('steps', []):
                        molecule = ''.join(step.get('tokens', [])[:-1])
                        current_run_plans.append(molecule)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Warning: Could not process file {json_path}. Error: {e}")
                    continue

            results_per_run[run_dir_name] = SampledPlans(
                plans=current_run_plans[:100],
                valid_samples_count=sum(current_run_successes),
                total_samples_count=len(current_run_successes)
            )

    return results_per_run

def collect_all_mcmc_plans(input_dir: str) -> Dict[str, Dict[str, SampledPlans]]:
    """
    Collects MCMC plans and returns them structured with the SampledPlans dataclass.
    """
    main_info: Dict[str, Dict[str, List[str]]] = {}
    step_counts = [5]
    
    for method_dir in sorted(os.listdir(input_dir)):
        sub_info: Dict[str, List[str]] = {}
        mcmc_strategy_dir = os.path.join(input_dir, method_dir)
        
        try:
            task, mcmc_strategy = method_dir.split('-', 1)
        except ValueError:
            continue 
            
        if mcmc_strategy != 'restart':
            continue
            
        monomer_class = task.split('_', 1)[1]
        
        for filename in sorted(os.listdir(mcmc_strategy_dir)):
            json_path = os.path.join(mcmc_strategy_dir, filename)
            with open(json_path, 'r') as f:
                sample_data = json.load(f)
            
            proposals = [step["proposal"]["tokens"] for step in sample_data.get("steps", [])]
            for step in step_counts:
                if step > len(proposals): continue
                
                molecule = ''.join(proposals[step-1][:-1])
                strategy = f'mcmc-{mcmc_strategy}-{step}'
                
                if strategy not in sub_info:
                    sub_info[strategy] = []
                sub_info[strategy].append(molecule)

        if monomer_class not in main_info:
            main_info[monomer_class] = {}
        main_info[monomer_class].update(sub_info)
    
    return {
        monomer_class: {
            strategy: SampledPlans(plans=plans[:100],
                                       valid_samples_count=100,
                					  total_samples_count=100 * int(strategy[-1]))
            for strategy, plans in sub_info.items()
        }
        for monomer_class, sub_info in main_info.items()
    }

def collect_specific_mcmc_plans(input_dir, monomer_class):
	all_plans = collect_all_mcmc_plans(input_dir)
	return all_plans.get(monomer_class, {})

def aggregate_results_by_style(all_results: Dict[str, Dict[str, SampledPlans]]) -> Dict[str, Dict[str, Dict]]:
    """
    Aggregate results by style across all domains and tasks.
    Returns: {domain: {style: {total_valid, total_samples, success_rate, tasks}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: {
        'total_valid': 0, 
        'total_samples': 0, 
        'tasks': []
    }))
    
    for domain, domain_results in all_results.items():
        for task, task_results in domain_results.items():
            for run_key, sp in task_results.items():
                style = run_key.split('-', 1)[0]
                
                aggregated[domain][style]['total_valid'] += sp.valid_samples_count
                aggregated[domain][style]['total_samples'] += sp.total_samples_count
                aggregated[domain][style]['tasks'].append(f"{task}: {sp.valid_samples_count}/{sp.total_samples_count}")
    
    # Calculate success rates
    for domain in aggregated:
        for style in aggregated[domain]:
            total_valid = aggregated[domain][style]['total_valid']
            total_samples = aggregated[domain][style]['total_samples']
            success_rate = (total_valid / total_samples * 100) if total_samples > 0 else 0
            aggregated[domain][style]['success_rate'] = success_rate
    
    return dict(aggregated)

def print_results_table(aggregated_results: Dict[str, Dict[str, Dict]]):
    """Print results in a clean table format"""
    print("=" * 80)
    print("PDDL Planning Results - Grammar-Constrained Generation")
    print("Model: Qwen/Qwen-2.5-7B-Instruct")
    print("=" * 80)
    
    # Overall summary table
    print("\nOVERALL RESULTS BY DOMAIN AND STYLE")
    print("-" * 60)
    print(f"{'Domain':<12} {'Style':<15} {'Valid':<8} {'Total':<8} {'Success Rate':<12}")
    print("-" * 60)
    
    for domain in sorted(aggregated_results.keys()):
        for style in sorted(aggregated_results[domain].keys()):
            stats = aggregated_results[domain][style]
            print(f"{domain:<12} {style:<15} {stats['total_valid']:<8} {stats['total_samples']:<8} {stats['success_rate']:<11.1f}%")
        print()
    
    # Detailed breakdown by domain
    for domain in sorted(aggregated_results.keys()):
        print(f"\n{domain.upper()} DOMAIN - DETAILED BREAKDOWN")
        print("-" * 50)
        
        for style in sorted(aggregated_results[domain].keys()):
            stats = aggregated_results[domain][style]
            print(f"\nStyle: {style}")
            print(f"  Overall: {stats['total_valid']}/{stats['total_samples']} ({stats['success_rate']:.1f}%)")
            print("  Per task:")
            for task_result in stats['tasks']:
                print(f"    {task_result}")

def get_model_name(model_id):
    """Map model ID to readable name"""
    model_map = {
        'llama_31_8b': 'Llama-3.1-8B-Instruct',
        'qwen_25_7b': 'Qwen/Qwen-2.5-7B-Instruct', 
        'qwen_25_14b': 'Qwen/Qwen-2.5-14B-Instruct'
    }
    return model_map.get(model_id, model_id)

def discover_experiments():
    """Auto-discover experiment paths from organized directory structure"""
    import glob
    import re
    from collections import defaultdict
    
    base_path = "experiments/PDDL/runs"
    experiments_by_model = defaultdict(lambda: defaultdict(dict))
    
    # Find all organized experiment directories
    pattern = os.path.join(base_path, "*")
    dirs = glob.glob(pattern)
    
    for dir_path in dirs:
        if os.path.isdir(dir_path):
            dir_name = os.path.basename(dir_path)
            
            # Parse format: {domain}_task{XX}_{model_id}
            match = re.match(r'(\w+)_task(\d+)_(\w+)', dir_name)
            if match:
                domain = match.group(1)
                task = f"task{match.group(2)}"
                model_id = match.group(3)
                
                experiments_by_model[model_id][domain][task] = dir_path
                print(f"Found: {domain} {task} ({model_id}) -> {dir_path}")
    
    return dict(experiments_by_model)

def print_results_table_for_model(aggregated_results: Dict[str, Dict[str, Dict]], model_name: str):
    """Print results in a clean table format for a specific model"""
    print("=" * 80)
    print(f"PDDL Planning Results")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    # Overall summary table
    print("\nOVERALL RESULTS BY DOMAIN AND STYLE")
    print("-" * 60)
    print(f"{'Domain':<12} {'Style':<15} {'Valid':<8} {'Total':<8} {'Syntactic V':<12}")
    print("-" * 60)
    
    for domain in sorted(aggregated_results.keys()):
        for style in sorted(aggregated_results[domain].keys()):
            stats = aggregated_results[domain][style]
            print(f"{domain:<12} {style:<15} {stats['total_valid']:<8} {stats['total_samples']:<8} {stats['success_rate']:<11.1f}%")
        print()
    
    # Style comparison across domains
    print("STYLE COMPARISON ACROSS ALL DOMAINS")
    print("-" * 45)
    
    style_totals = defaultdict(lambda: {'valid': 0, 'total': 0})
    for domain_results in aggregated_results.values():
        for style, stats in domain_results.items():
            style_totals[style]['valid'] += stats['total_valid']
            style_totals[style]['total'] += stats['total_samples']
    
    print(f"{'Style':<15} {'Valid':<8} {'Total':<8} {'Syntactic V':<12}")
    print("-" * 45)
    for style in sorted(style_totals.keys()):
        valid = style_totals[style]['valid']
        total = style_totals[style]['total']
        rate = (valid / total * 100) if total > 0 else 0
        print(f"{style:<15} {valid:<8} {total:<8} {rate:<11.1f}%")

def main():
    # Auto-discover experiment paths for all models
    experiments_by_model = discover_experiments()
    
    if not experiments_by_model:
        print("No experiment directories found in /graft2/code/emmanuel/ars/experiments/PDDL/runs")
        return
    
    # Process each model separately
    for model_id, experiments in experiments_by_model.items():
        model_name = get_model_name(model_id)
        
        # Collect all results for this model
        all_results = {}
        for domain, domain_experiments in experiments.items():
            all_results[domain] = {}
            for task, path in domain_experiments.items():
                all_results[domain][task] = collect_plans(path)
        
        # Aggregate and print results for this model
        aggregated = aggregate_results_by_style(all_results)
        print_results_table_for_model(aggregated, model_name)
        print("\n" + "="*80 + "\n")
    
    # Collect all results
    all_results = {}
    for domain, domain_experiments in experiments.items():
        all_results[domain] = {}
        for task, path in domain_experiments.items():
            all_results[domain][task] = collect_plans(path)
    
    # Aggregate and print results
    # aggregated = aggregate_results_by_style(all_results)
    # print_results_table(aggregated)
    
    # # Style comparison across domains
    # print("\n" + "=" * 80)
    # print("STYLE COMPARISON ACROSS ALL DOMAINS")
    # print("=" * 80)
    
    # style_totals = defaultdict(lambda: {'valid': 0, 'total': 0})
    # for domain_results in aggregated.values():
    #     for style, stats in domain_results.items():
    #         style_totals[style]['valid'] += stats['total_valid']
    #         style_totals[style]['total'] += stats['total_samples']
    
    # print(f"{'Style':<15} {'Valid':<8} {'Total':<8} {'Syntactic V':<12}")
    # print("-" * 45)
    # for style in sorted(style_totals.keys()):
    #     valid = style_totals[style]['valid']
    #     total = style_totals[style]['total']
    #     rate = (valid / total * 100) if total > 0 else 0
    #     print(f"{style:<15} {valid:<8} {total:<8} {rate:<11.1f}%")

if __name__ == "__main__":
    main()