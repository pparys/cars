import json
import os

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SampledMolecules:
    """A data structure to hold a list of molecules and their sample count."""
    molecules: List[str]
    valid_samples_count: int
    total_samples_count: int

def collect_molecules(input_dir: str) -> Dict[str, SampledMolecules]:
    """
    Collects molecules from each run directory, keeping each run separate.
    The key in the returned dictionary is the full run directory name.
    """
    results_per_run = {}

    for run_dir_name in sorted(os.listdir(input_dir)):
        full_run_path = os.path.join(input_dir, run_dir_name)

        if os.path.isdir(full_run_path):
            current_run_molecules = []
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
                        current_run_molecules.append(molecule)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Warning: Could not process file {json_path}. Error: {e}")
                    continue

            results_per_run[run_dir_name] = SampledMolecules(
                molecules=current_run_molecules[:100],
                valid_samples_count=sum(current_run_successes),
                total_samples_count=len(current_run_successes)
            )

    return results_per_run

def collect_all_mcmc_molecules(input_dir: str) -> Dict[str, Dict[str, SampledMolecules]]:
    """
    Collects MCMC molecules and returns them structured with the SampledMolecules dataclass.
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
            strategy: SampledMolecules(molecules=molecules[:100],
                                       valid_samples_count=100,
                					  total_samples_count=100 * int(strategy[-1]))
            for strategy, molecules in sub_info.items()
        }
        for monomer_class, sub_info in main_info.items()
    }

def collect_specific_mcmc_molecules(input_dir, monomer_class):
	all_molecules = collect_all_mcmc_molecules(input_dir)
	return all_molecules.get(monomer_class, {})