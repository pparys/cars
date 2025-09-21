#!/usr/bin/env python3
"""
Validate PDDL plans using VAL validator
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SampledPlans:
    """A data structure to hold a list of plans and their sample count."""
    plans: List[str]
    valid_samples_count: int
    total_samples_count: int

@dataclass 
class ValidationResults:
    """Results of plan validation"""
    total_plans: int
    syntactically_valid: int
    semantically_valid: int
    validation_details: List[Dict]

def validate_single_plan(domain_file: str, problem_file: str, plan_text: str) -> Dict:
    """
    Validate a single plan using VAL validator
    Returns: {syntactically_valid: bool, semantically_valid: bool, error_msg: str}
    """
    result = {
        'syntactically_valid': False,
        'semantically_valid': False, 
        'error_msg': ''
    }
    
    # Create temporary plan file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.plan', delete=False) as tmp_plan:
        tmp_plan.write(plan_text)
        tmp_plan_path = tmp_plan.name
    
    try:
        # First check if plan is syntactically valid (can be parsed)
        if not plan_text.strip():
            result['error_msg'] = 'Empty plan'
            return result
            
        # Check if plan has valid PDDL action format
        lines = [line.strip() for line in plan_text.split('\n') if line.strip()]
        valid_syntax = True
        for line in lines:
            if not line.startswith('(') or not line.endswith(')'):
                valid_syntax = False
                break
        
        if valid_syntax:
            result['syntactically_valid'] = True
        else:
            result['error_msg'] = 'Invalid PDDL syntax'
            return result
        
        # Use VAL to validate semantic correctness
        val_result = subprocess.run([
            'validate', domain_file, problem_file, tmp_plan_path
        ], capture_output=True, text=True, timeout=30)
        
        if val_result.returncode == 0:
            result['semantically_valid'] = True
        else:
            result['error_msg'] = f"VAL validation failed: {val_result.stdout + val_result.stderr}"
            
    except subprocess.TimeoutExpired:
        result['error_msg'] = 'Validation timeout'
    except FileNotFoundError:
        result['error_msg'] = 'VAL validator not found'
    except Exception as e:
        result['error_msg'] = f'Validation error: {str(e)}'
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_plan_path)
        except:
            pass
    
    return result

def validate_sampled_plans(sampled_plans: SampledPlans, domain_file: str, problem_file: str) -> ValidationResults:
    """
    Validate all plans in a SampledPlans object
    """
    validation_details = []
    syntactically_valid = 0
    semantically_valid = 0
    
    print(f"Validating {len(sampled_plans.plans)} plans...")
    
    for i, plan in enumerate(sampled_plans.plans):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(sampled_plans.plans)}")
            
        result = validate_single_plan(domain_file, problem_file, plan)
        validation_details.append({
            'plan_id': i,
            'plan_text': plan[:100] + "..." if len(plan) > 100 else plan,
            'syntactically_valid': result['syntactically_valid'],
            'semantically_valid': result['semantically_valid'],
            'error_msg': result['error_msg']
        })
        
        if result['syntactically_valid']:
            syntactically_valid += 1
        if result['semantically_valid']:
            semantically_valid += 1
    
    return ValidationResults(
        total_plans=len(sampled_plans.plans),
        syntactically_valid=syntactically_valid,
        semantically_valid=semantically_valid,
        validation_details=validation_details
    )

def get_domain_and_problem_files(domain_name: str, task_name: str) -> tuple:
    """
    Get the domain and problem file paths for a given domain and task
    """
    base_path = "/data/saiva/ars/experiments/PDDL/domains"
    
    domain_file = f"{base_path}/{domain_name}/domain.pddl"
    problem_file = f"{base_path}/{domain_name}/test/{task_name}.pddl"
    
    if not os.path.exists(domain_file):
        raise FileNotFoundError(f"Domain file not found: {domain_file}")
    if not os.path.exists(problem_file):
        raise FileNotFoundError(f"Problem file not found: {problem_file}")
        
    return domain_file, problem_file

def parse_experiment_path(exp_path: str) -> tuple:
    """
    Parse experiment path to extract domain and task
    Returns: (domain, task)
    """
    import re
    
    # Try to extract from path like: .../blocks_task04_qwen_25_7b
    path_parts = exp_path.split('/')
    for part in reversed(path_parts):
        match = re.match(r'(\w+)_task(\d+)_', part)
        if match:
            domain = match.group(1)
            task = f"task{match.group(2)}"
            return domain, task
    
    # Fallback: try to extract from any part of the path
    for part in path_parts:
        if 'blocks' in part.lower():
            domain = 'blocks'
        elif 'depot' in part.lower():
            domain = 'depot' 
        elif 'satellite' in part.lower():
            domain = 'satellite'
        else:
            continue
            
        # Look for task number
        task_match = re.search(r'task(\d+)', part)
        if task_match:
            task = f"task{task_match.group(1)}"
            return domain, task
    
    raise ValueError(f"Could not parse domain and task from path: {exp_path}")

def print_validation_summary(results: ValidationResults, domain: str, task: str):
    """Print a summary of validation results"""
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS: {domain.upper()} {task.upper()}")
    print(f"{'='*60}")
    print(f"Total plans: {results.total_plans}")
    print(f"Syntactically valid: {results.syntactically_valid} ({results.syntactically_valid/results.total_plans*100:.1f}%)")
    print(f"Semantically valid: {results.semantically_valid} ({results.semantically_valid/results.total_plans*100:.1f}%)")
    
    # Show some error examples
    error_types = {}
    for detail in results.validation_details:
        if detail['error_msg']:
            error_key = detail['error_msg'][:50]  # First 50 chars
            error_types[error_key] = error_types.get(error_key, 0) + 1
    
    if error_types:
        print(f"\nCommon errors:")
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {error}... ({count} times)")

def main():
    """
    Example usage of the validation system
    """
    from collect_plans import collect_plans  # Import your existing function
    
    # Example: validate a specific experiment
    exp_path = "/data/saiva/ars/experiments/PDDL/runs/blocks_task04_qwen_25_7b"
    
    if len(os.sys.argv) > 1:
        exp_path = os.sys.argv[1]
    
    try:
        # Parse domain and task from path
        domain, task = parse_experiment_path(exp_path)
        print(f"Detected domain: {domain}, task: {task}")
        
        # Get domain and problem files
        domain_file, problem_file = get_domain_and_problem_files(domain, task)
        print(f"Domain file: {domain_file}")
        print(f"Problem file: {problem_file}")
        
        # Collect plans from experiment
        sampled_plans_dict = collect_plans(exp_path)
        
        # Validate each run/style
        for run_name, sampled_plans in sampled_plans_dict.items():
            print(f"\nValidating run: {run_name}")
            results = validate_sampled_plans(sampled_plans, domain_file, problem_file)
            print_validation_summary(results, domain, task)
            
    except Exception as e:
        print(f"Error: {e}")
        print(f"Usage: python validate_plans.py [experiment_path]")

if __name__ == "__main__":
    main()