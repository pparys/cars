#!/usr/bin/env python3
"""
Generate few-shot prompts for PDDL plan generation
"""

import os
import glob
import re
from pathlib import Path

def read_file(filepath):
	"""Read file content"""
	try:
		with open(filepath, 'r') as f:
			return f.read().strip()
	except Exception as e:
		print(f"Error reading {filepath}: {e}")
		return None

def extract_problem_content(pddl_content):
	"""Extract the problem part (objects, init, goal) from PDDL file"""
	# Remove the problem definition wrapper, keep only objects, init, goal
	lines = pddl_content.split('\n')
	content_lines = []
	inside_relevant = False
	paren_count = 0
	
	for line in lines:
		stripped = line.strip()
		if stripped.startswith('(:objects') or stripped.startswith('(:init') or stripped.startswith('(:goal'):
			inside_relevant = True
			
		if inside_relevant:
			content_lines.append(line)
			# Count parentheses to know when we're done
			paren_count += line.count('(') - line.count(')')
			
			# If we've closed all parens and we were in a section, we might be done
			if paren_count == 0 and stripped.endswith(')'):
				# Check if this completes the problem (we have objects, init, and goal)
				content_str = '\n'.join(content_lines)
				if '(:objects' in content_str and '(:init' in content_str and '(:goal' in content_str:
					break
	
	return '\n'.join(content_lines)

def format_plan_actions(plan_content):
	"""Format plan actions as a single line sequence"""
	actions = []
	for line in plan_content.split('\n'):
		line = line.strip()
		if line and not line.startswith(';') and line.startswith('(') and line.endswith(')'):
			actions.append(line)
	
	return ' '.join(actions)

def build_domain_prompts(domain_name):
	"""Build prompts for all test tasks in a domain"""
	base_path = f"PDDL/domains/{domain_name}"
	domain_file = f"{base_path}/domain.pddl"
	grammar_file = f"{base_path}/grammar.lark"
	examples_tasks_dir = f"{base_path}/examples/tasks"
	examples_solutions_dir = f"{base_path}/examples/solutions"
	test_tasks_dir = f"{base_path}/test"
	
	print("YES")
	print(domain_name)
	remove_last = None if domain_name == 'satellite' else -1
	print(remove_last)
	
	# Read domain
	domain_content = read_file(domain_file)
	if not domain_content:
		print(f"Could not read domain file for {domain_name}")
		return []
	
	grammar_content = read_file(grammar_file)
	if not grammar_content:
		print(f"Could not read domain file for {domain_name}")
		return []
	
	# Get example tasks and solutions
	example_task_files = sorted(glob.glob(f"{examples_tasks_dir}/task*.pddl"))
	examples = []
	
	for task_file in example_task_files:
		task_name = os.path.basename(task_file)
		solution_file = f"{examples_solutions_dir}/{task_name}.soln"
		
		task_content = read_file(task_file)
		solution_content = read_file(solution_file)
		
		if task_content and solution_content:
			problem_part = extract_problem_content(task_content)
			plan_sequence = format_plan_actions(solution_content)
			
			examples.append({
				'problem': problem_part,
				'solution': plan_sequence,
				'task_name': task_name.replace('.pddl', '')
			})
	
	print(f"Found {len(examples)} examples for {domain_name}")
	
	# Get test tasks
	test_task_files = sorted(glob.glob(f"{test_tasks_dir}/task*.pddl"))
	prompts = []
	
	for test_file in test_task_files:
		test_task_name = os.path.basename(test_file).replace('.pddl', '')
		test_content = read_file(test_file)
		
		if not test_content:
			continue
			
		test_problem = extract_problem_content(test_content)
		
		# Build the prompt
		prompt = f"""You are a PDDL planning expert. You are given a domain, and some examples of planning problems and a valid sequences to achieve the goal. 

A plan is valid if and only if for every action in the sequence, all of its preconditions (as defined in the domain file) are satisfied in the state of the world before the action is executed.

To ensure correctness, you must reason step-by-step internally:
1.  Analyze the initial state.
2.  For each step, select an action that makes progress toward the goal.
3.  Update the world state based on the action's effects.
4.  Repeat until all goal conditions are met.

Your final output must be ONLY the valid sequence of actions.

Domain: {domain_name.upper()}

Domain Definition:
{domain_content}

Grammar for valid actions in LARK format:
{grammar_content}

"""
		
		# Add examples
		for i, example in enumerate(examples, 1):
			prompt += f"""
Problem:
{example['problem'][:remove_last]}
 
Solution:
{example['solution']}
"""
		
		# Add test problem
		prompt += f"""
Problem:
{test_problem[:remove_last]} 

Solution:"""
		
		prompts.append({
			'domain': domain_name,
			'task': test_task_name,
			'prompt': prompt
		})
	
	return prompts

def main():
	"""Generate all prompts"""
	output_dir = "PDDL/prompts"
	os.makedirs(output_dir, exist_ok=True)
	
	all_prompts = []
	
	for domain_name in ["blocks", "depot", "satellite"]:
		print(f"\nProcessing {domain_name}...")
		
		domain_prompts = build_domain_prompts(domain_name)
		all_prompts.extend(domain_prompts)
		
		# Create domain-specific output directory
		domain_output_dir = f"{output_dir}/{domain_name}"
		os.makedirs(domain_output_dir, exist_ok=True)
		
		# Save prompts
		for prompt_data in domain_prompts:
			filename = f"{domain_output_dir}/{prompt_data['task']}_prompt.txt"
			with open(filename, 'w') as f:
				f.write(prompt_data['prompt'])
			print(f"  Created: {filename}")
	
	# Summary
	print(f"\n=== SUMMARY ===")
	for domain in ["blocks", "depot", "satellite"]:
		domain_count = len([p for p in all_prompts if p['domain'] == domain])
		print(f"{domain}: {domain_count} prompts generated")
	
	print(f"\nTotal: {len(all_prompts)} prompts saved to {output_dir}/")
	
	# Show a sample prompt
	if all_prompts:
		print(f"\n=== SAMPLE PROMPT (first 500 chars) ===")
		sample = all_prompts[0]['prompt'][:500]
		print(sample + "..." if len(all_prompts[0]['prompt']) > 500 else sample)

if __name__ == "__main__":
	main()