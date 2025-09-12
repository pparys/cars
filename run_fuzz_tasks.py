import json
import os

import torch
from tqdm import tqdm

import utils
import lib
import mcmc

def load_fuzz_tasks():
	runs_params = [
		("xml", "datasets/fuzzing/prompts/xml_gen.txt", "datasets/fuzzing/grammars/xml.ebnf"),
		("sql", "datasets/fuzzing/prompts/sql_gen.txt", "datasets/fuzzing/grammars/sql.ebnf"),
		("sql-gr", "datasets/fuzzing/prompts/sql_gen_with_grammar.txt", "datasets/fuzzing/grammars/sql.ebnf")
	]
 
	tasks = []
	
	for run_type, prompt_file, grammar_file in runs_params:
		with open(prompt_file, "r") as f:
			prompt = f.read().strip()
							
		grammar_str = open(grammar_file).read()
  
		tasks.append({
			"id": run_type,
			"prompt": prompt,
			"grammar": grammar_str,
		})
	return tasks
	
	
def run_mcmc_fuzz_tasks(benchmark, styles):
	model_id = "meta-llama/Llama-3.1-8B-Instruct"
	if not torch.cuda.is_available():
		model_id = "hsultanbey/codegen350multi_finetuned"

	model = lib.ConstrainedModel(model_id, None, torch_dtype=torch.bfloat16)
	#model = lib.ConstrainedModel(model_id, None, torch_dtype=torch.float32) ###PP!!!!!

	root_log_dir = "fuzz_runs"

	split_log_dir = f"{root_log_dir}/{utils.timestamp()}-{benchmark}"
	# make sure the directory exists
	os.makedirs(split_log_dir, exist_ok=True)

	n_samples = 1 #100
	n_steps = 1000 # 10
	max_new_tokens = 512
 
	tasks = load_fuzz_tasks()
	benchmark_tasks = [task for task in tasks if task["id"] == benchmark]

	for task in tqdm(benchmark_tasks):
		task_id = task["id"] 
		task_prompt = task["prompt"]
		task_grammar = task["grammar"]

		model._set_grammar_constraint(task_grammar)
		for sample_style in styles:
			print(f"Benchmark: {task_id}, Sample Style: {sample_style}")

			mcmc_runner = mcmc.MCMC(
				model=model,
				prompt=task_prompt,
				sample_style=sample_style,
				name_prefix=task_id,
				root_log_dir=split_log_dir,
			)
			mcmc_runner.get_samples(
				n_samples=n_samples,
				n_steps=n_steps,
				max_new_tokens=max_new_tokens,
			)


if __name__ == "__main__":
	from argparse import ArgumentParser

	parser = ArgumentParser()
	parser.add_argument("--benchmark", required=True, choices=["xml", "sql", "sql-gr"])
	parser.add_argument("--styles", default=None)

	args = parser.parse_args()
	benchmark = args.benchmark
	print(f"Benchmark: {benchmark}")

	run_mcmc_fuzz_tasks(benchmark, mcmc.parse_styles_arg(args.styles))