#!/bin/bash

if [ "$#" -ne 2 ]; then
	echo "Usage: $0 sampling_style model"
	exit 1
fi

python run_task.py experiments/PDDL/domains/blocks/grammar.lark experiments/PDDL/prompts/blocks/task04_prompt.txt $1 $2
python run_task.py experiments/PDDL/domains/blocks/grammar.lark experiments/PDDL/prompts/blocks/task05_prompt.txt $1 $2
python run_task.py experiments/PDDL/domains/blocks/grammar.lark experiments/PDDL/prompts/blocks/task06_prompt.txt $1 $2

python run_task.py experiments/PDDL/domains/depot/grammar.lark experiments/PDDL/prompts/depot/task02_prompt.txt $1 $2
python run_task.py experiments/PDDL/domains/depot/grammar.lark experiments/PDDL/prompts/depot/task04_prompt.txt $1 $2
python run_task.py experiments/PDDL/domains/depot/grammar.lark experiments/PDDL/prompts/depot/task06_prompt.txt $1 $2

python run_task.py experiments/PDDL/domains/satellite/grammar.lark experiments/PDDL/prompts/satellite/task04_prompt.txt $1 $2
python run_task.py experiments/PDDL/domains/satellite/grammar.lark experiments/PDDL/prompts/satellite/task05_prompt.txt $1 $2
python run_task.py experiments/PDDL/domains/satellite/grammar.lark experiments/PDDL/prompts/satellite/task06_prompt.txt $1 $2

