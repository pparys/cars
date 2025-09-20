#!/bin/bash

if [ "$#" -ne 2 ]; then
	echo "Usage: $0 sampling_style model"
	exit 1
fi

for run in {1..2}; do    
    python run_task.py experiments/PDDL/domains/blocks/grammar.lark experiments/PDDL/prompts/blocks/task04_prompt.txt $1 $2
    python run_task.py experiments/PDDL/domains/blocks/grammar.lark experiments/PDDL/prompts/blocks/task05_prompt.txt $1 $2
    python run_task.py experiments/PDDL/domains/blocks/grammar.lark experiments/PDDL/prompts/blocks/task08_prompt.txt $1 $2
    python run_task.py experiments/PDDL/domains/blocks/grammar.lark experiments/PDDL/prompts/blocks/task09_prompt.txt $1 $2


    python run_task.py experiments/PDDL/domains/depot/grammar.lark experiments/PDDL/prompts/depot/task02_prompt.txt $1 $2
    python run_task.py experiments/PDDL/domains/depot/grammar.lark experiments/PDDL/prompts/depot/task04_prompt.txt $1 $2
    python run_task.py experiments/PDDL/domains/depot/grammar.lark experiments/PDDL/prompts/depot/task07_prompt.txt $1 $2
    python run_task.py experiments/PDDL/domains/depot/grammar.lark experiments/PDDL/prompts/depot/task08_prompt.txt $1 $2

    python run_task.py experiments/PDDL/domains/satellite/grammar.lark experiments/PDDL/prompts/satellite/task03_prompt.txt $1 $2
    python run_task.py experiments/PDDL/domains/satellite/grammar.lark experiments/PDDL/prompts/satellite/task05_prompt.txt $1 $2
    python run_task.py experiments/PDDL/domains/satellite/grammar.lark experiments/PDDL/prompts/satellite/task07_prompt.txt $1 $2
    python run_task.py experiments/PDDL/domains/satellite/grammar.lark experiments/PDDL/prompts/satellite/task08_prompt.txt $1 $2

    echo "Completed run $run"
    echo ""
done