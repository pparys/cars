#!/bin/bash

if [ "$#" -ne 3 ]; then
	echo "Usage: $0 dataset sampling_style model"
	exit 1
fi

for a in datasets/$1/*.ebnf; do
	python run_task.py $a ${a%ebnf}txt $2 $3
done
