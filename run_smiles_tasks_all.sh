#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 sampling_style"
	exit 1
fi

for a in datasets/smiles/*.lark; do
	python run_task.py $a ${a%lark}txt $1
done
