#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 sampling_style"
	exit 1
fi

python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt $1
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json_with_grammar.txt $1
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql.txt $1
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql_with_grammar.txt $1
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt $1
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml_with_grammar.txt $1
