#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 sampling_style"
	exit 1
fi

python run_task.py json.lark generate_json.txt $1
python run_task.py json.lark generate_json_with_grammar.txt $1
python run_task.py sql.lark generate_sql.txt $1
python run_task.py sql.lark generate_sql_with_grammar.txt $1
python run_task.py xml.lark generate_xml.txt $1
python run_task.py xml.lark generate_xml_with_grammar.txt $1
