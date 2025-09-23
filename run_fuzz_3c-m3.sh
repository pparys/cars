#!/bin/bash

python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql.txt rsft 3
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql.txt ars 3
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql.txt cars 3
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql.txt restart 3

python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml_with_grammar.txt cars 3
