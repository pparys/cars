#!/bin/bash

python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql_with_grammar.txt rs 3
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql_with_grammar.txt rsft 3
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql_with_grammar.txt ars 3
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql_with_grammar.txt cars 3
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql_with_grammar.txt restart 3

