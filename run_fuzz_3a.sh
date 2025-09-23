#!/bin/bash

python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql.txt rsft 2
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql.txt ars 2
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql.txt cars 2
python run_task.py datasets/fuzzing/sql.lark datasets/fuzzing/generate_sql.txt restart 2

python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml_with_grammar.txt rsft 2
