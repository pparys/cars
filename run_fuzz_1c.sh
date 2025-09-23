#!/bin/bash

python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt rs 2
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt rsft 2
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt ars 2
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt cars 2
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt restart 2

python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt ars 2

