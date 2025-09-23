#!/bin/bash

python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt rs 3
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt rsft 3
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt ars 3
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt cars 3
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt restart 3

python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt rsft 3
