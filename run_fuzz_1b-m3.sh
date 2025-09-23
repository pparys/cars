#!/bin/bash

python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt rsft 3
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt cars 3
python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json.txt restart 3
