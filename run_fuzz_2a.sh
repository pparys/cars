#!/bin/bash

python run_task.py datasets/fuzzing/json.lark datasets/fuzzing/generate_json_with_grammar.txt restart 2

python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt cars 2
