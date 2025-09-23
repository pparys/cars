#!/bin/bash

python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt restart 3
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml_with_grammar.txt restart 3
