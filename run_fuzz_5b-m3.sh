#!/bin/bash

python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt rs 3
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt rsft 3
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt ars 3
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt cars 3
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt restart 3
