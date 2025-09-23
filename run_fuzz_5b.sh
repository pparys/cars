#!/bin/bash

python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt rsft 2
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt ars 2
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt cars 2
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml.txt restart 2
