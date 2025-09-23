#!/bin/bash

python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml_with_grammar.txt rs 2
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml_with_grammar.txt rsft 2
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml_with_grammar.txt ars 2
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml_with_grammar.txt cars 2
python run_task.py datasets/fuzzing/xml.lark datasets/fuzzing/generate_xml_with_grammar.txt restart 2
