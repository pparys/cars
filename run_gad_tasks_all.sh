#!/bin/bash

CUDA_VISIBLE_DEVICES=2 nohup python run_gad_tasks.py --split BV4 --styles=rs,cftrs >log-BV4.txt &
CUDA_VISIBLE_DEVICES=3 nohup python run_gad_tasks.py --split CP --styles=rs,cftrs >log-CP.txt &
CUDA_VISIBLE_DEVICES=4 nohup python run_gad_tasks.py --split SLIA --styles=rs,cftrs >log-SLIA.txt &
