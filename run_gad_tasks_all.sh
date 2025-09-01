#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup python run_gad_tasks.py --split BV4  >log-BV4.txt &
CUDA_VISIBLE_DEVICES=2 nohup python run_gad_tasks.py --split CP >log-CP.txt &
#CUDA_VISIBLE_DEVICES=4 nohup python run_gad_tasks.py --split SLIA >log-SLIA.txt &

#W SLIA pominiete 3, 7 (potrzebuje wiecej pamieci)