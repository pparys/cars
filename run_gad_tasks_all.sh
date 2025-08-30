#!/bin/bash

nohup python run_gad_tasks.py --split BV4 --styles=rs,cftrs >log-BV4.txt &
nohup python run_gad_tasks.py --split CP --styles=rs,cftrs >log-CP.txt &
nohup python run_gad_tasks.py --split SLIA --styles=rs,cftrs >log-SLIA.txt &
