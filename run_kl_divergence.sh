#!/bin/bash

for a in runs_log/*1; do
	tmp=${a##*/}
	python plot_kl_divergence.py $tmp
done
