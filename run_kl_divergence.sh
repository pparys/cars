#!/bin/bash

# Plots for Llama-3.1-8B
for a in runs_log/*1; do
	tmp=${a##*/}
	python plot_kl_divergence.py $tmp 1
done

# Plots for Qwen2.5-7B
for a in runs_log/*2; do
	tmp=${a##*/}
	python plot_kl_divergence.py $tmp 2
done
