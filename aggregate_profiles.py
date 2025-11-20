#!/usr/bin/env python3
import json
import sys
import glob
from pathlib import Path
import numpy as np


def load_profile(path):
    with open(path) as f:
        return json.load(f)


def get_run_name(path):
    parts = Path(path).parts
    if len(parts) >= 3:
        problem = parts[-3]  # e.g., "smiles_chain_extenders-0b0c52a1-2"
        problem_clean = '-'.join(problem.split('-')[:-2]) if '-' in problem else problem
        return problem_clean
    return Path(path).parent.parent.name


def aggregate_profiles(profile_paths):
    if len(profile_paths) == 0:
        print("No profile files found")
        sys.exit(1)
    
    print(f"Found {len(profile_paths)} CARS profile(s)")
    print()
    
    profiles = []
    names = []
    
    for path in profile_paths:
        try:
            profiles.append(load_profile(path))
            names.append(get_run_name(path))
            print(f"Loaded: {path}")
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    
    if len(profiles) == 0:
        print("No valid profiles loaded")
        sys.exit(1)
    
    print()
    print("=" * 100)
    print(f"CARS AGGREGATE STATISTICS ({len(profiles)} runs)")
    print("=" * 100)
    print()
    
    # Extract all metrics
    def extract_metric(profiles, getter, default=0):
        values = []
        for p in profiles:
            try:
                values.append(getter(p))
            except:
                values.append(default)
        return values
    
    # Sampling metrics
    total_attempts = extract_metric(profiles, lambda p: p['sampling_stats']['total_samples_attempted'])
    successful = extract_metric(profiles, lambda p: p['sampling_stats']['successful_samples'])
    failed = extract_metric(profiles, lambda p: p['sampling_stats']['failed_samples'])
    total_tokens = extract_metric(profiles, lambda p: p['sampling_stats']['total_tokens_generated'])
    avg_tokens = extract_metric(profiles, lambda p: p['sampling_stats']['avg_tokens_per_sample'])
    success_rates = extract_metric(profiles, lambda p: p['sampling_stats']['successful_samples'] / max(1, p['sampling_stats']['total_samples_attempted']) * 100)
    
    # Trie metrics
    trie_nodes = extract_metric(profiles, lambda p: p['trie_stats']['total_nodes'])
    trie_depth = extract_metric(profiles, lambda p: p['trie_stats']['max_depth'])
    trie_reuse = extract_metric(profiles, lambda p: p['sampling_stats']['trie_reuse_rate'] * 100)
    branching = extract_metric(profiles, lambda p: p['trie_stats']['avg_branching_factor'])
    
    # Memory metrics
    trie_cpu_mb = extract_metric(profiles, lambda p: p['memory_stats']['trie_cpu_bytes'] / (1024**2))
    trie_gpu_mb = extract_metric(profiles, lambda p: p['memory_stats']['trie_gpu_bytes'] / (1024**2))
    trie_total_mb = [cpu + gpu for cpu, gpu in zip(trie_cpu_mb, trie_gpu_mb)]
    
    # Time metrics
    total_time = extract_metric(profiles, lambda p: p['total_time'])
    inference_time = extract_metric(profiles, lambda p: p['compute_stats']['total_inference_time'])
    logits_time = extract_metric(profiles, lambda p: p['compute_stats']['total_logits_processing_time'])
    trie_ops_time = extract_metric(profiles, lambda p: p['compute_stats']['total_trie_operations_time'])
    
    # Compute derived metrics
    tokens_per_second = [tok / max(t, 1) for tok, t in zip(total_tokens, total_time)]
    
    # Time percentages
    inference_pct = [inf / max(tot, 1) * 100 for inf, tot in zip(inference_time, total_time)]
    logits_pct = [log / max(tot, 1) * 100 for log, tot in zip(logits_time, total_time)]
    trie_pct = [tri / max(tot, 1) * 100 for tri, tot in zip(trie_ops_time, total_time)]
    unaccounted_pct = [100 - (i + l + t) for i, l, t in zip(inference_pct, logits_pct, trie_pct)]
    
    # Operation counts
    forward_passes = extract_metric(profiles, lambda p: p['compute_stats']['num_forward_passes'])
    trie_lookups = extract_metric(profiles, lambda p: p['compute_stats']['num_trie_lookups'])
    trie_insertions = extract_metric(profiles, lambda p: p['compute_stats']['num_trie_insertions'])
    recomputations = extract_metric(profiles, lambda p: p['compute_stats']['num_recomputations'])
    
    def print_stats(name, values, unit="", decimals=2):
        if len(values) == 0:
            return
        
        values_array = np.array(values)
        total = np.sum(values_array)
        mean = np.mean(values_array)
        std = np.std(values_array)
        min_val = np.min(values_array)
        max_val = np.max(values_array)
        median = np.median(values_array)
        
        if decimals == 0:
            print(f"{name:<40} Total: {total:>10,.0f}{unit}  Mean: {mean:>8,.0f}{unit}  Median: {median:>8,.0f}{unit}  (Range: {min_val:,.0f}-{max_val:,.0f})")
        else:
            print(f"{name:<40} Total: {total:>10,.{decimals}f}{unit}  Mean: {mean:>8,.{decimals}f}{unit}  Median: {median:>8,.{decimals}f}{unit}  (Range: {min_val:.{decimals}f}-{max_val:.{decimals}f})")
    
    print("SAMPLING STATISTICS")
    print("-" * 100)
    print_stats("Total samples attempted", total_attempts, "", 0)
    print_stats("Successful samples", successful, "", 0)
    print_stats("Failed samples", failed, "", 0)
    print_stats("Success rate", success_rates, "%", 1)
    print_stats("Total tokens generated", total_tokens, "", 0)
    print_stats("Avg tokens per sample", avg_tokens, "", 1)
    print_stats("Throughput", tokens_per_second, " tok/s", 2)
    print()
    
    print("TRIE STATISTICS")
    print("-" * 100)
    print_stats("Trie nodes", trie_nodes, "", 0)
    print_stats("Trie max depth", trie_depth, "", 0)
    print_stats("Trie reuse rate", trie_reuse, "%", 1)
    print_stats("Avg branching factor", branching, "", 2)
    print()
    
    print("MEMORY USAGE")
    print("-" * 100)
    print_stats("Trie CPU memory", trie_cpu_mb, " MB", 2)
    print_stats("Trie GPU memory", trie_gpu_mb, " MB", 2)
    print_stats("Total trie memory", trie_total_mb, " MB", 2)
    print()
    
    print("COMPUTE STATISTICS")
    print("-" * 100)
    print_stats("Total time", total_time, " s", 2)
    print_stats("Inference time", inference_time, " s", 2)
    print_stats("Logits processing time", logits_time, " s", 2)
    print_stats("Trie operations time", trie_ops_time, " s", 2)
    print()
    print_stats("Inference time %", inference_pct, "%", 1)
    print_stats("Logits processing %", logits_pct, "%", 1)
    print_stats("Trie operations %", trie_pct, "%", 1)
    print_stats("Unaccounted time %", unaccounted_pct, "%", 1)
    print()
    
    print("OPERATION COUNTS")
    print("-" * 100)
    print_stats("Forward passes", forward_passes, "", 0)
    print_stats("Trie lookups", trie_lookups, "", 0)
    print_stats("Trie insertions", trie_insertions, "", 0)
    print_stats("Recomputations", recomputations, "", 0)
    print()
    
    print()
    print("=" * 100)
    print("PER-RUN BREAKDOWN")
    print("=" * 100)
    print()
    
    print(f"{'Run':<35} {'Success%':>10} {'Tokens':>10} {'Trie Nodes':>12} {'Reuse%':>10} {'Time(s)':>10} {'Memory(MB)':>12}")
    print("-" * 100)
    
    for i, name in enumerate(names):
        print(f"{name[:34]:<35} "
              f"{success_rates[i]:>10.1f} "
              f"{total_tokens[i]:>10,} "
              f"{trie_nodes[i]:>12,} "
              f"{trie_reuse[i]:>10.1f} "
              f"{total_time[i]:>10.1f} "
              f"{trie_total_mb[i]:>12.1f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_profiles.py <profile_path_pattern>")
        print()
        print("Examples:")
        print("python aggregate_profiles.py 'runs_log/*/cars-*/profile.json'")
        sys.exit(1)
    
    profile_paths = []
    for pattern in sys.argv[1:]:
        if Path(pattern).exists() and Path(pattern).is_file():
            profile_paths.append(pattern)
        else:
            matches = glob.glob(pattern, recursive=True)
            profile_paths.extend(matches)
    
    profile_paths = list(set(profile_paths))
    
    if len(profile_paths) == 0:
        print(f"No profile files found matching: {sys.argv[1:]}")
        print()
        print("Try:")
        print("  python aggregate_profiles.py 'runs_log/*/cars-*/profile.json'")
        sys.exit(1)
    
    aggregate_profiles(profile_paths)


if __name__ == "__main__":
    main()