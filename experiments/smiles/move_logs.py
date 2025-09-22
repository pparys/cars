#!/usr/bin/env python3
"""
Copy PDDL experiment directories to organized structure based on model number
"""

import os
import shutil
import glob
import re

def get_model_id(model_num):
    """Map model number to model ID"""
    if model_num == '1':
        return "llama_31_8b"
    elif model_num == '2':
        return "qwen_25_7b"
    elif model_num == '3':
        return "qwen_25_14b"
    else:
        return f"unknown_model_{model_num}"

def parse_experiment_dir(dir_name):
    # Extract the model number (last number after final dash)
    model_num = dir_name.split('-')[-1]
    print(dir_name)
   
    task = dir_name.split('-', 1)[0].split('_')[1]
    return task, model_num

def copy_experiments():
    source_pattern = "runs_log/smiles_*"
    target_base = "experiments/smiles/runs"
    
    # Find all matching directories
    experiment_dirs = glob.glob(source_pattern)
    
    if not experiment_dirs:
        print("No experiment directories found matching pattern")
        return
    
    # Create target base directory if it doesn't exist
    os.makedirs(target_base, exist_ok=True)
    
    print(f"Found {len(experiment_dirs)} experiment directories to copy")
    print("-" * 60)
    
    for source_dir in sorted(experiment_dirs):
        dir_name = os.path.basename(source_dir)
        
        # Parse the directory name
        task, model_num = parse_experiment_dir(dir_name)
        model_id = get_model_id(model_num)
        
        # Create new directory name: {domain}_task{task}_{model_id}
        new_dir_name = f"{task}_{model_id}"
        target_dir = os.path.join(target_base, new_dir_name)
        
        print(f"Copying: {dir_name}")
        print(f"Task: {task}, Model: {model_id}")
        print(f"Target: {new_dir_name}")
        
        try:
            if os.path.exists(target_dir):
                print(f"  Target exists, merging contents...")
                # Copy contents of source into existing target
                for item in os.listdir(source_dir):
                    source_item = os.path.join(source_dir, item)
                    target_item = os.path.join(target_dir, item)
                    
                    if os.path.isdir(source_item):
                        if os.path.exists(target_item):
                            # Merge subdirectories recursively
                            shutil.copytree(source_item, target_item, dirs_exist_ok=True)
                        else:
                            shutil.copytree(source_item, target_item)
                    else:
                        # Copy individual files
                        shutil.copy2(source_item, target_item)            
            else:
                shutil.copytree(source_dir, target_dir)
                print(f"Copied successfully")
        except Exception as e:
            print(f"Error copying: {e}")
        
        print()
    
    print("Copy operation completed!")
    
    # Show final structure
    print("\nFinal directory structure:")
    if os.path.exists(target_base):
        for item in sorted(os.listdir(target_base)):
            item_path = os.path.join(target_base, item)
            if os.path.isdir(item_path):
                print(f"  {item}")

if __name__ == "__main__":
    copy_experiments()