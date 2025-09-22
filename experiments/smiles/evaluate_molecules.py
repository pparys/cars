import pandas as pd
import numpy as np
import configparser
from pathlib import Path
import json
from collections import defaultdict
import os
from fuseprop.chemutils import get_mol
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, QED
from collect_molecules import collect_molecules, collect_specific_mcmc_molecules
from syntheseus_retro_star_benchmark import RetroStarReactionModel
from syntheseus import Molecule

class InternalDiversity():
	def distance(self, mol1, mol2, dtype="Tanimoto"):
		assert dtype in ["Tanimoto"]
		if dtype == "Tanimoto":
			sim = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2))
			return 1 - sim
		else:
			raise NotImplementedError

	def get_diversity(self, mol_list, dtype="Tanimoto"):
		if len(mol_list) < 2:
			return 0.0
		similarity = 0
		mol_list = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in mol_list] 
		for i in range(len(mol_list)):
			sims = DataStructs.BulkTanimotoSimilarity(mol_list[i], mol_list[:i])
			similarity += sum(sims)
		n = len(mol_list)
		n_pairs = n * (n - 1) / 2
		diversity = 1 - similarity / n_pairs
		return diversity

def compute_retro_score(smile_strings, threshold=0.1):
    # return 0
	valid_molecules = []
	valid_smiles = []
	
	for smiles in smile_strings:
		try:
			rdkit_mol = Chem.MolFromSmiles(smiles)
			if rdkit_mol is not None:
				mol_obj = Molecule(smiles)
				valid_molecules.append(mol_obj)
				valid_smiles.append(smiles)
		except Exception as e:
			continue
	
	val_metric = len(valid_molecules) / len(smile_strings)
	
	if len(valid_molecules) == 0:
		return 0.0
	
	model = RetroStarReactionModel()
	reactions = model._get_reactions(valid_molecules, num_results=1)
	
	synthesizable_count = 0
	for mol_reactions in reactions:
		if len(mol_reactions) > 0 and mol_reactions[0].metadata['probability'] > threshold:
			synthesizable_count += 1
	
	orig_retro_metric = synthesizable_count / len(valid_molecules)
	return orig_retro_metric * val_metric

def compute_novelty(mol_samples, exemplar_smiles):
	"""Compute novelty: fraction of molecules not in training exemplars"""
	exemplar_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 3, 2048) 
					for s in exemplar_smiles if Chem.MolFromSmiles(s) is not None]
	
	novel_count = 0
	for mol in mol_samples:
		mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048)
		# Check if exact match with any exemplar
		is_novel = True
		for ex_fp in exemplar_fps:
			if DataStructs.TanimotoSimilarity(mol_fp, ex_fp) == 1.0:
				is_novel = False
				break
		if is_novel:
			novel_count += 1
	
	return novel_count / len(mol_samples) if mol_samples else 0.0

def compute_druglikeness_metrics(mol_samples):
	"""Compute QED and Lipinski rule of 5 compliance"""
	if not mol_samples:
		return 0.0, 0.0
	
	qed_scores = []
	lipinski_passes = 0
	
	for mol in mol_samples:
		# QED (Quantitative Estimate of Drug-likeness)
		qed_scores.append(QED.qed(mol))
		
		# Lipinski Rule of 5
		mw = Descriptors.MolWt(mol)
		logp = Crippen.MolLogP(mol)
		hbd = Lipinski.NumHDonors(mol)
		hba = Lipinski.NumHAcceptors(mol)
		
		if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
			lipinski_passes += 1
	
	avg_qed = np.mean(qed_scores)
	lipinski_ratio = lipinski_passes / len(mol_samples)
	
	return avg_qed, lipinski_ratio

def compute_scaffold_diversity(mol_samples):
	"""Compute Bemis-Murcko scaffold diversity"""
	from rdkit.Chem.Scaffolds import MurckoScaffold
	
	if not mol_samples:
		return 0.0
	
	scaffolds = set()
	for mol in mol_samples:
		try:
			scaffold = MurckoScaffold.GetScaffoldForMol(mol)
			scaffolds.add(Chem.MolToSmiles(scaffold))
		except:
			continue
	
	return len(scaffolds) / len(mol_samples)

def eval_membership(mol_sample, monomer_class):
	if monomer_class == "acrylates":
		patterns = ["C=CC(=O)O*"]
		pattern_mols = [Chem.MolFromSmarts(p) for p in patterns]
		for pattern in pattern_mols:
			if mol_sample.HasSubstructMatch(pattern):
				return True
		return False
	elif monomer_class == "chain_extenders" or monomer_class == "chain":
		patterns = ['CO', 'OC', 'N']
		pattern_mols = [Chem.MolFromSmarts(p) for p in patterns]
		for pattern in pattern_mols:
			if mol_sample.HasSubstructMatch(pattern):
				return True
		return False
	elif monomer_class == "isocyanates":
		pattern = Chem.MolFromSmarts('[*]N=C=O')
		return mol_sample.HasSubstructMatch(pattern)
	else:
		raise ValueError(f"Invalid monomer class: {monomer_class}")

def load_exemplars(monomer_class):
	"""Load exemplar molecules for novelty computation"""
	# You'll need to adjust paths based on your setup
	exemplar_files = {
		"acrylates": "experiments/smiles/exemplars/acrylates.txt",
		"chain": "experiments/smiles/exemplars/chain_extenders.txt", 
		"isocyanates": "experiments/smiles/exemplars/isocyanates.txt"
	}
	
	exemplars = []
	file = exemplar_files[monomer_class]
	with open(file) as f:
		exemplars = [line.rstrip() for line in f]
	return exemplars

def evaluate_mol(smiles_samples, monomer_class, exemplar_smiles=None):
	# print(smiles_samples)
	mol_samples = [get_mol(s) for s in smiles_samples]
	mol_samples = [x for x in mol_samples if x is not None]

	val_metric = len(mol_samples) / len(smiles_samples) if smiles_samples else 0.0

	if len(mol_samples) == 0:
		return {
			'validity': 0.0,
			'diversity': 0.0,
			'retro_score': 0.0,
			'membership': 0.0,
			'novelty': 0.0,
			'qed': 0.0,
			'lipinski': 0.0,
			'scaffold_div': 0.0
		}

	div = InternalDiversity()
	div_metric = div.get_diversity(mol_samples)
 
	retro_metric = compute_retro_score(smiles_samples, threshold=0.4)
	mem_metric = sum([eval_membership(s, monomer_class) for s in mol_samples]) / len(smiles_samples)
	
	# New metrics
	novelty_metric = compute_novelty(mol_samples, exemplar_smiles) if exemplar_smiles else 0.0
	qed_metric, lipinski_metric = compute_druglikeness_metrics(mol_samples)
	scaffold_metric = compute_scaffold_diversity(mol_samples)
	
	return {
		'validity': val_metric,
		'diversity': div_metric,
		'retro_score': retro_metric,
		'membership': mem_metric,
		'novelty': novelty_metric,
		'qed': qed_metric,
		'lipinski': lipinski_metric,
		'scaffold_div': scaffold_metric
	}

def parse_run_dir_name(run_dir_name):
	"""Parse run directory name to extract dataset and model"""
	parts = run_dir_name.split('_')
	
	# First part is dataset
	dataset = parts[0]
	
	# Rest is model name
	model = '_'.join(parts[1:])
	
	return dataset, model

if __name__ == "__main__":
	runs_base_dir = "experiments/smiles/runs"
	mcmc_dir = "experiments/smiles/mcmc"  # Adjust if needed
	
	# Organize runs by model and dataset
	runs_by_model_dataset = defaultdict(lambda: defaultdict(str))
	
	for run_dir_name in sorted(os.listdir(runs_base_dir)):
		run_path = os.path.join(runs_base_dir, run_dir_name)
		if not os.path.isdir(run_path):
			continue
		
		dataset, model = parse_run_dir_name(run_dir_name)
		runs_by_model_dataset[model][dataset] = run_path
	
	# Load exemplars for each dataset (adjust paths as needed)
	exemplars = {
		'acrylates': load_exemplars('acrylates'),
		'chain': load_exemplars('chain'),
		'isocyanates': load_exemplars('isocyanates')
	}
	
	# Process each model
	for model in sorted(runs_by_model_dataset.keys()):
		all_results = []
		
		for dataset in sorted(runs_by_model_dataset[model].keys()):
			run_dir = runs_by_model_dataset[model][dataset]
			
			# Use your collect_molecules function
			samples = collect_molecules(run_dir)
			
			# Also collect MCMC samples if available
			mcmc_samples = {}
			if os.path.exists(mcmc_dir):
				mcmc_samples = collect_specific_mcmc_molecules(mcmc_dir, dataset)
			
			# Combine regular and MCMC samples
			all_samples = {**samples, **mcmc_samples}
			
			# Map dataset names for membership evaluation
			monomer_class = dataset if dataset in ['acrylates', 'isocyanates'] else 'chain'
			exemplar_smiles = exemplars.get(dataset, [])
			
			for style, sampled_data in all_samples.items():
				molecules = sampled_data.molecules
				valid_count = sampled_data.valid_samples_count
				total_count = sampled_data.total_samples_count
				
				print(f"Evaluating {model} - {dataset} - {style} - {valid_count}/{total_count} molecules")
				
				if valid_count < 100:
					print("  Skipping due to insufficient molecules (<100)")
					metrics = {
						'validity': 0.0,
						'diversity': 0.0,
						'retro_score': 0.0,
						'membership': 0.0,
						'novelty': 0.0,
						'qed': 0.0,
						'lipinski': 0.0,
						'scaffold_div': 0.0
					}
				else:
					metrics = evaluate_mol(molecules, monomer_class, exemplar_smiles)
				
				all_results.append({
					"model": model,
					"dataset": dataset,
					"style": style,
					"valid_samples": valid_count,
					"total_samples": total_count,
					**metrics
				})
		
		if all_results:
			print()
			df = pd.DataFrame(all_results)
			df = df.sort_values(by=["dataset", "style"])
			
			# Create a copy for averaging (before formatting)
			df_numeric = df.copy()
			
			# Extract method prefix from style (e.g., "ars" from "ars-2025-09-19_23-24-29")
			df_numeric['method'] = df_numeric['style'].str.split('-').str[0]
			
			# Calculate averages by method for each dataset
			print("=" * 100)
			print(f"RESULTS FOR MODEL: {model}")
			print("=" * 100)
			print("\nDETAILED RESULTS (All Runs):")
			print("-" * 100)
			
			# Format for detailed display
			df_display = df.copy()
			df_display['validity'] = df_display['validity'].map('{:.2f}'.format)
			df_display['diversity'] = df_display['diversity'].map('{:.2f}'.format)
			df_display['retro_score'] = df_display['retro_score'].map('{:.4f}'.format)
			df_display['membership'] = df_display['membership'].map('{:.2f}'.format)
			df_display['novelty'] = df_display['novelty'].map('{:.2f}'.format)
			df_display['qed'] = df_display['qed'].map('{:.3f}'.format)
			df_display['lipinski'] = df_display['lipinski'].map('{:.2f}'.format)
			df_display['scaffold_div'] = df_display['scaffold_div'].map('{:.2f}'.format)
			print(df_display.to_string(index=False))
			
			# Average by method (style prefix)
			print("\n" + "=" * 100)
			print("AVERAGED BY METHOD (across runs):")
			print("-" * 100)
			
			metric_cols = ['validity', 'diversity', 'retro_score', 'membership', 'novelty', 'qed', 'lipinski', 'scaffold_div']
			
			df_method_avg = df_numeric.groupby(['dataset', 'method']).agg({
				'valid_samples': 'mean',
				'total_samples': 'mean',
				**{col: 'mean' for col in metric_cols}
			}).reset_index()
			
			df_method_avg['model'] = model
			df_method_avg = df_method_avg[['model', 'dataset', 'method', 'valid_samples', 'total_samples'] + metric_cols]
			df_method_avg = df_method_avg.sort_values(by=['dataset', 'method'])
			
			# Format method averages
			df_method_display = df_method_avg.copy()
			df_method_display['valid_samples'] = df_method_display['valid_samples'].map('{:.0f}'.format)
			df_method_display['total_samples'] = df_method_display['total_samples'].map('{:.0f}'.format)
			df_method_display['validity'] = df_method_display['validity'].map('{:.2f}'.format)
			df_method_display['diversity'] = df_method_display['diversity'].map('{:.2f}'.format)
			df_method_display['retro_score'] = df_method_display['retro_score'].map('{:.4f}'.format)
			df_method_display['membership'] = df_method_display['membership'].map('{:.2f}'.format)
			df_method_display['novelty'] = df_method_display['novelty'].map('{:.2f}'.format)
			df_method_display['qed'] = df_method_display['qed'].map('{:.3f}'.format)
			df_method_display['lipinski'] = df_method_display['lipinski'].map('{:.2f}'.format)
			df_method_display['scaffold_div'] = df_method_display['scaffold_div'].map('{:.2f}'.format)
			print(df_method_display.to_string(index=False))
			
			# Average by dataset (across all methods and runs)
			print("\n" + "=" * 100)
			print("AVERAGED BY DATASET (across all methods and runs):")
			print("-" * 100)
			
			df_dataset_avg = df_numeric.groupby(['method']).agg({
				'valid_samples': 'mean',
				'total_samples': 'mean',
				**{col: 'mean' for col in metric_cols}
			}).reset_index()
			
			df_dataset_avg['model'] = model
			df_dataset_avg = df_dataset_avg[['model', 'method', 'valid_samples', 'total_samples'] + metric_cols]
			df_dataset_avg = df_dataset_avg.sort_values(by=['method'])
			
			# Format dataset averages
			df_dataset_display = df_dataset_avg.copy()
			df_dataset_display['valid_samples'] = df_dataset_display['valid_samples'].map('{:.0f}'.format)
			df_dataset_display['total_samples'] = df_dataset_display['total_samples'].map('{:.0f}'.format)
			df_dataset_display['validity'] = df_dataset_display['validity'].map('{:.2f}'.format)
			df_dataset_display['diversity'] = df_dataset_display['diversity'].map('{:.2f}'.format)
			df_dataset_display['retro_score'] = df_dataset_display['retro_score'].map('{:.4f}'.format)
			df_dataset_display['membership'] = df_dataset_display['membership'].map('{:.2f}'.format)
			df_dataset_display['novelty'] = df_dataset_display['novelty'].map('{:.2f}'.format)
			df_dataset_display['qed'] = df_dataset_display['qed'].map('{:.3f}'.format)
			df_dataset_display['lipinski'] = df_dataset_display['lipinski'].map('{:.2f}'.format)
			df_dataset_display['scaffold_div'] = df_dataset_display['scaffold_div'].map('{:.2f}'.format)
			print(df_dataset_display.to_string(index=False))
			
			# Save all results
			output_dir = Path(f'experiments/smiles')
			output_dir.mkdir(parents=True, exist_ok=True)
			
			# Save detailed results
			df_display.to_csv(output_dir / f'{model}_results_detailed.csv', index=False)
			
			# Save method averages
			df_method_avg.to_csv(output_dir / f'{model}_results_by_method.csv', index=False)
			
			# Save dataset averages
			df_dataset_avg.to_csv(output_dir / f'{model}_results_by_dataset.csv', index=False)
			
			print(f"\n✓ Saved detailed results to {output_dir / f'{model}_results_detailed.csv'}")
			print(f"✓ Saved method averages to {output_dir / f'{model}_results_by_method.csv'}")
			print(f"✓ Saved dataset averages to {output_dir / f'{model}_results_by_dataset.csv'}")
			print("=" * 100 + "\n")