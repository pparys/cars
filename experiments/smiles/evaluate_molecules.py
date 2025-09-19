import pandas as pd
import numpy as np
import configparser

from fuseprop.chemutils import get_mol
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
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
	reactions = model._get_reactions(valid_molecules, num_results=1)  # Just get top reaction
	
	synthesizable_count = 0
	for mol_reactions in reactions:
		if len(mol_reactions) > 0 and mol_reactions[0].metadata['probability'] > threshold:
			synthesizable_count += 1
	
	orig_retro_metric = synthesizable_count / len(valid_molecules)

	return orig_retro_metric * val_metric

def eval_membership(mol_sample, monomer_class):
	if monomer_class == "acrylates":
		patterns = ["C=CC(=O)O*"]
		pattern_mols = [Chem.MolFromSmarts(p) for p in patterns]
		for pattern in pattern_mols:
			if mol_sample.HasSubstructMatch(pattern):
				return True
		return False
	elif monomer_class == "chain_extenders":
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
		raise ValueError("Invalid monomer class")

def evaluate_mol(smiles_samples, monomer_class):
	mol_samples = [get_mol(s) for s in smiles_samples]
	mol_samples = [x for x in mol_samples if x is not None]

	val_metric = len(mol_samples) / len(smiles_samples)

	div = InternalDiversity()
	div_metric = div.get_diversity(mol_samples)
 
	retro_metric = compute_retro_score(smiles_samples, threshold=0.4)

	mem_metric = sum([eval_membership(s, monomer_class) for s in mol_samples]) / len(smiles_samples)
	return val_metric, div_metric, retro_metric, mem_metric

if __name__ == "__main__":
	config = configparser.ConfigParser()
	config.read('experiments/smiles/experiment.config')

	mcmc_dir = config['PATHS']['mcmc_dir']
	samples_base_dir = config['PATHS']['samples_base_dir']
	models = config['MODELS']['models'].split(',')
	datasets = config['DATASETS']['datasets'].split(',')

	samples_config = {}
 
	for model in models:
		all_results = []
		for dataset in datasets:
			samples_config[dataset] = f"{samples_base_dir}/{model}/{dataset}"
	
		for dataset, sample_dir in samples_config.items():
			samples = collect_molecules(sample_dir)
			mcmc_samples = collect_specific_mcmc_molecules(mcmc_dir, dataset)
			for style, sampled_molecules in (samples | mcmc_samples).items():
				print(f"Evaluating {dataset} - {style} - {sampled_molecules.valid_samples_count}")
				if sampled_molecules.valid_samples_count < 100:
					print("Skipping due to insufficient molecular synthesis")
					valid_metric, div_metric, retro_metric, mem_metric = 0, 0, 0, 0
				else:
					valid_metric, div_metric, retro_metric, mem_metric = evaluate_mol(sampled_molecules.molecules, monomer_class=dataset)
				all_results.append({
					"dataset": dataset,
					"style": style,
					"valid_samples": sampled_molecules.valid_samples_count,
					"total_samples": sampled_molecules.total_samples_count,
					"validity": valid_metric,
					"diversity": div_metric,
					"retro_score": retro_metric,
					"membership": mem_metric
				})

		if all_results:
			print()
			df = pd.DataFrame(all_results)
			df = df.sort_values(by=["dataset", "style"])
			df['validity'] = df['validity'].map('{:.2}'.format)
			df['diversity'] = df['diversity'].map('{:.2f}'.format)
			df['retro_score'] = df['retro_score'].map('{:.4f}'.format)
			df['membership'] = df['membership'].map('{:.2}'.format)

			print(df.to_string(index=False))
	
			df.to_csv(f'experiments/smiles/{model}/smiles-results.csv')
