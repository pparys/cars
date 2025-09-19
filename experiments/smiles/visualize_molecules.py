import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import configparser

from matplotlib.patches import Rectangle
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
	'font.size': 11,
	'font.family': 'serif',
	'axes.labelsize': 12,
	'axes.titlesize': 13,
	'xtick.labelsize': 10,
	'ytick.labelsize': 10,
	'legend.fontsize': 10,
	'figure.titlesize': 14,
	'axes.spines.top': False,
	'axes.spines.right': False,
	'axes.linewidth': 0.8,
	'grid.alpha': 0.3,
	'grid.linewidth': 0.5
})

def load_and_process_data(csv_path):
	"""Load and process the molecular synthesis results."""
	df = pd.read_csv(csv_path)
	
	# Extract method from style column (format like "ars-2025-09-17_18-34-54")
	df['method'] = df['style'].str.split('-').str[0]
	
	# Calculate efficiency metric
	df['efficiency'] = df['valid_samples'] / df['total_samples']
	
	return df

def create_performance_radar(df, save_path=None):
	"""
	Create radar charts for each dataset comparing methods.
	"""
	# Aggregate by method and dataset
	grouped = df.groupby(['dataset', 'method']).agg({
		'validity': 'mean',
		'diversity': 'mean', 
		'retro_score': 'mean',
		'membership': 'mean'
	}).reset_index()
	
	datasets = sorted(df['dataset'].unique())
	
	method_order = ['rs', 'rsft', 'mcmc', 'ars', 'cars']
	available_methods = df['method'].unique()
	ordered_methods = [m for m in method_order if m in available_methods]
	
	# Set up the radar chart
	metrics = ['Validity', 'Diversity', 'Retro Score', 'Membership']
	angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
	angles += angles[:1]  # Complete the circle
	
	# Create subplots for each dataset
	n_datasets = len(datasets)
	fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 6), 
							subplot_kw=dict(projection='polar'))
	
	if n_datasets == 1:
		axes = [axes]
	
	colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
	# colors = ['#FF6B6B', '#4ECDC4', '#96CEB4', '#FFEAA7']
	method_colors = {method: colors[i] for i, method in enumerate(ordered_methods)}
	
	for dataset_idx, dataset in enumerate(datasets):
		ax = axes[dataset_idx]
		dataset_data = grouped[grouped['dataset'] == dataset]
		
		for method in ordered_methods:
			method_data = dataset_data[dataset_data['method'] == method]
			if len(method_data) == 0:
				continue
				
			method_row = method_data.iloc[0]
			values = [
				method_row['validity'],
				method_row['diversity'], 
				method_row['retro_score'],
				method_row['membership']
			]
			values += values[:1]
			
			ax.plot(angles, values, 'o-', linewidth=2.5, label=method.upper(), 
					color=method_colors[method], markersize=6)
			ax.fill(angles, values, alpha=0.15, color=method_colors[method])
		
		ax.set_xticks(angles[:-1])
		ax.set_xticklabels(metrics, fontsize=11)
		ax.set_ylim(0, 1)
		ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
		ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
		ax.grid(True, alpha=0.3)
		
		# Add title for each dataset
		ax.set_title(f'Molecular Synthesis - {dataset.title()}', fontsize=12, pad=20)
		
		# Add legend only to the last subplot
		if dataset_idx == n_datasets - 1:
			ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
	
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()

def create_efficiency_analysis(df, save_path=None):
	"""
	Create efficiency vs quality trade-off analysis.
	"""
	# Aggregate by method and dataset
	grouped = df.groupby(['dataset', 'method']).agg({
		'efficiency': 'mean',
		'validity': 'mean',
		'diversity': 'mean',
		'retro_score': 'mean',
		'membership': 'mean'
	}).reset_index()
	
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
	
	method_order = ['rs', 'rsft', 'mcmc', 'ars', 'cars']
	available_methods = grouped['method'].unique()
	ordered_methods = [m for m in method_order if m in available_methods]
	
	# Colors for different methods (ordered)
	colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
	# colors = ['#FF6B6B', '#4ECDC4', '#96CEB4', '#FFEAA7']
	method_colors = {method: colors[i] for i, method in enumerate(ordered_methods)}
	
	# Plot 1: Efficiency vs Validity
	for method in ordered_methods:
		method_data = grouped[grouped['method'] == method]
		if len(method_data) == 0:
			continue
		ax1.scatter(method_data['efficiency'], method_data['validity'], 
				   label=method.upper(), alpha=0.7, s=80, 
				   color=method_colors[method], edgecolors='white', linewidth=1)
		
		# Add trend line
		if len(method_data) > 1:
			z = np.polyfit(method_data['efficiency'], method_data['validity'], 1)
			p = np.poly1d(z)
			x_line = np.linspace(method_data['efficiency'].min(), method_data['efficiency'].max(), 100)
			ax1.plot(x_line, p(x_line), '--', color=method_colors[method], alpha=0.6, linewidth=1.5)
	
	ax1.set_xlabel('Sampling Efficiency')
	ax1.set_ylabel('Validity')
	ax1.set_title('Efficiency vs Validity Trade-off')
	ax1.grid(True, alpha=0.3)
	ax1.legend()
	
	# Plot 2: Efficiency vs Diversity  
	for method in ordered_methods:
		method_data = grouped[grouped['method'] == method]
		if len(method_data) == 0:
			continue
		ax2.scatter(method_data['efficiency'], method_data['diversity'],
				   label=method.upper(), alpha=0.7, s=80,
				   color=method_colors[method], edgecolors='white', linewidth=1)
		
		# Add trend line
		if len(method_data) > 1:
			z = np.polyfit(method_data['efficiency'], method_data['diversity'], 1)
			p = np.poly1d(z)
			x_line = np.linspace(method_data['efficiency'].min(), method_data['efficiency'].max(), 100)
			ax2.plot(x_line, p(x_line), '--', color=method_colors[method], alpha=0.6, linewidth=1.5)
	
	ax2.set_xlabel('Sampling Efficiency') 
	ax2.set_ylabel('Diversity')
	ax2.set_title('Efficiency vs Diversity Trade-off')
	ax2.grid(True, alpha=0.3)
	
	# Plot 3: Efficiency vs Retro Score
	for method in ordered_methods:
		method_data = grouped[grouped['method'] == method]
		if len(method_data) == 0:
			continue
		ax3.scatter(method_data['efficiency'], method_data['retro_score'],
				   label=method.upper(), alpha=0.7, s=80,
				   color=method_colors[method], edgecolors='white', linewidth=1)
		
		# Add trend line
		if len(method_data) > 1:
			z = np.polyfit(method_data['efficiency'], method_data['retro_score'], 1)
			p = np.poly1d(z)
			x_line = np.linspace(method_data['efficiency'].min(), method_data['efficiency'].max(), 100)
			ax3.plot(x_line, p(x_line), '--', color=method_colors[method], alpha=0.6, linewidth=1.5)
	
	ax3.set_xlabel('Sampling Efficiency')
	ax3.set_ylabel('Retro Score') 
	ax3.set_title('Efficiency vs Retro Score Trade-off')
	ax3.grid(True, alpha=0.3)
	
	# Plot 4: Efficiency vs Membership
	for method in ordered_methods:
		method_data = grouped[grouped['method'] == method]
		if len(method_data) == 0:
			continue
		ax4.scatter(method_data['efficiency'], method_data['membership'],
				   label=method.upper(), alpha=0.7, s=80,
				   color=method_colors[method], edgecolors='white', linewidth=1)
		
		# Add trend line
		if len(method_data) > 1:
			z = np.polyfit(method_data['efficiency'], method_data['membership'], 1)
			p = np.poly1d(z)
			x_line = np.linspace(method_data['efficiency'].min(), method_data['efficiency'].max(), 100)
			ax4.plot(x_line, p(x_line), '--', color=method_colors[method], alpha=0.6, linewidth=1.5)
	
	ax4.set_xlabel('Sampling Efficiency')
	ax4.set_ylabel('Membership')
	ax4.set_title('Efficiency vs Membership Trade-off')
	ax4.grid(True, alpha=0.3)
	
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()

def create_dataset_comparison_heatmap(df, save_path=None):
	"""
	Create a heatmap showing performance across datasets.
	"""
	pivot_data = df.groupby(['dataset', 'method']).agg({
		'validity': 'mean',
		'diversity': 'mean', 
		'retro_score': 'mean',
		'membership': 'mean',
		'efficiency': 'mean'
	}).reset_index()
	
	method_order = ['rs', 'rsft', 'mcmc', 'ars', 'cars']
	# method_order = ['rs', 'rsft', 'ars', 'cars']
	available_methods = pivot_data['method'].unique()
	ordered_methods = [m for m in method_order if m in available_methods]
	
	fig, axes = plt.subplots(1, 5, figsize=(20, 6))
	metrics = ['validity', 'diversity', 'retro_score', 'membership', 'efficiency']
	titles = ['Validity', 'Diversity', 'Retro Score', 'Membership', 'Efficiency']
	
	for i, (metric, title) in enumerate(zip(metrics, titles)):
		pivot_matrix = pivot_data.pivot(index='dataset', columns='method', values=metric)
		
		available_ordered_methods = [m for m in ordered_methods if m in pivot_matrix.columns]
		pivot_matrix = pivot_matrix[available_ordered_methods]
		
		im = axes[i].imshow(pivot_matrix.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
		
		axes[i].set_xticks(range(len(pivot_matrix.columns)))
		axes[i].set_xticklabels([col.upper() for col in pivot_matrix.columns], rotation=45)
		axes[i].set_yticks(range(len(pivot_matrix.index)))
		axes[i].set_yticklabels(pivot_matrix.index)
		
		for j in range(len(pivot_matrix.index)):
			for k in range(len(pivot_matrix.columns)):
				value = pivot_matrix.iloc[j, k]
				if not pd.isna(value):
					axes[i].text(k, j, f'{value:.2f}', ha='center', va='center',
							   color='white' if value < 0.5 else 'black', fontsize=9)
		
		axes[i].set_title(title, fontsize=12)
		
		if i == len(metrics) - 1:
			cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
			cbar.set_label('Score', rotation=270, labelpad=15)
	
	plt.suptitle('Cross-Dataset Performance Analysis', fontsize=14, y=1.02)
	plt.tight_layout()
	
	if save_path:
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.show()

def generate_all_visualizations(csv_path, target_dir):
	"""Generate all ICLR-quality visualizations."""
	print("Loading and processing data...")
	df = load_and_process_data(csv_path)
	
	print("Creating Figure 1: Dataset-specific Multi-dimensional Performance Comparison...")
	create_performance_radar(df, f'{target_dir}/dataset_performance_radars.png')
	
	print("Creating Figure 2: Efficiency-Quality Trade-offs...")  
	create_efficiency_analysis(df, f'{target_dir}/efficiency_analysis.png')
	
	print("Creating Figure 3: Cross-Dataset Performance Analysis...")
	create_dataset_comparison_heatmap(df, f'{target_dir}/dataset_heatmap.png')
	
	print("All visualizations generated successfully!")
	
	# Print summary statistics
	print("\n" + "="*60)
	print("DATASET SUMMARY STATISTICS")
	print("="*60)
	
	summary = df.groupby(['dataset', 'method']).agg({
		'validity': ['mean', 'std'],
		'diversity': ['mean', 'std'],
		'retro_score': ['mean', 'std'], 
		'membership': ['mean', 'std'],
		'efficiency': ['mean', 'std']
	}).round(3)
	
	print(summary)

if __name__ == "__main__":
    
	config = configparser.ConfigParser()
	config.read('experiments/smiles/experiment.config')

	models = config['MODELS']['models'].split(',')
 
	for model in models:
		csv_file_path = f'experiments/smiles/{model}/smiles-results.csv'
		target_dir = f'experiments/smiles/{model}'
		generate_all_visualizations(csv_file_path, target_dir)