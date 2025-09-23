import pandas as pd
import numpy as np
from scipy import stats

def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for mean.
    
    Args:
        data: array-like data
        n_bootstrap: number of bootstrap samples
        confidence: confidence level (default 0.95 for 95% CI)
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    data = np.array(data)
    bootstrapped_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    mean = np.mean(data)
    alpha = 1 - confidence
    lower = np.percentile(bootstrapped_means, (alpha/2) * 100)
    upper = np.percentile(bootstrapped_means, (1 - alpha/2) * 100)
    
    return mean, lower, upper

def create_latex_table(df, metric_columns, method_order=None, format_str='{:.2f}'):
    """
    Create LaTeX table with bootstrap confidence intervals.
    
    Args:
        df: DataFrame with columns including 'method' and metric columns
        metric_columns: list of column names to include in table
        method_order: optional list specifying order of methods (default: alphabetical)
        format_str: format string for numbers (default: 2 decimal places)
    
    Returns:
        LaTeX table string
    """
    # Group by method
    grouped = df.groupby('method')
    
    # Determine method order
    if method_order is None:
        method_order = sorted(df['method'].unique())
    
    # Calculate statistics for each method and metric
    results = []
    for method in method_order:
        method_data = grouped.get_group(method) if method in grouped.groups else pd.DataFrame()
        row = {'method': method}
        
        for metric in metric_columns:
            if len(method_data) > 0 and metric in method_data.columns:
                values = method_data[metric].values
                mean, lower, upper = bootstrap_ci(values)
                row[metric] = (mean, lower, upper)
            else:
                row[metric] = (np.nan, np.nan, np.nan)
        
        results.append(row)
    
    # Build LaTeX table
    n_cols = len(metric_columns)
    
    # Header
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Method comparison with 95\\% bootstrap confidence intervals.}\n"
    latex += "\\label{tab:method_comparison}\n"
    latex += "\\small\n"
    
    # Column specification
    col_spec = "l" + "c" * n_cols
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += "\\toprule\n"
    
    # Column headers
    header = "Method & " + " & ".join(metric_columns) + " \\\\\n"
    latex += header
    latex += "\\midrule\n"
    
    # Data rows
    for row in results:
        method = row['method'].upper()
        values = []
        
        for metric in metric_columns:
            mean, lower, upper = row[metric]
            if np.isnan(mean):
                values.append("--")
            else:
                # Format: mean (CI_lower, CI_upper)
                mean_str = format_str.format(mean)
                ci_str = f"({format_str.format(lower)}, {format_str.format(upper)})"
                # Alternative compact format: mean±error
                # error = (upper - lower) / 2
                # values.append(f"{mean_str}$\\pm${format_str.format(error)}")
                values.append(f"{mean_str} {ci_str}")
        
        latex += method + " & " + " & ".join(values) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def create_compact_latex_table(df, metric_columns, method_order=None, format_str='{:.2f}', 
                                highlight_best=True):
    """
    Create compact LaTeX table with mean±std format.
    
    Args:
        df: DataFrame with columns including 'method' and metric columns
        metric_columns: list of column names to include in table
        method_order: optional list specifying order of methods
        format_str: format string for numbers
        highlight_best: whether to bold the best value per column
    
    Returns:
        LaTeX table string
    """
    grouped = df.groupby('method')
    
    if method_order is None:
        method_order = sorted(df['method'].unique())
    
    # Calculate statistics
    results = []
    for method in method_order:
        method_data = grouped.get_group(method) if method in grouped.groups else pd.DataFrame()
        row = {'method': method}
        
        for metric in metric_columns:
            if len(method_data) > 0 and metric in method_data.columns:
                values = method_data[metric].values
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0
                row[metric] = (mean, std)
            else:
                row[metric] = (np.nan, np.nan)
        
        results.append(row)
    
    # Find best values if highlighting
    best_values = {}
    if highlight_best:
        for metric in metric_columns:
            valid_means = [row[metric][0] for row in results if not np.isnan(row[metric][0])]
            if valid_means:
                best_values[metric] = max(valid_means)
    
    # Build LaTeX table
    n_cols = len(metric_columns)
    
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Method comparison (mean $\\pm$ std over 3 runs). Best values in bold.}\n"
    latex += "\\label{tab:method_comparison}\n"
    latex += "\\small\n"
    
    col_spec = "l" + "c" * n_cols
    latex += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex += "\\toprule\n"
    
    # Simplify column names for header
    header_names = [col.capitalize() for col in metric_columns]
    header = "Method & " + " & ".join(header_names) + " \\\\\n"
    latex += header
    latex += "\\midrule\n"
    
    # Data rows
    for row in results:
        method = row['method'].upper()
        values = []
        
        for metric in metric_columns:
            mean, std = row[metric]
            if np.isnan(mean):
                values.append("--")
            else:
                mean_str = format_str.format(mean)
                std_str = format_str.format(std)
                cell = f"{mean_str}$\\pm${std_str}"
                
                # Bold if best
                if highlight_best and metric in best_values and abs(mean - best_values[metric]) < 1e-6:
                    cell = f"\\textbf{{{cell}}}"
                
                values.append(cell)
        
        latex += method + " & " + " & ".join(values) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

if __name__ == "__main__":
    # Example usage
    csv_file = "experiments/smiles/qwen_25_14b_results_by_method.csv"
    
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    # Specify which metrics to include in table
    # Choose from: validity, diversity, retro_score, membership, novelty, qed, lipinski, scaffold_div
    metrics_to_show = ['validity', 'diversity', 'retro_score', 'membership']
    
    # Specify method order (optional)
    method_order = ['ars', 'cars', 'rs', 'rsft']
    
    # Generate LaTeX table with bootstrap CI
    print("=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVAL TABLE:")
    print("=" * 80)
    latex_ci = create_latex_table(df, metrics_to_show, method_order=method_order)
    print(latex_ci)
    
    print("\n" + "=" * 80)
    print("COMPACT TABLE (mean ± std):")
    print("=" * 80)
    latex_compact = create_compact_latex_table(df, metrics_to_show, method_order=method_order)
    print(latex_compact)
    
    # Save to file
    with open('experiments/smiles/latex_table_ci.tex', 'w') as f:
        f.write(latex_ci)
    
    with open('experiments/smiles/latex_table_compact.tex', 'w') as f:
        f.write(latex_compact)
    
    print("\n✓ Tables saved to experiments/smiles/")