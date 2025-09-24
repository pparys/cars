import cars, sys, scipy.stats, os
from distr_utils import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Default styling used for all plots
plt.rcParams.update({
    'font.size': 18,                
    'axes.labelsize': 20,            
    'axes.titlesize': 20,           
    'xtick.labelsize': 16,          
    'ytick.labelsize': 16,         
    'legend.fontsize': 18,          
    'figure.titlesize': 24,          

    'font.family': 'serif',
    'mathtext.fontset': 'cm',        

    'axes.linewidth': 1.5,           
    'axes.spines.top': False,        
    'axes.spines.right': False,      

    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',        
    'ytick.direction': 'in',
    'xtick.minor.visible': False,   
    'ytick.minor.visible': False,

    'legend.frameon': True,        

    'axes.grid': True,              
    'grid.linestyle': ':',           
    'grid.alpha': 0.5,              
    'axes.axisbelow': True,          

    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

colors = {
    'RS': '#8c564b',       # Brown
    'ARS': '#1f77b4',      # Blue
    'RSFT': '#9467bd',     # Purple
    'CARS': '#ff7f0e',     # Orange
}

def print_success_rates(dir : str):
    r = get_success_rates(dir)
    for x,y in r:
        print(f"{x}/{y}", end=" ")
    return [y/x if x>0 else float('inf') for x,y in r]

def print_success_rate_all(model):
    res = {}
    datasets = ["BV4", "SLIA"]
    styles = cars.all_sample_styles()
    
    for d in datasets:
        for s in styles:
            res[d, s] = []
    
    for task, dir in get_all_task_dirs():
        if not dir.endswith(model):
            continue
        print(f"{task}-{dir[-1]} --> ", end="")
        for style, subdir in get_all_style_dirs(dir):
            if style in styles:
                print(f"{style}: ", end = "")
                r = print_success_rates(subdir)
                for d in datasets:
                    if task.startswith(d):
                        res[d, style] += r
        print()
    
    print(res)
    
    for d in datasets:
        for s in styles:
            res[d,s] = scipy.stats.gmean(res[d,s])
    print(res)
    
    for d in datasets:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        STYLES = list(map(str.upper, styles))
        dane = [100*res[d,s] for s in styles]
        dane_no_inf = [10000 if r==float('inf') else r for r in dane]
        
        bar_colors = [colors.get(style, '#333333') for style in STYLES]
        
        bars = ax.bar(STYLES, dane_no_inf, color=bar_colors, 
                     edgecolor='black', linewidth=1.2, alpha=0.8)
        
        ax.set_title(f'Number of Calls Needed to Produce 100 Samples\n{d} Dataset', pad=10)
        ax.set_ylabel('Number of Calls')
        ax.set_xlabel('Algorithm')
        
        for bar, v in zip(bars, dane):
            yval = bar.get_height()
            label_text = "inf" if v == float('inf') else f"{v:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, yval + max(dane_no_inf)*0.01, 
                   label_text, ha='center', va='bottom')
        
        y_max = max(dane_no_inf) * 1.1
        ax.set_ylim(0, y_max)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_dir = "plots/needed"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"needed-bars-{d}.png"), dpi=300, 
                   bbox_inches='tight', pad_inches=0.1)
        plt.close()

if __name__ == "__main__":
    print_success_rate_all(sys.argv[1] if len(sys.argv)==2 else "")