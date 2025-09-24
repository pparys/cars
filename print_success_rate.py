import cars, sys, scipy.stats, os, statistics, numpy
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
        res[d] = []
    
    for task, dir in get_all_task_dirs():
        if not dir.endswith(model):
            continue
        print(f"{task}-{dir[-1]} --> ", end="")
        results = {s: [] for s in styles}
        for style, subdir in get_all_style_dirs(dir):
            if style in styles:
                print(f"{style}: ", end = "")
                r = print_success_rates(subdir)
                results[style] += r
        #        for d in datasets:
        #            if task.startswith(d):
        #                res[d, style] += r
        print()
        for d in datasets:
            if task.startswith(d):
                med_res = [100*statistics.median(results[s]) for s in styles]
                #med_res = [10000 if d>10000 else d for d in med_res]
                #print(med_res)
                res[d].append(med_res)
    
    print(res)

    for d in datasets:
        datax = res[d]
        data = [[10000 if d>10000 else d for d in dd] for dd in datax]

        STYLES = list(map(str.upper, styles))
        bar_colors = [colors.get(style, '#333333') for style in STYLES]

        num_bars = len(data)

        num_series = 4

        # Odległość między grupami
        odstep = 0.1

        # Ustawiamy szerokość słupków
        width = (1.0 - odstep) / num_series

        # Tworzymy wykres
        fig, ax = plt.subplots()

        # Tworzymy wykres słupkowy
        ind = numpy.arange(num_bars)  # Pozycje dla grupy słupków

        # Dodajemy słupki dla każdej serii
        for i in range(num_series):
            serie = [data[j][i] for j in range(num_bars)]  # Zbieramy dane dla i-tej serii
            ax.bar(ind + i * width, serie, width, color=bar_colors[i])

        # Ustawiamy etykiety osi X i Y
        ax.set_ylabel('Number of Calls')
        ax.set_xlabel('Test Case')
        ax.set_title(f'Number of Calls Needed to Produce 100 Samples\n{d} Dataset', pad=10)

        # Ustawiamy etykiety na osi X (Grupy 1, 2, 3, 4)
        ax.xaxis.set_ticks_position('none')
        ax.set_xticks(ind + width * (num_series - 1) / 2)
        ax.set_xticklabels([f'{i+1}' for i in range(num_bars)])

        # Dodajemy legendę
        ax.legend(STYLES, loc='best')

        y_max = max(max(l) for l in data) * 1.1
        ax.set_ylim(0, y_max)
        
        ax.grid(True, alpha=0.3, axis='y')

        # Wyświetlamy wykres
        plt.tight_layout()
        output_dir = "plots/needed"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"needed-bars-{d}.png"), dpi=300, 
                   bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"{d}: ")
        for i in range(4):
            onlyi = [d[i] for d in datax]
            print(STYLES[i], scipy.stats.gmean(onlyi))
        

if __name__ == "__main__":
    print_success_rate_all(sys.argv[1] if len(sys.argv)==2 else "")