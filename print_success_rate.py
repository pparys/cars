import cars, sys, scipy.stats
from distr_utils import *
import matplotlib.pyplot as plt

def print_success_rates(dir : str):
    r = get_success_rates(dir)
    for x,y in r:
        print(f"{x}/{y}", end=" ")
    return [y/x if x>0 else float('inf') for x,y in r]


def print_success_rate_all(model):
    res = {}
    datasets =  ["BV4", "SLIA"]
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
        import matplotlib.pyplot as plt
        
        STYLES = list(map(str.upper, styles))

        dane = [100*res[d,s] for s in styles]
        dane_no_inf = [10000 if r==float('inf') else r for r in dane]
        colors = ['red', 'blue', 'green', 'orange']
        
        bars = plt.bar(STYLES, dane_no_inf, color=colors, edgecolor='black')
        
        plt.title(f'Num of calls needed to produce 100 samples, {d}')
        plt.ylabel('Number of calls')
        plt.xlabel('Algorithm')

        for bar,v in zip(bars,dane):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f"{v:.0f}", ha='center', va='bottom')

        #plt.show()        
        
        output_dir = "plots/needed"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"needed-bars-{d}.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    print_success_rate_all(sys.argv[1] if len(sys.argv)==2 else "")
