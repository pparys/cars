import sys
from distr_utils import *
import cars


def print_if_all_data():
    for task, dir in get_all_task_dirs():
        if dir[-1]!='3':
            continue
        print(f"{task}-{dir[-1]} --> ", end="")
        wyn = {"ars": 0, "cars": 0, "rs": 0, "rsft": 0, "restart": 0}
        bad = ""
        for style, subdir in get_all_style_dirs(dir):
            if style in cars.all_sample_styles():
                res = get_success_rates(subdir)
                if len(res)==0:
                    print(f"empty {style}, ", end="")
                    bad = "bad"
                elif len(res)==0:
                    print(f"double {style}, ", end="")
                    bad = "bad"
                else:
                    x,y = res[0]
                    if x<100 and y<2000:
                        print(f"short {style}, ", end="")
                        bad = "bad"
                    else:
                        wyn[style] += 1
            elif style=="restart":
                x = len(load_runs_log_from_dir(subdir))
                if x==100:
                    wyn["restart"] += 1
                else:
                    print("bad restart, ", end="")
        print(f"rs: {wyn['rs']}, rsft: {wyn['rsft']}, ars: {wyn['ars']}, cars: {wyn['cars']}, restart: {wyn['restart']}", end = "")
        if wyn['rs']==3 and wyn['rsft']==3 and wyn['ars']==3 and wyn['cars']==3 and wyn['restart']==3: 
            print(f"  OK {bad}")
        else:
            print("  MISSING!!!")


if __name__ == "__main__":
    print_if_all_data()
