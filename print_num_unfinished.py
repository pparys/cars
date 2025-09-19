import sys
from distr_utils import *


def print_kl_divergence_all(style : str):
    print("Number of unfinished for", style)
    for task, dir in get_all_task_dirs():
        unfin, all = get_num_unfinished(style, dir)
        e = "!!!!!!" if unfin>0 else ""
        print(f"{task} --> {unfin}/{all} {e}")


if __name__ == "__main__":
    print_kl_divergence_all(sys.argv[1] if len(sys.argv)==2 else "restart")
