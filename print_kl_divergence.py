import sys
from distr_utils import *


def print_kl_divergence_all(main_style : str):
    print("KL-divergence for", main_style)
    for task, dir in get_all_task_dirs():
        kl = get_kl_divergence(main_style, dir)
        if kl is not None:
            print(f"{task} --> {kl:.5f}")


if __name__ == "__main__":
    print_kl_divergence_all(sys.argv[1] if len(sys.argv)==2 else "cars")
