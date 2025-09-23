import sys
from distr_utils import *


def print_kl_divergence_all(main_style : str, model : str):
    print(f"KL-divergence for {main_style}, model {model}")
    for task, dir in get_all_task_dirs():
        if dir.endswith(f"-{model}"):
            #if task!="fuzzing-json-generate_json":
            #    continue
            kl, count = get_kl_divergence(main_style, dir)
            if count>0:
                print(f"{task} --> {kl:.5f}", f" ({count} samples)" if count<300 else "")


if __name__ == "__main__":
    print_kl_divergence_all(sys.argv[1] if len(sys.argv)>=2 else "cars", sys.argv[2] if len(sys.argv)>=3 else "1")
