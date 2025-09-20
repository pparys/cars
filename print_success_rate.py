import cars
from distr_utils import *

def print_success_rates(dir : str):
    for x,y in get_success_rates(dir):
        print(f"{x}/{y}", end=" ")


def print_success_rate_all():
    for task, dir in get_all_task_dirs():
        print(f"{task} --> ", end="")
        for style, subdir in get_all_style_dirs(dir):
            if style in cars.all_sample_styles():
                print(f"{style}: ", end = "")
                print_success_rates(subdir)
        print()


if __name__ == "__main__":
    print_success_rate_all()
