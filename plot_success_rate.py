from distr_utils import *

def plot_main():
    for big_task in ["BV4", "SLIA", "fuzzing", "smiles"]:
        tasks = []
        for task, dir in get_all_task_dirs():
            if task.startswith(big_task):
                task = task.removeprefix(big_task).removeprefix('-').removeprefix('_').replace('-generate_sql', '').replace('-generate_xml', '').replace('-generate_json', '')
                tasks.append((task, dir))
        for style in ["ars", "cars"]:
            plot_success_rates(big_task, tasks, style, "plots/success")
            plot_success_rates(big_task, tasks, style, "plots/success", cut=100)


if __name__ == "__main__":
    plot_main()
