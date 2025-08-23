from distr_utils import *

def load_gad_tasks(split, subset=None):
    assert split in ["SLIA", "CP", "BV4"] 
    slia_tasks_path = f"datasets/GAD-dataset/{split}.jsonl"
    slia_tasks = []
    with open(slia_tasks_path, "r") as f:
        for line in f:
            task = json.loads(line)
            slia_tasks.append(task)
    if subset is not None:
        slia_tasks = [slia_tasks[i] for i in subset]
    return slia_tasks

def plot_kl(split: str, runs_dir: str, output_dir: str):
    assert split in ["SLIA", "CP", "BV4"]
    tasks = load_gad_tasks(split)
    # list all dirs in the runs_dir
    runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    runs = [os.path.join(runs_dir, r) for r in runs]
    for i, task in enumerate(tasks):
        print(f"Plotting task {task['id']}")
        task_id = task["id"]
        # find all the runs corresponding to the task_id
        task_runs = [r for r in runs if r.rsplit("-", 1)[0].endswith(task_id)]
        # sort the task runs to be "prefix", "priority", "restart"
        order = {"prefix": 0, "priority": 1, "restart": 2, "asap": 3}
        def sort_key(run):
            for k in order.keys():
                if k in run:
                    return order[k]
            return 3
        task_runs = sorted(task_runs, key=sort_key)

        print(f"Found {len(task_runs)} runs for task {task_id}")
        if len(task_runs) == 0:
            print(f"No runs found for task {task_id}")
            continue
        split_output_dir = os.path.join(output_dir, split)
        plot_kl_runs(task_runs, f"{i:02}-{task_id}", split_output_dir)
    
def plot_splits_kl_scatter(split: str, runs_dir: str, output_dir: str):
    assert split in ["SLIA", "CP", "BV4"]
    tasks = load_gad_tasks(split)
    # list all dirs in the runs_dir
    runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    runs = [os.path.join(runs_dir, r) for r in runs]
    run_data = []
    for i, task in enumerate(tasks):
        print(f"Plotting task {task['id']}")
        task_id = task["id"]
        # find all the runs corresponding to the task_id
        task_runs = [r for r in runs if r.rsplit("-", 1)[0].endswith(task_id)]
        # sort the task runs to be "prefix", "priority", "restart"
        order = {"prefix": 0, "priority": 1, "restart": 2}
        def sort_key(run):
            for k in order.keys():
                if k in run:
                    return order[k]
            return 3
        task_runs = sorted(task_runs, key=sort_key)

        print(f"Found {len(task_runs)} runs for task {task_id}")
        if len(task_runs) == 0:
            print(f"No runs found for task {task_id}")
            continue
        run_data.append((task_runs, task_id))

    plot_kl_scatter(run_data, split, output_dir)
    plot_kl_scatter_asap(run_data, split, output_dir)


def plot_main():
    data = [
        ("SLIA", "gad_dataset_runs/2025-05-14_SLIA"),
        ("BV4", "gad_dataset_runs/2025-05-14_BV4"),
        ("CP", "gad_dataset_runs/2025-05-14_CP"),
    ]
    output_dir = "plots"

    for split, runs_dir in data:
        plot_kl(split, runs_dir, f"{output_dir}/kl")
        plot_splits_kl_scatter(split, runs_dir, f"{output_dir}/scatter")

if __name__ == "__main__":
    plot_main()