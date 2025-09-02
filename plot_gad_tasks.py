from distr_utils import *

def load_gad_tasks(split:str, runs_dir:str):
    assert split in ["SLIA", "CP", "BV4"] 
    tasks_path = f"datasets/GAD-dataset/{split}.jsonl"
    tasks = []
    with open(tasks_path, "r") as f:
        for line in f:
            task = json.loads(line)
            tasks.append(task)
    # list all dirs in the runs_dir
    runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    runs = [os.path.join(runs_dir, r) for r in runs]
    run_data = []
    for i, task in enumerate(tasks):
        task_id = task["id"]
        # find all the runs corresponding to the task_id
        task_runs = [r for r in runs if r.rsplit("-", 1)[0].endswith(task_id)] # and r.rsplit("-", 1)[1]=="ars"]

        print(f"Found {len(task_runs)} runs for task {task_id}")
        if len(task_runs) == 0:
            print(f"No runs found for task {task_id}")
            continue
        run_data.append((task_runs, task_id))
    return run_data

def filter_ars_runs(run_data):
    return [([r for r in runs if r.endswith("-ars")],id) for (runs,id) in run_data]

def plot_success(split: str, run_data, output_dir: str):
    assert split in ["SLIA", "CP", "BV4"]
    run_data = filter_ars_runs(run_data)
    plot_success_rates(run_data, split, output_dir)
    plot_success_rates(run_data, split, output_dir, cut=100)
    #plot_success_rates(run_data, split, output_dir, cut=50)

def compute_kl(split: str, run_data):
    assert split in ["SLIA", "CP", "BV4"]
    for (runs, id), (ars, _) in zip(run_data, filter_ars_runs(run_data)):
        compute_kl_chi2(runs, ars, id)
"""
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
"""
    
def plot_main():
    data = [
        ("SLIA", "gad_dataset_runs/SLIA"),
        ("BV4", "gad_dataset_runs/BV4"),
        ("CP", "gad_dataset_runs/CP"),
    ]
    output_dir = "plots_new"

    for split, runs_dir in data:
        run_data = load_gad_tasks(split, runs_dir)
        plot_success(split, run_data, f"{output_dir}/success")
        compute_kl(split, run_data)

if __name__ == "__main__":
    plot_main()
