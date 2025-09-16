from distr_utils import *

def print_success_rates(dir : str):
    for file in sorted(os.listdir(dir)):
        path = f"{dir}/{file}"
        with open(path, "r") as f:
            run_data = json.load(f)
        assert len(run_data["successes"]) == 1000
        print(run_data["successes"].count(True), end=" ")
        

def success_rate():
    data_dir = "runs_log"
    for task_dir in sorted(os.listdir(data_dir)):
        name = task_dir[:task_dir.rfind('-')]
        dir = f"{data_dir}/{task_dir}"
        print(f"{name} --> ", end="")
        for method_dir in sorted(os.listdir(dir)):
            method = method_dir[:method_dir.find('-')]
            subdir = f"{dir}/{method_dir}"
            print(f"{method}: ", end = "")
            print_success_rates(subdir)
        print()

if __name__ == "__main__":
    success_rate()
