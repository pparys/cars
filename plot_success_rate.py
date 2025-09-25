from distr_utils import *

def plot_main():
    #for big_task in ["BV4", "SLIA", "fuzzing", "smiles"]:
    for big_task in ["BV4", "fuzzing"]: 
        tasks = []
        for task, dir in get_all_task_dirs():
            if task.startswith(big_task) and dir.endswith("-1"):
                task = task.removeprefix(big_task).removeprefix('-').removeprefix('_').replace('-generate_sql', '').replace('-generate_xml', '').replace('-generate_json', '')
                task = task[:task.rfind('-')]
                if task in ["find_inv_bvsge_bvashr1_4bit", "find_inv_ne_bvudiv0_4bit", "find_inv_ne_bvurem1_4bit", "json", "sql", "xml"]:
                    # or dir in ["runs_log/fuzzing-sql-generate_sql-71d4ccd4-1/cars-2025-09-19_23-15-50", "runs_log/fuzzing-sql-generate_sql_with_grammar-88960849-1/cars-2025-09-21_03-38-41"]:
                    tasks.append((task, dir))
        print(tasks)
        style = "cars"
        plot_success_rates(big_task, tasks, style, "plots/success")
        plot_success_rates(big_task, tasks, style, "plots/success", cut=100)


if __name__ == "__main__":
    plot_main()
