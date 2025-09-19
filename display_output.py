import sys, json
from distr_utils import *

def display_output(subdir : str):
    samples = extract_samples([load_runs_log_from_dir(subdir)], 'tokens')
    print(samples)
    multiline = any(any('\n' in word for word in a) for a,_ in samples)

    for a,_ in samples:
        for b in a:
            if b != '<|eot_id|>':
                print(b, end="")
        print()
        if multiline:
            print("\n################")
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Required arguments: file or folder with data")
        sys.exit(1)
    
    display_output(sys.argv[1])
