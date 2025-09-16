import sys, json

def display_output(path : str):
    with open(path, "r") as f:
        run_data = json.load(f)
    steps = run_data["steps"]
    multiline = any(any('\n' in word for word in a['tokens']) for a in steps)
    print(multiline)

    for a in steps:
        for b in a['tokens']:
            if b != '<|eot_id|>':
                print(b, end="")
        print()
        if multiline:
            print("\n################")
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Required arguments: filename")
        sys.exit(1)
    
    display_output(sys.argv[1])
