import hashlib, os, re, sys, torch, utils
import ars, lib_ars, lib_mcmc, mcmc

def determine_out_name(s1, s2):
    s1 = s1.removesuffix(".ebnf").removesuffix(".lark")
    s2 = s2.removesuffix(".txt")
    pref = []
    for a, b in zip(s1.split('/'), s2.split('/')):
        if a == b:
            pref.append(a)
        else:
            break
    pref = '/'.join(pref)
    s1 = s1.removeprefix(pref).removeprefix("/")
    s2 = s2.removeprefix(pref).removeprefix("/")
    pref = pref.removeprefix("datasets").removeprefix('/')
    pref = re.sub(r'[^a-zA-Z0-9]', '_', pref)
    s1 = re.sub(r'[^a-zA-Z0-9]', '_', s1)
    s2 = re.sub(r'[^a-zA-Z0-9]', '_', s2)
    res = "-".join(n for n in [pref, s1, s2] if n)
    return res


def pair_hash(s1, s2):
    s = s1+"&?!@&"+s2
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:8]


def run_task(grammar_file, prompt_file, sample_style):
    print(f"Loading grammar from file {grammar_file}")
    with open(grammar_file, "r") as f:
        grammar = f.read()

    print(f"Loading prompt from file {prompt_file}")
    with open(prompt_file, "r") as f:
        prompt = f.read()

    root_log_dir = "runs_log"
    log_dir = f"{root_log_dir}/{determine_out_name(grammar_file, prompt_file)}-{pair_hash(grammar, prompt)}/{sample_style}-{utils.timestamp()}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Saving results in folder {log_dir}")

    n_steps = 1000
    max_new_tokens = 512
    
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    if not torch.cuda.is_available():
        model_id = "hsultanbey/codegen350multi_finetuned"
        n_steps = 30
        max_new_tokens = 50
    
    if sample_style in ars.all_sample_styles():
        model = lib_ars.ConstrainedModel(model_id, grammar, torch_dtype=torch.bfloat16)
        runner = ars.ARS(model = model, prompt = prompt, sample_style = sample_style, log_dir = log_dir)
        runner.get_samples(n_samples = 1, n_steps = n_steps, max_new_tokens = max_new_tokens)

    elif sample_style in mcmc.all_sample_styles():
        n_samples = 10
        n_steps = 11
        model = lib_mcmc.ConstrainedModel(model_id, grammar, torch_dtype=torch.bfloat16)
        runner = mcmc.MCMC(model = model, prompt = prompt, propose_style = sample_style, log_dir = log_dir)
        runner.get_samples(n_samples = n_samples, n_steps = n_steps, max_new_tokens = max_new_tokens)
    
    else:
        print("Unknown sample style ", sample_style)
        print("Available styles are:", ars.all_sample_styles() + mcmc.all_sample_styles())
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Arguments: grammar_file prompt_file sample_style")
        sys.exit(1)
    
    run_task(sys.argv[1], sys.argv[2], sys.argv[3])
