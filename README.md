This library realizes sampling from a language model constrained by a context-free grammar.

## Requirements

Should work at least with Python in version 3.11. It requires some basic Python packages:

```pip install torch transformers gpustat numpy accelerate llguidance xgrammar scipy matplotlib```

## Basic usage

```python run_task.py grammar_file prompt_file sample_style model```

where:
- `grammar_file` is a file containing a context-free grammar, in a format understandable by the `llguidance` library (in particular both `ebnf` or `lark` formats work well),
- `prompt_file` is a text file containing the prompt,
- `sample style` is one of `rs` (Rejection Sampling), `ars` (Adaptive Rejection Sampling), `rsft` (Rejection Sampling with constrained First Token),
  `prefix` (MCMC-Prefix), `priority` (MCMC-Priority), `restart` (MCMC-Restart),
- `model` is a number between 0 and 3.

The MCMC sampling methods are desciribed in the paper [Constrained Sampling for Language Models Should
Be Easy: An MCMC Perspective](https://arxiv.org/pdf/2506.05754)

Supported models are:
- 0: `hsultanbey/codegen350multi_finetuned` (a small model, good for local tests on a machine without a GPU)
- 1: `meta-llama/Llama-3.1-8B-Instruct`
- 2: `Qwen/Qwen2.5-7B-Instruct`
- 3: `Qwen/Qwen2.5-14B-Instruct`

The program outputs generated sequences on standard output, together with some basic information.
In a more formalized way, the generated sequences are written in a json file in the `runs_log` folder.
The subfolder name has three parts: first, based on paths to the grammar and prompts; second, being a hash of the grammar and prompt; third, being the model number.

## System environment variables

With `TCFG_LOG_LEVEL=INFO` or `TCFG_LOG_LEVEL=DEBUG`, the program outputs more information.

`HF_HOME` may contain a path to a folder in which the language models will be stored.

`CUDA_VISIBLE_DEVICES` may contain a number of GPU on which the calculations will be run.

## Additional parameters

Some more parameters can be set in the `run_task.py` file, in particular:
- `max_new_tokens` is the maximal number of tokens produced in the sampled sequence;
- `n_samples` is the number of generated sequences;
- for rs, ars, rsft, cars styles, the program stops after `n_steps` calls to the LLM (even if `n_samples` sequences are not produced);
- for MCMC styles, `n_steps` is the number of steps `k` (as described in the paper);
- it is also easy to add more models from huggingface, however for MCMC styles they should also be listed in the `mcmc/lib.py` file.
