# Constrained Adaptive Rejection Sampling

This library allows sampling from a language model constrained by a context-free grammar.

## Requirements

The library is guaranteed to work with Python 3.11; it has not been tested with other versions.
It also requires several Python packages:

```pip install torch transformers gpustat numpy accelerate llguidance xgrammar scipy matplotlib```

## Basic Usage

To run the sampling task, use the following command:

```python run_task.py grammar_file prompt_file sample_style model```

where:
- `grammar_file` is a file containing a context-free grammar in a format understandable by the `llguidance` library (both `ebnf` or `lark` formats are supported),
- `prompt_file` is a text file containing the prompt,
- `sample style` is one of the following methods:  
  `rs` (Rejection Sampling),  
  `ars` (Adaptive Rejection Sampling),  
  `rsft` (Rejection Sampling with constrained First Token),  
  `cars` (Constrained Adaptive Rejection Sampling),  
  `prefix` (MCMC-Prefix),  
  `priority` (MCMC-Priority),  
  `restart` (MCMC-Restart),
- `model` is a number between 0 and 3, specifying the model to be used.

The MCMC sampling methods are desciribed in the paper: [Constrained Sampling for Language Models Should
Be Easy: An MCMC Perspective](https://arxiv.org/pdf/2506.05754).

### Supported Models

- 0: `hsultanbey/codegen350multi_finetuned` (a small model, suitable for local testing on machines without a GPU)
- 1: `meta-llama/Llama-3.1-8B-Instruct`
- 2: `Qwen/Qwen2.5-7B-Instruct`
- 3: `Qwen/Qwen2.5-14B-Instruct`

### Output

The program outputs the generated sequences to the standard output, along with basic information.
In a more formalized way, the generated sequences are saved in a JSON file located in the `runs_log` folder.
The subfolder name consists of three parts:
1. a part derived from the grammar and prompt file locations,
2. a hash of the grammar and prompt,
3. the model number.

## System Environment Variables

The following environment variables can be used to configure the program:
- `TCFG_LOG_LEVEL`: Set to `INFO` or `DEBUG` for more detailed output.
- `HF_HOME`: Specifies the path to the folder where language models will be stored.
- `CUDA_VISIBLE_DEVICES`: Specifies the GPU number on which the calculations will be run.

## Additional Parameters

Several parameters can be set within the `run_task.py` file, including:
- `max_new_tokens`: The maximum number of tokens to generate in the sampled sequence.
- `n_samples`: The number of sequences to generate.
- For rs, ars, rsft, cars styles, the program stops after `n_steps` calls to the LLM, even if `n_samples` sequences have not been produced.
- For MCMC styles, `n_steps` represents the number of steps `k` (as described in the paper).
- It is also easy to add more models from Hugging Face. However, for MCMC styles, they must also be listed in the `mcmc/lib.py` file.

