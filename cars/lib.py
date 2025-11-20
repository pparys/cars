import os
import json
import gc, math
import time

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from llguidance_grammar_recognizer import LlguidanceTokenRecognizer
from cars.logits_process import GrammarAlignedOracleLogitsProcessor

def scores_to_top_k_tokens(scores, k):
    result = []
    for step_i, step_scores in enumerate(scores):
        probs = torch.log_softmax(step_scores, dim=-1)
        top_probs, top_token_ids = torch.topk(probs, k=k)
        top_probs = top_probs.tolist()
        top_choices = list(zip(top_token_ids, top_probs))
        result.append(top_choices)
    return result

class ConstrainedModel():
    HF_CHAT_MODELS = [
        # Llama
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        # Qwen
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        # DeepSeek
        # "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        # Microsoft
        # "microsoft/Phi-4-mini-instruct",
        "microsoft/Phi-3.5-mini-instruct",
        # Google
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        # Mistral
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.1",

    ]
    HF_BASE_MODELS = [
        "hsultanbey/codegen350multi_finetuned"
    ]

    def __init__(self, model_id: str, grammar_str: str | None = None, profiler=None, **kwargs):
        with open("secrets.json") as f:
            secrets = json.load(f)
            if secrets["HF_TOKEN"] != 'your_token':
                os.environ["HF_TOKEN"] = secrets["HF_TOKEN"]

        self.model_id = model_id
        self.profiler = profiler

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Tokenizer: {self.tokenizer.name_or_path}")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device_map, 
            **kwargs
        )
        print(f"Model: {self.model.name_or_path}")
        self.model.eval()
        if (model_id == "hsultanbey/codegen350multi_finetuned"):
            self.model.resize_token_embeddings(len(self.tokenizer)) #PP: added

        print(f"Model device: {self.model.device}")
        if grammar_str is not None:
            self._set_grammar_constraint(grammar_str)


    def reset_sampling(self, learn_level : int = 3, constrain_first : bool = False):
        self.gcd_logits_processor = GrammarAlignedOracleLogitsProcessor(
            self.tokenizer, 
            self.grammar_constraint, 
            self.model.device,
            learn_level = learn_level, 
            constrain_first = constrain_first,
            profiler = self.profiler 
        )


    def _set_grammar_constraint(self, grammar_str: str):
        self.grammar_constraint = LlguidanceTokenRecognizer(grammar_str, self.tokenizer)
        
    def _format_prompt(self, prompt: str) -> str:
        """
        Formats the prompt accordingly if it is a chat model.
        """
        if self.model_id in self.HF_BASE_MODELS:
            return prompt
        elif self.model_id in self.HF_CHAT_MODELS:
            messages = [
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            assert type(formatted_prompt) == str
            return formatted_prompt
        else:
            raise ValueError(f"Unknown model type for model {self.model_id}")

    def _unbatch_sequences(self, sequences: torch.Tensor) -> list[torch.Tensor]:
        """
        Unbatch the sequences and return them as a list of tensors.
        """
        result = []
        for i in range(sequences.shape[0]):
            # Remove padding from the sequence
            # Find the index of first eos token, keep the first eos token and remove everything after
            # If no eos token, keep the whole sequence
            eos_mask = sequences[i] == self.tokenizer.eos_token_id
            eos_idx = torch.nonzero(eos_mask)
            if eos_idx.shape[0] > 0:
                eos_idx = eos_idx[0].item()
                result.append(sequences[i][:eos_idx + 1])
            else:
                result.append(sequences[i])
        return result

    def _generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=None,
        )

        self.gcd_logits_processor.reset()
        logits_processor_list = LogitsProcessorList([self.gcd_logits_processor, InfNanRemoveLogitsProcessor()])

        # Profile inference time
        inference_start = time.time()
        
        output = self.model.generate(
            input_ids,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
            logits_processor=logits_processor_list,
        )
        
        if self.profiler:
            self.profiler.record_inference_time(time.time() - inference_start)
        
        output_ids = output.sequences
        raw_logprob = self.gcd_logits_processor.generation_ended(output_ids)
        
        output_ids = output_ids[:, input_ids.shape[1]:]
        output_scores = torch.stack(output.scores, dim=1)
        # Check that the length of the output and the scores match
        assert output_ids.shape[1] == output_scores.shape[1]

        return output_ids, output_scores, raw_logprob.item()


    def _get_seq_logprob_from_scores(self, scores: torch.Tensor, query_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of the sequences in `query_ids` given the `scores`.
        `scores` has shape (batch_size, seq_len, vocab_size).
        `query_ids` has shape (batch_size, seq_len).
        Result has shape (batch_size,).
        """

        assert scores.shape[0] == query_ids.shape[0], "Batch sizes must match"
        assert scores.shape[1] == query_ids.shape[1], "Sequence lengths must match"
    
        # Apply log_softmax to get log-probabilities
        logprobs = torch.log_softmax(scores, dim=-1)
        #logprobsOK = torch.nn.functional.log_softmax(scores.to(torch.get_default_dtype()), dim=-1)
        #print("DTYPE:", torch.get_default_dtype())
    
        batch_size, seq_len = query_ids.shape
    
        # Initialize result tensor
        result = torch.zeros(batch_size, device=scores.device)
    
        # Process each sequence in the batch
        for i in range(batch_size):
            # Get logprobs for this sequence's tokens
            seq_token_logprobs = logprobs[i, torch.arange(seq_len), query_ids[i]]
            #seq_token_logprobsOK = logprobsOK[i, torch.arange(seq_len), query_ids[i]]

            # Find the first EOS token's position (if any)
            eos_mask = query_ids[i] == self.tokenizer.eos_token_id
            eos_positions = torch.nonzero(eos_mask)

            if eos_positions.shape[0] > 0:
                # Include up to and including the first EOS token
                first_eos_pos = eos_positions[0].item()
                # Sum only up to and including the first EOS token
                result[i] = seq_token_logprobs[:first_eos_pos + 1].sum()
                #resultOK[i] = seq_token_logprobsOK[:first_eos_pos + 1].sum()
            else:
                # No EOS token, sum all logprobs
                result[i] = seq_token_logprobs.sum()
                #resultOK[i] = seq_token_logprobsOK.sum()

        return result


    def _get_generation_scores(
        self,
        prompt_ids: torch.Tensor,
        query_ids: torch.Tensor,
        constrain: bool = False,
        prefix_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return the model generation scores at each step of the query using a single forward pass.
        More efficient than _get_generation_scores for longer sequences.
        """
        # If prompt_ids has batch size of 1, duplicate it to match the batch size of query_ids
        if prompt_ids.shape[0] == 1:
            prompt_ids = prompt_ids.repeat(query_ids.shape[0], 1)

        # Concatenate prompt_ids and query_ids to get the full sequence
        if prefix_ids is not None:
            input_ids = torch.cat([prompt_ids, prefix_ids, query_ids], dim=-1)
            prompt_prefix_len = prompt_ids.shape[1] + prefix_ids.shape[1]
        else:
            input_ids = torch.cat([prompt_ids, query_ids], dim=-1)
            prompt_prefix_len = prompt_ids.shape[1]
        
        # Single forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, return_dict=True)
            
        # Extract the logits for each position
        all_logits = outputs.logits
        # Get scores for each position corresponding to query_ids
        scores = all_logits[:, prompt_prefix_len-1:prompt_prefix_len+query_ids.shape[1]-1, :]
        
        # Apply grammar constraint if needed
        if constrain:
            self.grammar_constraint.reset()
            logits_processor = GrammarConstrainedLogitsProcessor(self.grammar_constraint, 0)
            modified_scores = []
            
            # Initialize with prompt_ids for the first step
            if prefix_ids is not None:
                current_ids = prefix_ids.clone()
            else:
                # Initialize current_ids, as an empty tensor with shape (batch_size, 0)
                # current_ids = torch.empty((1, 0), dtype=torch.long).to(self.model.device)
                current_ids = torch.empty((query_ids.shape[0], 0), dtype=torch.long).to(self.model.device)
            
            for i in range(query_ids.shape[1]):
                # Apply grammar constraint for this position
                current_step_logits = scores[:, i, :].clone()
                constrained_logits = logits_processor(current_ids, current_step_logits)
                modified_scores.append(constrained_logits)
                
                # Update current_ids for the next step
                current_ids = torch.cat([current_ids, query_ids[:, i:i+1]], dim=-1)
            
            # Stack along sequence dimension
            modified_scores = torch.stack(modified_scores, dim=1)
            scores = modified_scores
        
        assert scores.shape[1] == query_ids.shape[1]
        
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return scores

    def _get_seq_logprob(
        self, 
        prompt_ids: torch.Tensor,
        query_ids: torch.Tensor,
        # grammar_str: str | None = None,
        constrain: bool = False,
        prefix_ids: torch.Tensor | None = None,     
    ) -> torch.Tensor:
        """
        Get the log probability of a sequence given the model.
        """
        scores = self._get_generation_scores(prompt_ids, query_ids, constrain, prefix_ids)
        logprob = self._get_seq_logprob_from_scores(scores, query_ids)
        return logprob