import copy
import math
import torch.nn.functional as F
import torch
import logging
import time
import xgrammar
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings
from transformers_gad.llguidance_grammar_recognizer import LlguidanceTokenRecognizer
from transformers_gad.oracle.oracle_trie import Trie

logger = logging.getLogger(__name__)

def pretty_print_floats(d, precision=10):
    def format_value(v):
        if isinstance(v, float):
            return f"{v:.{precision}f}"
        return v

    formatted_items = [f"{k}: {format_value(v)}" for k, v in d.items()]
    logger.debug("{" + ", ".join(formatted_items) + "}")

class GrammarAlignedOracleLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, grammar_constraint, oracle_trie, device, adaptive=True, constrain_first=False):
        # Parser variables
        self.tokenizer = tokenizer
        self.grammar_constraint = grammar_constraint
        self.adaptive = adaptive
        self.constrain_first = constrain_first
        self.device = device

        # ASAp oracle trie
        self.oracle_trie = oracle_trie

        self.generate_start_index = None
        self.generated_tokens = None
        self.current_index = None

        # Generation Log
        #self.save_log = save_log
        #self.history = []
        self.logits_process_time = 0
        #print("Starting logits processor")

    def adjust_scores(self, scores):
        """
        resolve each stack to a tensor of True/False for each token
        indicating acceptance
        """
        #print("ACCEPTANCE:", self.grammar_constraint._grammar_bitmask)
        acceptance = self.grammar_constraint.filter_vocab()
        #print("ACCEPTANCE:", acceptance)
        #print("length:", len(acceptance[0]))
        #for i in acc_list:
        #    print(i, self.tokenizer.decode([i]))

        current_parent, _ = self.oracle_trie.search_last_parent(self.generated_tokens)

        if self.constrain_first and (current_parent == self.oracle_trie.root) and current_parent.fresh_node:
            acc_list = self.grammar_constraint.nonzero_bits()
            current_parent.insert_accepted_tokens(scores, acc_list)
    
        #print("Scores:")
        #print(scores)
        adjusted_scores = scores
        if current_parent is None:
            assert not self.adaptive
        elif current_parent.fresh_node: 
            logger.debug("FRESH NODE - leaving original scores")
        else:	#PP now only if node is not fresh 
            logger.debug("NODE EXISTS - reading from trie")
            adjusted_scores = self.apply_oracle_adjustments(scores, current_parent)
            xgrammar.apply_token_bitmask_inplace(adjusted_scores, acceptance.to(self.device, non_blocking=True)) # Scores to -inf where False
            #print("Adjusted scores:")
            #print(adjusted_scores)
            #for a in adjusted_scores[0]:
            #    if a.item()!=float('-inf'):
            #        print(a, a.item())

        if self.adaptive and current_parent.fresh_node:
            acc_list = self.grammar_constraint.nonzero_bits()
            current_parent.insert_accepted_tokens(scores, acc_list)

        #if self.save_log:
        #    self.store_detailed_history(acceptance, scores, adjusted_scores)

        return adjusted_scores

    def apply_oracle_adjustments(self, scores, current_parent):
        """
        Multiply expected future grammarticality
        Use the normalized (and unmasked) probabiltiy

        Parameters:
        - scores (torch.Tensor): Unnormalized logits from language model
        - current_parent (TrieNode): The trie node for the current prefix
        """
        adjusted_scores = scores.clone()
        #log_likelihoods = F.log_softmax(adjusted_scores, dim=-1)

        for token_id, child in current_parent.children.items():
            #log_likelihood = log_likelihoods[0, idx].item()
            
            # Get theta (log of expected future grammaticality) for this specific token
            success_rate = child.success_rate

            if not isinstance(success_rate, torch.Tensor):
                success_rate = torch.tensor(success_rate, dtype=torch.float)
            log_theta = torch.log(success_rate)
            
            # Calculate adjusted score
            adjusted_score = scores[0, token_id] + log_theta
            adjusted_scores[0, token_id] = adjusted_score

            pretty_print_floats({
                "token_id": token_id,
                "token": str(self.tokenizer.decode([token_id])),
                "raw_score": scores[0, token_id].item(),
                "success_rate": success_rate,
                "log_theta" : log_theta,
                "adjusted_score": adjusted_score,
            })

        return adjusted_scores

    def process_scores(self, input_ids, scores):
        start_time = time.time()
        assert len(input_ids)==1

        if self.generate_start_index is None:
            self.generate_start_index = input_ids.size(1) # the end of input sequence of tokens
            logger.debug("PARSER STARTING")

        self.generated_tokens = input_ids[0, self.generate_start_index:]

        text = self.tokenizer.decode(self.generated_tokens) #, skip_special_tokens=True)
        logger.info("Current text: \"%s\" / %s", text, self.generated_tokens)

        # Advance parser states
        self.grammar_constraint.advance_token_ids(self.generated_tokens) #  PP it throws exception when text nongrammatical

        adjusted_scores = self.adjust_scores(scores)
        end_time = time.time()
        self.logits_process_time += end_time - start_time

        return adjusted_scores
    
    def check_full_string(self, input_ids): # PP: new method, should be called at the end
        self.generated_tokens = input_ids[0, self.generate_start_index:]
        text = self.tokenizer.decode(self.generated_tokens)
        logger.debug("Full generated text: \"%s\" / %s", text, self.generated_tokens)

        _, prob = self.oracle_trie.search_last_parent(self.generated_tokens) # needed to compute probability

        self.generated_tokens = input_ids[:, self.generate_start_index:]
        self.batch_parsing_states = self.grammar_constraint.advance_token_ids(
            input_ids, self.batch_parsing_states, self.generate_start_index
        ) #PP: may throw exception
        return prob

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_scores(input_ids, scores)

    def reset(self):
        #self.reset_history()
        self.grammar_constraint.reset()
        self.generate_start_index = None
        self.generated_tokens = None

    #def reset_history(self):
    #    self.history = []

    def reset_trie(self): # possibly useful, but doesn't used now (we always create a new object of the logit processor)
        self.oracle_trie = Trie()

    def get_accepted_tokens(self, acceptance):
        """
        Get the indices of accepted tokens and their corresponding string values for each item in the batch.

        Parameters:
        - acceptance (torch.Tensor): A boolean tensor indicating accepted tokens for each item in the batch.
        """
        batch_size, _ = acceptance.shape
        acceptance_np = acceptance.cpu().numpy()
        accepted_x, accepted_y = acceptance_np.nonzero()

        # Initialize the dictionary with empty lists for indices
        accepted_token_indices = {i: [] for i in range(batch_size)}
        for x, y in zip(accepted_x, accepted_y):
            accepted_token_indices[x].append(y)

        # Convert token IDs to tokens
        accepted_tokens = {
            i: [self.grammar_constraint.tokenizer.decode([token_id]) for token_id in token_ids]
            for i, token_ids in accepted_token_indices.items()
        }

        return accepted_tokens

#    def store_detailed_history(self, acceptance, scores, adjusted_scores):
#        """
#        Processes and stores information for accepted tokens including their IDs, tokens,
#        raw scores, and logits.
#
#        Parameters:
#        - acceptance (torch.Tensor): A boolean tensor indicating accepted tokens for each item in the batch.
#        - scores (torch.Tensor): The raw scores from the model output.
#        - adjusted_scores (torch.Tensor): The adjusted scores after applying expected future grammaticality.
#        """
#        likelihoods = F.softmax(scores, dim=-1)
#        adjusted_likelihoods = F.softmax(adjusted_scores, dim=-1)
#
#        # Initializing the list to store detailed information for each step
#        batch_accepted_info = []
#
#        for batch_index in range(acceptance.size(0)):  # Iterate over batch items
#            accepted_info = []
#            accepted_indices = acceptance[batch_index].nonzero().squeeze(-1)
#
#            for idx in accepted_indices:
#                token_id = idx.item()
#                raw_score = scores[batch_index, idx].item()
#                likelihood = likelihoods[batch_index, idx].item()
#                adjusted_likelihood = adjusted_likelihoods[batch_index, idx].item()
#                token = self.grammar_constraint.tokenizer.decode([token_id])
#
#                # Store detailed information as a dictionary
#                accepted_info.append({
#                    "token_id": token_id,
#                    "token": str(token),
#                    "raw_score": raw_score,
#                    "raw_likelihood": likelihood,
#                    "adjusted_score": adjusted_scores[batch_index, idx].item(),
#                    "adjusted_likelihood": adjusted_likelihood
#                })
#                pretty_print_floats(accepted_info[-1])
#
#            batch_accepted_info.append(accepted_info)
#
#        # Store this detailed information in the history
#        self.history.append(batch_accepted_info)
