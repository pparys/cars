import copy
import math
import torch.nn.functional as F
import torch
import logging
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
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
    def __init__(self, grammar_constraint, oracle_trie, parse_start_index=None, save_log=False):
        # Parser variables
        self.grammar_constraint = grammar_constraint
        self.batch_parsing_states = None
        self.parse_start_index = parse_start_index

        # ASAp oracle trie
        self.oracle_trie = oracle_trie

        # To start with a longer prefix in enumerative search
        self.generate_start_index = None
        self.generated_tokens = None

        # Generation Log
        self.save_log = save_log
        self.history = []
        #print("Starting logits processor")

    def adjust_scores(self, scores, device):
        """
        resolve each stack to a tensor of True/False for each token
        indicating acceptance
        """
        acceptance = self.grammar_constraint.batch_filter_vocab(
            self.batch_parsing_states, device
        )
        current_parent, _ = self.oracle_trie.search_last_parent(self.generated_tokens)
    
        if current_parent.fresh_node: 
            logger.debug("FRESH NODE - leaving original scores")
            adjusted_scores = scores
        else:	#PP now only if node is not fresh 
            logger.debug("NODE EXISTS - reading from trie")
            adjusted_scores = self.apply_oracle_adjustments(acceptance, scores, current_parent)
            adjusted_scores[~acceptance] = -math.inf  # Scores to -inf where False

        current_parent.insert_accepted_tokens(scores, acceptance)

        if self.save_log:
            self.store_detailed_history(acceptance, scores, adjusted_scores)

        return adjusted_scores

    def apply_oracle_adjustments(self, acceptance, scores, current_parent):
        """
        Multiply expected future grammarticality
        Use the normalized (and unmasked) probabiltiy

        Parameters:
        - acceptance (torch.Tensor): A characteristic vector of valid tokens
                                     used to updated only valid tokens 
        - scores (torch.Tensor): Unnormalized logits from language model
        - current_parent (TrieNode): The trie node for the current prefix
        """
        adjusted_scores = scores.clone()
        likelihoods = F.softmax(adjusted_scores, dim=-1)
        log_likelihoods = torch.log(likelihoods)

        for batch_index in range(acceptance.size(0)):
            accepted_indices = acceptance[batch_index].nonzero().squeeze(-1)

            for idx in accepted_indices:
                token_id = idx.item()
                log_likelihood = log_likelihoods[batch_index, idx].item()
                
                # Get theta (log of expected future grammaticality) for this specific token
                success_rate = current_parent.get_success_rate(token_id)

                if not isinstance(success_rate, torch.Tensor):
                    success_rate = torch.tensor(success_rate, dtype=torch.float)
                log_theta = torch.log(success_rate)
                
                # Calculate adjusted score
                adjusted_score = scores[batch_index, idx] + log_theta
                adjusted_scores[batch_index, idx] = adjusted_score

                pretty_print_floats({
                    "token_id": token_id,
                    "token": str(self.grammar_constraint.tokenizer.decode([token_id])),
                    "raw_score": scores[batch_index, idx].item(),
                    "log_likelihood": log_likelihood,
                    "success_rate": success_rate,
                    "log_theta" : log_theta
                })

        return adjusted_scores

    def process_scores(self, input_ids, scores):
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        if self.batch_parsing_states is None:
            self.batch_parsing_states = [
                copy.deepcopy(
                    self.grammar_constraint.string_recognizer.get_initial_accept_state()
                )
                for _ in range(len(input_ids))
            ]

        # assume the generation starts from the same index
        if self.generate_start_index is None:
            # the default is the end of input sequence of tokens
            self.generate_start_index = self.parse_start_index \
                if self.parse_start_index else input_ids.size(1)
        self.generated_tokens = input_ids[:, self.generate_start_index:]

        text = self.grammar_constraint.tokenizer.decode(self.generated_tokens[0], skip_special_tokens=True)
        logger.debug("Current text: \"%s\" / %s", text, self.generated_tokens[0])

        # Advance parser states
        self.batch_parsing_states = self.grammar_constraint.advance_token_ids(
            input_ids, self.batch_parsing_states, self.parse_start_index
        ) #  PP it throws exception when text nongrammatical

        adjusted_scores = self.adjust_scores(scores, scores.device)

        return adjusted_scores
    
    def check_full_string(self, input_ids): # PP: new method, should be called at the end
        self.generated_tokens = input_ids[:, self.generate_start_index:]
        text = self.grammar_constraint.tokenizer.decode(self.generated_tokens[0])
        logger.debug("Full generated text: \"%s\" / %s", text, self.generated_tokens[0])

        _, prob = self.oracle_trie.search_last_parent(self.generated_tokens) # needed to compute probability

        self.generated_tokens = input_ids[:, self.generate_start_index:]
        self.batch_parsing_states = self.grammar_constraint.advance_token_ids(
            input_ids, self.batch_parsing_states, self.parse_start_index
        ) #PP: may throw exception
        return prob

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        return self.process_scores(input_ids, scores)

    def reset(self):
        self.reset_parser()
        self.reset_history()

    def reset_parser(self):
        self.batch_parsing_states = None
        if self.grammar_constraint.is_incremental:
            self.grammar_constraint.reset()

        self.generate_start_index = None
        self.generated_tokens = None

    def reset_history(self):
        self.history = []

    def reset_trie(self):
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

    def store_detailed_history(self, acceptance, scores, adjusted_scores):
        """
        Processes and stores information for accepted tokens including their IDs, tokens,
        raw scores, and logits.

        Parameters:
        - acceptance (torch.Tensor): A boolean tensor indicating accepted tokens for each item in the batch.
        - scores (torch.Tensor): The raw scores from the model output.
        - adjusted_scores (torch.Tensor): The adjusted scores after applying expected future grammaticality.
        """
        likelihoods = F.softmax(scores, dim=-1)
        adjusted_likelihoods = F.softmax(adjusted_scores, dim=-1)

        # Initializing the list to store detailed information for each step
        batch_accepted_info = []

        for batch_index in range(acceptance.size(0)):  # Iterate over batch items
            accepted_info = []
            accepted_indices = acceptance[batch_index].nonzero().squeeze(-1)

            for idx in accepted_indices:
                token_id = idx.item()
                raw_score = scores[batch_index, idx].item()
                likelihood = likelihoods[batch_index, idx].item()
                adjusted_likelihood = adjusted_likelihoods[batch_index, idx].item()
                token = self.grammar_constraint.tokenizer.decode([token_id])

                # Store detailed information as a dictionary
                accepted_info.append({
                    "token_id": token_id,
                    "token": str(token),
                    "raw_score": raw_score,
                    "raw_likelihood": likelihood,
                    "adjusted_score": adjusted_scores[batch_index, idx].item(),
                    "adjusted_likelihood": adjusted_likelihood
                })
                pretty_print_floats(accepted_info[-1])

            batch_accepted_info.append(accepted_info)

        # Store this detailed information in the history
        self.history.append(batch_accepted_info)
