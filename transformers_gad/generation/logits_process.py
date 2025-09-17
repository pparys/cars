import copy
import math
import torch
import logging
import time
import xgrammar
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings
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
    def __init__(self, tokenizer, grammar_constraint, device, learn_level=3, constrain_first=False):
        # Parser variables
        self.tokenizer = tokenizer
        self.grammar_constraint = grammar_constraint
        self.learn_level = learn_level
        self.constrain_first = constrain_first
        self.device = device

        # ASAp oracle trie
        self.oracle_trie = Trie()
        self.current_index = None
        self.reset()


    def reset(self):
        #self.reset_history()
        self.grammar_constraint.reset()
        self.generate_start_index = None
        self.generated_tokens = None
        self.oracle_node = self.oracle_trie.root
        self.oracle_node_depth = 0
        self.recompute_needed = False
        self.logits_process_time = 0


    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids : torch.LongTensor, scores : torch.FloatTensor) -> torch.FloatTensor:
        start_time = time.time()
        
        self._set_generated_tokens(input_ids)
        is_root = (len(self.generated_tokens) == 0)

        # Advance the parser (unless we want to sample a full incorrect sample, in level 1)
        if self.learn_level != 1:
            if not self.grammar_constraint.try_advance_token_ids(self.generated_tokens):
                self._generation_failed() # throws exception

        # Enter appropriate trie node (possibly creating it)
        if not is_root:
            assert len(self.generated_tokens) == self.oracle_node_depth + 1
            last_token = self.generated_tokens[-1].item()
            if not (last_token in self.oracle_node.children):
                logger.debug(f"Creating new trie node for token {last_token}")
                self.oracle_node.create_child(last_token)
            self.oracle_node = self.oracle_node.children[last_token]
            self.oracle_node_depth += 1

        # If this is a new oracle node (either created above, or the root during the first call),
        # we compute its data
        if self.oracle_node.raw_logprob is None:
            self.oracle_node.raw_logprob = torch.log_softmax(scores, dim = -1).to('cpu')  # <--- on CPU
            self.oracle_node.log_theta = torch.zeros(1, scores.size(1)) #, device = self.device) <--- on CPU
            adjust_scores = (is_root and self.constrain_first)
            if self.learn_level >= 3 or adjust_scores: # filtering out the "cone"
                acceptance = self.grammar_constraint.filter_vocab()
                xgrammar.apply_token_bitmask_inplace(self.oracle_node.log_theta, acceptance) # .to(self.device, non_blocking = True))
                self.recompute_needed = True
                logger.debug("Setting bitmask")
        else:
            adjust_scores = True
        
        # Adjust scores using previously computed log_theta
        if adjust_scores:
            scores = scores.clone()
            scores += self.oracle_node.log_theta.to(self.device, non_blocking = True)

        end_time = time.time()
        self.logits_process_time += end_time - start_time

        return scores


    def _set_generated_tokens(self, input_ids : torch.LongTensor):
        assert len(input_ids)==1

        if self.generate_start_index is None:
            self.generate_start_index = input_ids.size(1) # the end of input sequence of tokens
            logger.debug("PARSER STARTING")

        self.generated_tokens = input_ids[0, self.generate_start_index:]

        if logger.level <= logging.INFO:
            text = self.tokenizer.decode(self.generated_tokens)
            logger.info("Current text: \"%s\" / %s", text, self.generated_tokens)


    def _generation_failed(self):
        assert len(self.generated_tokens) == self.oracle_node_depth + 1
        if self.learn_level >= 1:
            logger.debug("Eliminating the current sample")
            self.oracle_node.log_theta[0, self.generated_tokens[-1]] = -float('inf')
            self._recompute_in_trie()
        raise ValueError(self.generated_tokens)


    def _recompute_in_trie(self):
        node = self.oracle_node
        depth = self.oracle_node_depth
        while depth > 0:
            new_log_theta = torch.log(torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum())
            depth -= 1
            node = node.parent
            logger.debug(f"log_theta of token {self.generated_tokens[depth]} decreased from {node.log_theta[0, self.generated_tokens[depth]]} to {new_log_theta}")
            node.log_theta[0, self.generated_tokens[depth]] = new_log_theta


    def generation_ended(self, input_ids : torch.LongTensor): # should be called at the end of generation
        self._set_generated_tokens(input_ids)
        assert len(self.generated_tokens) == self.oracle_node_depth + 1

        # Advance the parser
        if not self.grammar_constraint.try_advance_token_ids(self.generated_tokens):
            self._generation_failed() # throws exception
        
        if self.generated_tokens[-1] != self.tokenizer.eos_token_id:
            logger.debug(f"No EOS at the end {self.generated_tokens[-1]} {self.tokenizer.eos_token}")
            if not self.grammar_constraint.ll_matcher.is_accepting():
                self._generation_failed() # throws exception
        
        if self.recompute_needed:
            self._recompute_in_trie()

        return self.get_logprob()
    

    def get_logprob(self):
        assert len(self.generated_tokens) == self.oracle_node_depth + 1
        
        list = []
        node = self.oracle_node
        depth = self.oracle_node_depth
        while depth >= 0:
            list.append(node.raw_logprob[0, self.generated_tokens[depth]])
            depth -= 1
            node = node.parent
        logprob = torch.tensor(list).flip(0).sum()
        return logprob


#    def reset_trie(self): # possibly useful, but doesn't used now (we always create a new object of the logit processor)
#        self.oracle_trie = Trie()
