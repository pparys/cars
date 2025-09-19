import copy
import math
import torch
import logging
import time
import xgrammar
from transformers.generation.logits_process import LogitsProcessor
from transformers.utils import add_start_docstrings
from cars.oracle_trie import Trie
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class GrammarConstrainedProcessor(LogitsProcessor):
    def __init__(self, tokenizer, grammar_constraint, device, generate_start_index):
        self.tokenizer = tokenizer
        self.grammar_constraint = grammar_constraint
        self.device = device
        self.generate_start_index = generate_start_index
        #self.logits_process_time = 0

    def __call__(self, input_ids : torch.LongTensor, scores : torch.FloatTensor) -> torch.FloatTensor:
        #start_time = time.time()
        
        self._set_generated_tokens(input_ids)

        if not self.grammar_constraint.try_advance_token_ids(self.generated_tokens):
            raise ValueError("Nongrammatical? Should not happen")

        acceptance = self.grammar_constraint.filter_vocab()
        scores = scores.clone()
        xgrammar.apply_token_bitmask_inplace(scores, acceptance.to(self.device, non_blocking = True))

        #end_time = time.time()
        #self.logits_process_time += end_time - start_time

        return scores


    def _set_generated_tokens(self, input_ids : torch.LongTensor):
        assert len(input_ids)==1

        #if self.generate_start_index is None:
        #    self.generate_start_index = input_ids.size(1) # the end of input sequence of tokens
        #    logger.debug("PARSER STARTING")

        self.generated_tokens = input_ids[0, self.generate_start_index:]

        if logger.level <= logging.INFO:
            text = self.tokenizer.decode(self.generated_tokens)
            logger.info("Current text: \"%s\" / %s", text, self.generated_tokens)
