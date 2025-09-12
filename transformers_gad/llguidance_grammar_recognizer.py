import llguidance, llguidance.hf, llguidance.torch
import os, logging

logger = logging.getLogger(__name__)

class LlguidanceTokenRecognizer:
    def __init__(self, grammar_str, tokenizer):
        #print(grammar_str)
        ll_grammar = llguidance.grammar_from("grammar", grammar_str)
        self.ll_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
        err = llguidance.LLMatcher.validate_grammar(ll_grammar, self.ll_tokenizer)
        if err:
            raise ValueError(f"Grammar error: {err}")
        self.ll_matcher = llguidance.LLMatcher(self.ll_tokenizer, ll_grammar, log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")))
        self.current_index = 0
        self._grammar_bitmask = llguidance.torch.allocate_token_bitmask(1, self.ll_tokenizer.vocab_size)
        #print("VOCAB SIZE:",  ll_tokenizer.vocab_size, len(self._grammar_bitmask[0]))
    
    def reset(self):
        #print("PARSER RESET")
        self.ll_matcher.reset()
        self.current_index = 0

    def try_advance_token_ids(self, token_ids):
        logger.debug(f"PARSER - EATING {len(token_ids)-self.current_index} TOKENS: {token_ids[self.current_index:]}")
        new_tokens = token_ids[self.current_index:].tolist()
        r = self.ll_matcher.try_consume_tokens(new_tokens)
        if r == 0 and len(new_tokens) == 1 and new_tokens[0] == self.ll_tokenizer.eos_token and self.ll_matcher.is_accepting():
            r = 1  # the ll_matcher does not want to read the EOS token after accepting, let's do that manually
        self.current_index += r
        if r < len(new_tokens):
            logger.debug(f"Parsing error: {r} {len(new_tokens)} {self.ll_matcher.get_error()}")
        return r == len(new_tokens)
    
    def filter_vocab(self):
        llguidance.torch.fill_next_token_bitmask(self.ll_matcher, self._grammar_bitmask, 0)
        return self._grammar_bitmask
    
    def nonzero_bits(self): # it assumes that filter_vocab was already executed
        set_bits = []
        tensor = self._grammar_bitmask[0]
        for i in tensor.nonzero().flatten():
            for j in range(32):
                if (tensor[i]>>j)&1:
                    set_bits.append(i*32+j)
        return set_bits

#def bitmask_get_bit(tensor, i):
#    return (tensor[i // 32] >> (i % 32)) & 1

