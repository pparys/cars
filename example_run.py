import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarAlignedOracleLogitsProcessor #, GrammarConstrainedLogitsProcessor
from transformers_gad.oracle.oracle_trie import Trie


#MODEL_ID = "TinyLlama/TinyLlama_v1.1"
#MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"
MODEL_ID = "hsultanbey/codegen350multi_finetuned"
GRAMMAR_PATH = "examples/test/binary_len_5_new.ebnf"
MAX_NEW_TOKENS = 512

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.resize_token_embeddings(len(tokenizer)) #PP: dodane
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
logger = logging.getLogger(__name__)

logger.debug("Liczba tokenów model.config.vocab_size: %d",  model.config.vocab_size)
logger.debug("Liczba tokenów len(tokenizer): %d", len(tokenizer))
logger.debug("Liczba tokenów tokenizer.vocab.size: %d", tokenizer.vocab_size)
logger.debug("pad_token_id: %d", model.config.pad_token_id)

# Load EBNF grammar
with open(GRAMMAR_PATH, "r") as f:
    grammar_str = f.read()
grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

# Initialize logits processor for the grammar
gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar, save_log = True, oracle_trie = Trie())
#gcd_processor = GrammarConstrainedLogitsProcessor(grammar)
inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
logits_processors = LogitsProcessorList([
    inf_nan_remove_processor,
    gad_oracle_processor,
    #gcd_processor,
])

# Tokenize prompt into ids
prompt = "Generate a binary string of length 5"
inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt" , padding=True #, truncation=True, return_attention_mask=True
    )

# Inference Loop
outputs = []
#for _ in tqdm(range(10), desc="Running Inference"):
for i in range(1000):
    # Incremental parser state must be reset after each generation
    gad_oracle_processor.reset()

    # Generate sequences
    try:
        output = model.generate(
            inputs["input_ids"],
            do_sample=True,
            max_new_tokens=MAX_NEW_TOKENS,
            logits_processor=logits_processors, attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id
        )
        prob = gad_oracle_processor.check_full_string(output)
        
    except ValueError:
        print("Sample", i, "failed")
        continue;

    # Detokenize generated output
    input_length = 1 if model.config.is_encoder_decoder else inputs["input_ids"].shape[1]
    #generated_tokens = output.sequences[:, input_length:]
    #generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    #outputs.append(generations[0])
    text = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    outputs.append(text)
    print("Sample", i, "success:", text, "/", output[0][input_length:].tolist(), "probability:", prob)

#print("Number of interations:", i)
#print(outputs)
