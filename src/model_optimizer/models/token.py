
from transformers import AutoTokenizer
def get_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                                  trust_remote_code=True)
    # Set tokenizer padding token if needed
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer