import json
from base64 import b64decode
import random
import regex as re # type: ignore
from .tokenizer import Tokenizer

TS_DATA_FILEPATH = "data/TinyStoriesV2-GPT4-valid.txt"
TS_VOCAB_FILEPATH = "models/tiny_stories_train_vocab.json"
TS_MERGES_FILEPATH = "models/tiny_stories_train_merges.txt"

def get_max_len(vocab_filepath: str):
    """
    Check that the longest token in the vocabulary makes sense.
    """
    with open(vocab_filepath, "r") as f:
        vocab = json.loads(f.readline())
        tokens = [b64decode(v.encode("utf-8")) for v in vocab.values()]
        return max(tokens, key=len)

def sample_docs(filepath: str, tokenizer: Tokenizer, delimiter: str):
    """
    Sample 10 docs from selected corpora and return compression ratio
    (bytes/token).
    """
    pattern = re.escape(delimiter) + r'(.*?)' + re.escape(delimiter)
    with open(filepath, "r") as f:
        docs = re.findall(pattern, f.read(), re.DOTALL)
        samples = random.choices(docs, k=10)
    compression_ratios = []
    for sample in samples:
        sample_bytes = sample.encode('utf-8')
        tokenized_sample = tokenizer.encode(sample)
        ratio = len(sample_bytes) / len(tokenized_sample)
        compression_ratios.append(ratio)
    return compression_ratios

if __name__ == "__main__":
    DELIMITER = '<|endoftext|>'
    ts_tokenizer = Tokenizer.from_files(TS_VOCAB_FILEPATH, TS_MERGES_FILEPATH, [DELIMITER])
    ratios = sample_docs(TS_DATA_FILEPATH, ts_tokenizer, DELIMITER)
    print(f"Compression ratios: {ratios}")
    print(f"Average: {sum(ratios)/len(ratios):.2f} bytes/token")
    print(f"Word with max len is {get_max_len(TS_VOCAB_FILEPATH)}")
