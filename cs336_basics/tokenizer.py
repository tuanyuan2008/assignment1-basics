"""
BPE tokenizer that loads a provided vocabulary and list of merges and uses them to encode
and decode text to/from token IDs.
"""

import json
from typing import Iterable
from collections.abc import Iterator
from base64 import b64decode
import regex as re  # type: ignore
from cs336_basics.bpe_helper import pretokenize, pretokenize_iter, TokenType

class Tokenizer():
    """
    BPE tokenizer that loads a provided vocabulary and list of merges and uses them to encode and decode text to/from token IDs.
    """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.reverse_vocab: dict[bytes, int] = {}
        for k, v in self.vocab.items():
            self.reverse_vocab[v] = k
        self.merges = merges  # Apply merges to our pretokens in the same order of creation
        self.special_tokens = special_tokens
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens. 
        """
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)

        vocab = {}
        for k, v in vocab_json.items():
            vocab[int(k)] = b64decode(v.encode('utf-8'))

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    # Use maxsplit=1 to split into 2 parts, handling first space token
                    parts = line.rstrip('\n').split(' ', 1)
                    if len(parts) != 2:
                        print(f"Error on line {line_num}: '{line.rstrip()}' has {len(parts)} parts")
                        continue
                    left, right = parts
                    merges.append((left.encode("utf-8"), right.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        if self.special_tokens:
            tokens = sorted(self.special_tokens, reverse=True)
            pattern = "(" + "|".join(re.escape(tok) for tok in tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        pretokenized_string = []
        for part in parts:
            if part in (self.special_tokens or []):
                pretokenized_string.append((part.encode("utf-8"), TokenType.SPECIAL))
            elif part:
                _, result = pretokenize(part)
                pretokenized_string.extend((res, TokenType.NORMIE) for res in result)

        # Look at each pretoken and apply the BPE merges
        tokenized_string: list[bytes] = []
        for token, token_type in pretokenized_string:
            if token_type == TokenType.SPECIAL:
                tokenized_string.append(token)
                continue
            token_list = list(token)
            while True:
                merged_this_pass = False
                # Identify the first applicable merge and use that to transform the pretoken
                for merge_left, merge_right in self.merges:
                    i = 0
                    while i < len(token_list) - 1:
                        if token_list[i] == merge_left and token_list[i + 1] == merge_right:
                            merged_token = token_list[i] + token_list[i + 1]
                            token_list = token_list[:i] + [merged_token] + token_list[i + 2:]
                            merged_this_pass = True
                            break
                        i += 1
                    # Go back to the list of merges and identify the next applicable merge
                    if merged_this_pass:
                        break
                if not merged_this_pass:
                    break
            tokenized_string.extend(token_list)

        encoded_string = []
        for new_token in tokenized_string:
            encoded_string.append(self.reverse_vocab[new_token])

        return encoded_string

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a 
        generator that lazily yields token IDs. This is required for 
        memory-eﬀicient tokenization of large files that we cannot directly 
        load into memory.
        """
        pretokenized_string = pretokenize_iter(iterable, self.special_tokens)
        for token, token_type in pretokenized_string:
            if token_type == TokenType.SPECIAL:
                yield self.reverse_vocab[token[0]]
                continue

            token_list = list(token)
            while True:
                merged_this_pass = False
                for merge_left, merge_right in self.merges:
                    i = 0
                    while i < len(token_list) - 1:
                        if token_list[i] == merge_left and token_list[i + 1] == merge_right:
                            merged_token = token_list[i] + token_list[i + 1]
                            token_list = token_list[:i] + [merged_token] + token_list[i + 2:]
                            merged_this_pass = True
                            break
                        i += 1
                    if merged_this_pass:
                        break
                if not merged_this_pass:
                    break

            for final_token in token_list:
                yield self.reverse_vocab[final_token]

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        output_text = b""
        for token_id in ids:
            output_text += self.vocab[token_id]
        return output_text.decode("utf-8", errors="replace")
