from typing import Iterable
from collections.abc import Iterator
from cs336_basics.bpe_helper import pretokenize, pretokenize_iter

class Tokenizer():
    """
    Tokenizer using BPE to tokenize text.
    """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.reverse_vocab: dict[bytes, int] = {}
        for k, v in self.vocab.items():
            self.reverse_vocab[v] = k
        self.merges = set(merges)
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        pass

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # TODO: split out special chars, pretokenize text in chunks
        _, pretokenized_string = pretokenize(text)

        # TODO: encapsulate in more loops? e.g., He, llo may be merged into Hello in a second loop (note this is unfort not the case for GPT2)
        tokenized_string: list[bytes] = []
        for token in pretokenized_string:
            token_list = list(token)
            j = 0
            while j < len(token_list) - 1:
                if (token_list[j], token_list[j + 1]) in self.merges:
                    merged_token = token_list[j] + token_list[j + 1]
                    token_list = token_list[:j] + [merged_token] + token_list[j + 2:]
                else:
                    j += 1
            print(f"token list is {token_list}")
            print(f"tokenized string is {tokenized_string}")
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
        _, pretokenized_string = pretokenize_iter(iterable)
        for token in pretokenized_string:
            token_list = list(token)
            j = 0
            while j < len(token_list) - 1:
                if (token_list[j], token_list[j + 1]) in self.merges:
                    merged_token = token_list[j] + token_list[j + 1]
                    token_list = token_list[:j] + [merged_token] + token_list[j + 2:]
                else:
                    yield token_list[j]
                    j += 1

            # for new_token in token_list:
            #     yield self.reverse_vocab[new_token]

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        output_text = b""
        for token_id in ids:
            output_text += self.vocab[token_id]
        return output_text.decode("utf-8", errors='replace')