"""Helper functions for BPE (Byte Pair Encoding) tokenization."""

import os
from collections import Counter, defaultdict
from typing import BinaryIO
import regex as re  # type: ignore

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_stats(
        working_vocab: dict[tuple[bytes, ...], int]
) -> defaultdict[tuple[bytes, bytes], int]:
    """Map from bigram tuple to count in corpora."""
    pairs: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    for symbols, freq in working_vocab.items():
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
   
    for word_tuple in v_in:
        # Convert tuple to list for easier manipulation
        word_list = list(word_tuple)
       
        # Find and merge consecutive pairs
        i = 0
        while i < len(word_list) - 1:
            if word_list[i] == pair[0] and word_list[i + 1] == pair[1]:
                # Merge the pair into a single token by concatenating bytes
                merged_token = pair[0] + pair[1]
                word_list = word_list[:i] + [merged_token] + word_list[i + 2:]
                # Don't increment i, check the same position again
            else:
                i += 1
        
        # Convert back to tuple and store
        new_word_tuple = tuple(word_list)
        v_out[new_word_tuple] = v_in[word_tuple]
    
    return v_out

def compute_bpe_merge(
        working_vocab: dict[tuple[bytes, ...], int], num_merges: int
) -> list[tuple[bytes, bytes]]:
    """
    Compute the BPE merges for the given number of merges.
    """
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(working_vocab)
        best = max(pairs.items(), key=lambda item: (item[1], item[0]))[0]
        merges.append(best)
        working_vocab = merge_vocab(best, working_vocab)
    return merges

def pretokenize(chunk: str) -> Counter[tuple[bytes, ...]]:
    """Pretokenize a chunk of text and return UTF-8 byte-level token counts."""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    local_vocab: Counter[tuple[bytes, ...]] = Counter()
    matches_iterator = re.finditer(PAT, chunk)
    for match in matches_iterator:
        byte_tuple = tuple(bytes([b]) for b in match.group().encode("utf-8"))
        local_vocab[byte_tuple] += 1
    return local_vocab