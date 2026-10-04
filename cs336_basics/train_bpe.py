import os
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter
from multiprocessing import Pool

def process_chunk(input_path, start, end, special_tokens): # for start, end in zip(boundaries[:-1], boundaries[1:]):
        
            with open(input_path, "rb") as f:
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token

            pattern = "|".join(re.escape(token) for token in special_tokens)
            chunk = re.split(pattern, chunk)

            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pretokenized = []

            for elem in chunk:
                pretokenized += re.findall(PAT, elem)
            
            pretokenizedCounter = Counter(pretokenized)
            return pretokenizedCounter

def pretokenize(input_path, special_tokens):


    with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            # processing chunks

    args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(5) as p:
        results = p.starmap(process_chunk, args)

    total = Counter()

    for counter in results:
        total.update(counter)
    
    return total


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    
    total = pretokenize(input_path, special_tokens)



    word_bytes = {}

    for pretoken, count in total.items():

        byte_sequence = tuple(bytes([b]) for b in pretoken.encode())
        word_bytes[byte_sequence] = count

    pairs_counts = Counter()
    pair_to_words = {}

    for byte_sequence, count in word_bytes.items():
        for i in range(len(byte_sequence) - 1):

            pair = (byte_sequence[i], byte_sequence[i+1])
            pairs_counts[pair] += count

            if pair not in pair_to_words:
                pair_to_words[pair] = set()
            
            pair_to_words[pair].add(byte_sequence)
    
    best_pair = max(pairs_counts, key=lambda x: (pairs_counts[x], x))

    for word in list(pair_to_words[best_pair]):
        count = word_bytes[word]

        new_word = []

        i = 0

        while i < len(word):

            if (
                i < len(word) - 1
                and (word[i], word[i+1]) == best_pair
            ):
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        
        new_word = tuple(new_word)

        old_pairs = [
            (word[i], word[i+1])
            for i in range(len(word) - 1)
        ]

        for pair in old_pairs:
            pairs_counts[pair] -= count

            if pairs_counts[pair] == 0:
                del pairs_counts[pair]
        
        for pair in set(old_pairs):
            pair_to_words[pair].discard(word)

            if not pair_to_words[pair]:
                del pair_to_words[pair]


        del word_bytes[word]
        word_bytes[new_word] = count

        new_pairs = [
             (new_word[i], new_word[i+1])
            for i in range(len(new_word) - 1)
        ]

        for pair in new_pairs:
            pairs_counts[pair] += count
        
        for pair in set(new_pairs):
            pair_to_words.setdefault(pair, set()).add(new_word)

    


      



    


    

    

