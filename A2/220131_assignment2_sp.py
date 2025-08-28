import argparse
import os
import unicodedata

RESERVED = ["<pad>", "<unk>", "<s>", "</s>"]
MARKER = "▁"

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def normalize_text(text):
    """Apply Unicode NFKC normalization and lowercasing."""
    text = unicodedata.normalize("NFKC", text).casefold()
    cleaned = []
    prev_space = False
    for ch in text:
        if ch.isspace():
            if not prev_space:
                cleaned.append(" ")
                prev_space = True
        else:
            cleaned.append(ch)
            prev_space = False
    text = "".join(cleaned).strip()
    text = MARKER + text.replace(" ", MARKER)
    return text
    # return unicodedata.normalize("NFKC", text).lower()

def train_sp_tokenizer(text, vocab_size):
    """Adapt BPE training with space marker ▁."""
    vocab = RESERVED.copy()
    corpus = list(text)
    base_vocab = set(corpus)
    vocab.extend(sorted(base_vocab))
    merges = []
    while len(vocab) < vocab_size:
        pair_freq = {}
        for i in range(len(corpus) - 1):
            pair = (corpus[i], corpus[i+1])
            if pair not in pair_freq:
                pair_freq[pair] = 0
            pair_freq[pair] += 1
        if not pair_freq:
            break
        best = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
        new_sym = best[0] + best[1]
        new_corpus = []
        i = 0
        while i < len(corpus):
            if i < len(corpus)-1 and (corpus[i], corpus[i+1]) == best:
                new_corpus.append(new_sym)
                i += 2
            else:
                new_corpus.append(corpus[i])
                i += 1
        corpus = new_corpus
        vocab.append(new_sym)
        merges.append(best)
    rank = {pair:i for i,pair in enumerate(merges)}
    tokenizer = {"merges":merges,"rank":rank,"vocab":set(vocab)}
    return vocab, tokenizer

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_sp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def bpe_encode(text, rank):
    word = list(text)
    pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
    while True:
        candidate = None
        candidate_rank = None
        for p in pairs:
            if p in rank:
                r = rank[p]
                if candidate is None or r < candidate_rank:
                    candidate = p
                    candidate_rank = r
        if candidate is None:
            break
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word)-1 and (word[i], word[i+1]) == candidate:
                new_word.append(word[i]+word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = new_word
        pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
    return word

def tokenize(text, tokenizer, seed=42):
    """Implement sampling-based tokenization (deterministic if seed fixed)."""
    rank = tokenizer["rank"]
    out = []
    for ch in text:
        encoded = bpe_encode(ch, rank)
        out.extend(encoded)
    return out

def detokenize(tokens, tokenizer):
    text = "".join(tokens)
    text = text.replace(MARKER, " ")
    return text.strip()

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_sp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_sp_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "220131"

    train_text = load_training_data(args.train)
    train_text = normalize_text(train_text)
    vocab, tokenizer = train_sp_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = normalize_text(f.read())
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
