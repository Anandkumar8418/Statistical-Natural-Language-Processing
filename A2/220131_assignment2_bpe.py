import argparse
import os

END = b"</w>"

def _to_byte_symbols(byte_word):
    # byte_word is a bytes object; return list of 1-byte bytes + END
    return [bytes([b]) for b in byte_word] + [END]

def load_training_data(train_path):
    """Load and return raw text for training."""
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text
END = b"</w>"

def _to_byte_symbols(byte_word):
    # byte_word is a bytes object; return list of 1-byte bytes + END
    return [bytes([b]) for b in byte_word] + [END]


def train_bpe_tokenizer(text, vocab_size):
    """Learn BPE merges at BYTE level and return (vocab_strings, tokenizer)."""
    # Reserved tokens (strings)
    vocab_strings = ["<pad>", "<unk>", "<s>", "</s>"]

    # Build corpus at byte level
    words = text.encode("utf-8").split()             # list[bytes]
    corpus = [_to_byte_symbols(w) for w in words]    # list[list[bytes]]

    # Initial byte vocab
    base_vocab = set()
    for w in corpus:
        base_vocab.update(w)
    vocab_bytes = [END] + sorted(b for b in base_vocab if b != END)

    merges = []  # list of (bytes, bytes)

    def current_vocab_size():
        return len(vocab_strings) + len(vocab_bytes) + len(merges)

    while current_vocab_size() < vocab_size:
        # count bigram frequencies with dict
        bigram_freq = {}
        for w in corpus:
            for i in range(len(w) - 1):
                pair = (w[i], w[i+1])
                if pair not in bigram_freq:
                    bigram_freq[pair] = 0
                bigram_freq[pair] += 1

        if not bigram_freq:
            break

        # pick most frequent, break ties lexicographically
        best = max(bigram_freq.items(),
                   key=lambda x: (x[1], x[0]))[0]  # (a, b) bytes

        a, b = best
        new_sym = a + b
        new_corpus = []
        for w in corpus:
            i = 0
            nw = []
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i+1] == b:
                    nw.append(new_sym)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            new_corpus.append(nw)
        corpus = new_corpus

        merges.append((a, b))
        vocab_bytes.append(new_sym)

    tokenizer = {"merges": merges, "vocab_bytes": vocab_bytes}

    # human-readable vocab (for saving)
    for tok in vocab_bytes:
        try:
            vocab_strings.append(tok.decode("utf-8"))
        except UnicodeDecodeError:
            vocab_strings.append(tok.hex())

    return vocab_strings, tokenizer


def tokenize(text, tokenizer):
    """Tokenize input text using trained BPE model. Returns list[bytes]."""
    merges = tokenizer["merges"]
    words = text.encode("utf-8").split()
    corpus = [_to_byte_symbols(w) for w in words]

    for (a, b) in merges:
        new_corpus = []
        for w in corpus:
            i = 0
            nw = []
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i+1] == b:
                    nw.append(a + b)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            new_corpus.append(nw)
        corpus = new_corpus

    return [sym for w in corpus for sym in w]


def detokenize(tokens, tokenizer):
    words = []
    cur = []
    for tok in tokens:
        if isinstance(tok, int):
            tok = bytes([tok])
        elif isinstance(tok, str):
            tok = tok.encode("utf-8") if tok != "</w>" else END

        if tok == END:
            word_bytes = b"".join(cur)
            words.append(word_bytes.decode("utf-8"))
            cur = []
        else:
            cur.append(tok)

    # flush remaining
    if cur:
        word_bytes = b"".join(cur)
        words.append(word_bytes.decode("utf-8"))

    return " ".join(words)



def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")


def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            if isinstance(tok, int):
                tok = bytes([tok])
            if isinstance(tok, bytes):
                try:
                    tok_str = tok.decode("utf-8")
                except UnicodeDecodeError:
                    tok_str = tok.hex()
            else:
                tok_str = str(tok)
            f.write(tok_str + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        try:
            f.write(text)
        except UnicodeEncodeError:
            f.write("[ERROR: Unable to save detokenized text]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    # Replace with your actual roll number
    rollno = "220131"

    # Training
    train_text = load_training_data(args.train)
    vocab, tokenizer = train_bpe_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    # Tokenization
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    # Detokenization
    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
