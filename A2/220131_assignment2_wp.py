import argparse
import os
import numpy as np

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text
def _pretokenize(text):
    return text.split()

def _word_to_pieces(word):
    if not word:
        return []
    out = [word[0]]
    for ch in word[1:]:
        out.append("##" + ch)
    return out

def _merge_token(a, b):
    a_has = a.startswith("##")
    a_base = a[2:] if a_has else a
    b_base = b[2:] if b.startswith("##") else b
    return ("##" + a_base + b_base) if a_has else (a_base + b_base)

def train_wordpiece_tokenizer(text, vocab_size):
    """Learn WordPiece vocab with reserved tokens first."""
    vocab = ["<pad>", "<unk>", "<s>", "</s>"]
    words = _pretokenize(text)
    corpus = [_word_to_pieces(w) for w in words if w]

    base = set()
    for w in corpus:
        for t in w:
            base.add(t)
    vocab.extend(sorted(base))
    merges = []

    def cur_size():
        return len(vocab)

    while cur_size() < vocab_size:
        pair_freq = {}
        for w in corpus:
            for i in range(len(w) - 1):
                p = (w[i], w[i+1])
                if p not in pair_freq:
                    pair_freq[p] = 0
                pair_freq[p] += 1
        if not pair_freq:
            break

        N = 0
        for w in corpus:
            N += len(w)

        tok_freq = {}
        for w in corpus:
            for t in w:
                tok_freq[t] = tok_freq.get(t, 0) + 1

        best_pair = None
        best_score = -float("inf")
        for (a, b), f_ab in pair_freq.items():
            f_new = f_ab
            f_a = tok_freq.get(a, 0)
            f_b = tok_freq.get(b, 0)
            if f_new == 0 or N == 0:
                score = -float("inf")
            else:
                score = (f_new - (f_a + f_b)) * np.log(f_new / N)
            if (score > best_score) or (score == best_score and (best_pair is None or (a, b) < best_pair)):
                best_score = score
                best_pair = (a, b)

        if best_pair is None:
            break

        a, b = best_pair
        new_sym = _merge_token(a, b)

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

        vocab.append(new_sym)
        merges.append((a, b))

    tokenizer = {"vocab": set(vocab)}
    return vocab, tokenizer

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_wp_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def tokenize(text, tokenizer):
    vocab = tokenizer["vocab"]
    words = text.split()
    out = []
    for w in words:
        i = 0
        cur = []
        L = len(w)
        while i < L:
            j = L
            found = None
            while j > i:
                piece = w[i:j] if i == 0 else "##" + w[i:j]
                if piece in vocab:
                    found = piece
                    break
                j -= 1
            if found is None:
                cur.append("<unk>")
                break
            cur.append(found)
            i = j
        out.extend(cur)
    return out

def detokenize(tokens, tokenizer):
    words = []
    cur = []
    for tok in tokens:
        if tok in {"<pad>", "<unk>", "<s>", "</s>"}:
            if cur:
                words.append("".join(cur))
                cur = []
            continue
        if tok.startswith("##"):
            cur.append(tok[2:])
        else:
            if cur:
                words.append("".join(cur))
                cur = []
            cur.append(tok)
    if cur:
        words.append("".join(cur))
    return " ".join(words)

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_wp_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_wp_detokenized.txt"
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
    vocab, tokenizer = train_wordpiece_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
