import argparse
import os
import numpy as np

def load_training_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()
def _split_words(text):
    return [w for w in text.split() if w]

EOW = "</w>"
def _build_seed_vocab(words, max_sub_len=10, min_freq=2, seed_cap=20000):
    freq = {}
    chars = set()
    for w in words:
        for ch in w:
            chars.add(ch)
        n = len(w)
        for i in range(n):
            for L in range(1, min(max_sub_len, n - i) + 1):
                s = w[i:i+L]
                freq[s] = freq.get(s, 0) + 1
    seed = set(chars)
    for s, c in freq.items():
        if c >= min_freq:
            seed.add(s)
            if len(seed) >= seed_cap:
                break
    vocab = sorted(seed)
    probs = {}
    total = 0.0
    for t in vocab:
        c = freq.get(t, 1)
        p = float(c)
        probs[t] = p
        total += p
    for t in list(probs.keys()):
        probs[t] /= total
    return vocab, probs


def _logsumexp(vals):
    if not vals:
        return -float("inf")
    m = max(vals)
    if m == -float("inf"):
        return m
    s = 0.0
    for v in vals:
        s += np.exp(v - m)
    return m + np.log(s)


def _expected_counts_for_word(word, probs, token_set, max_tok_len):
    n = len(word)
    fwd = [-float("inf")] * (n + 1)
    bwd = [-float("inf")] * (n + 1)
    fwd[0] = 0.0
    for j in range(1, n + 1):
        cand = []
        i_min = max(0, j - max_tok_len)
        for i in range(i_min, j):
            t = word[i:j]
            if t in token_set:
                p = probs.get(t, 0.0)
                if p > 0.0:
                    cand.append(fwd[i] + np.log(p))
        fwd[j] = _logsumexp(cand)

    bwd[n] = 0.0
    for i in range(n - 1, -1, -1):
        cand = []
        j_max = min(n, i + max_tok_len)
        for j in range(i + 1, j_max + 1):
            t = word[i:j]
            if t in token_set:
                p = probs.get(t, 0.0)
                if p > 0.0:
                    cand.append(np.log(p) + bwd[j])
        bwd[i] = _logsumexp(cand)

    logZ = fwd[n]
    counts = {}
    if logZ == -float("inf"):
        return counts, 0.0

    i = 0
    for j in range(1, n + 1):
        i = max(0, j - max_tok_len)
    for i in range(n):
        j_max = min(n, i + max_tok_len)
        for j in range(i + 1, j_max + 1):
            t = word[i:j]
            if t in token_set:
                p = probs.get(t, 0.0)
                if p > 0.0:
                    log_q = fwd[i] + np.log(p) + bwd[j] - logZ
                    q = np.exp(log_q)
                    if q > 0.0:
                        counts[t] = counts.get(t, 0.0) + q

    expected_len = 0.0
    for t, c in counts.items():
        expected_len += c
    return counts, expected_len


def _em_update(words, probs, token_set, max_tok_len):
    tot_counts = {}
    total_expected_pieces = 0.0
    for w in words:
        counts, exp_len = _expected_counts_for_word(w, probs, token_set, max_tok_len)
        for t, c in counts.items():
            tot_counts[t] = tot_counts.get(t, 0.0) + c
        total_expected_pieces += exp_len
    total = 0.0
    new_probs = {}
    for t in token_set:
        c = tot_counts.get(t, 0.0)
        new_probs[t] = c
        total += c
    if total == 0.0:
        k = float(len(token_set))
        for t in token_set:
            new_probs[t] = 1.0 / k
        return new_probs
    for t in token_set:
        new_probs[t] /= total
        if new_probs[t] < 1e-12:
            new_probs[t] = 1e-12
    s = sum(new_probs.values())
    for t in token_set:
        new_probs[t] /= s
    return new_probs


def _prune(probs, keep_chars, target_size, drop_ratio=0.2):
    items = [(t, probs[t]) for t in probs]
    loss_items = []
    for t, p in items:
        loss = (0.0 if t in keep_chars else p * np.log(p))
        loss_items.append((loss, t))
    loss_items.sort(key=lambda x: (x[0], x[1]))
    if len(loss_items) <= target_size:
        kept = [t for _, t in loss_items]
        return kept
    k = max(target_size, int(len(loss_items) * (1.0 - drop_ratio)))
    kept = []
    for idx, (_, t) in enumerate(loss_items):
        if idx < k or t in keep_chars:
            kept.append(t)
    kept = sorted(set(kept))
    if len(kept) > target_size:
        kept = kept[:target_size]
    return kept


def train_unigram_tokenizer(text, vocab_size):
    """Initialize, run EM training loop, prune to vocab_size."""
    words = _split_words(text)
    vocab, probs = _build_seed_vocab(words, max_sub_len=10, min_freq=2, seed_cap=20000)
    token_set = set(vocab)
    keep_chars = set(ch for w in words for ch in w)

    max_tok_len = max(len(t) for t in token_set) if token_set else 1

    while len(token_set) > vocab_size:
        probs = _em_update(words, probs, token_set, max_tok_len)
        kept_list = _prune(probs, keep_chars, target_size=max(vocab_size, int(len(token_set) * 0.8)), drop_ratio=0.2)
        token_set = set(kept_list)
        s = sum(probs.get(t, 0.0) for t in token_set)
        if s == 0.0:
            k = float(len(token_set))
            probs = {t: 1.0 / k for t in token_set}
        else:
            probs = {t: max(probs.get(t, 0.0), 1e-12) for t in token_set}
            s = sum(probs.values())
            probs = {t: probs[t] / s for t in token_set}
        max_tok_len = max(len(t) for t in token_set)

    vocab_final = sorted(token_set)
    tokenizer = {
        "vocab": set(vocab_final),
        "probs": {t: probs.get(t, 1.0 / len(vocab_final)) for t in vocab_final},
        "max_len": max(len(t) for t in vocab_final) if vocab_final else 1
    }
    return vocab_final, tokenizer

def save_vocab(vocab, rollno, vocab_size):
    fname = f"{rollno}_assignment2_unigram_vocab_{vocab_size}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")

def tokenize(text, tokenizer):
    vocab = tokenizer["vocab"]
    probs = tokenizer["probs"]
    max_len = tokenizer["max_len"]
    words = _split_words(text)
    out = []
    for w in words:
        n = len(w)
        dp = [-float("inf")] * (n + 1)
        bp = [-1] * (n + 1)
        dp[0] = 0.0
        for i in range(n):
            if dp[i] == -float("inf"):
                continue
            j_max = min(n, i + max_len)
            for j in range(i + 1, j_max + 1):
                t = w[i:j]
                if t in vocab:
                    p = probs.get(t, 1e-12)
                    v = dp[i] + np.log(p)
                    if v > dp[j]:
                        dp[j] = v
                        bp[j] = i
        if dp[n] == -float("inf"):
            for ch in w:
                out.append(ch)
            out.append(EOW)
            continue
        toks = []
        cur = n
        while cur > 0:
            i = bp[cur]
            if i < 0:
                break
            toks.append(w[i:cur])
            cur = i
        toks.reverse()
        out.extend(toks)
        out.append(EOW)
    return out

def detokenize(tokens, tokenizer):
    words = []
    cur = []
    for t in tokens:
        if t == EOW:
            words.append("".join(cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        words.append("".join(cur))
    return " ".join(w for w in words if w)

def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_unigram_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok in tokens:
            f.write(tok + "\n")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_unigram_detokenized.txt"
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
    vocab, tokenizer = train_unigram_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)
