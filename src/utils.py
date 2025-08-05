import gc
import torch
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

def clear_gpu_cache():
    """
    Frees up unused GPU memory.
    Call between large ops to reduce fragmentation and OOM risk.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_synonyms(word, max_syn=5):
    """Return up to max_syn WordNet synonyms for a given word (nouns only)."""
    synsets = wn.synsets(word, pos=wn.NOUN)
    lemmas = set(
        lemma.name().replace('_', '-').lower()
        for syn in synsets for lemma in syn.lemmas()
        if lemma.name().lower() != word.lower()
    )
    # limit to the most common ones
    return list(lemmas)[:max_syn]


def rule_variants(desc, domain):
    base, tld = domain.lower().split('.', 1)
    tokens = base.replace('-', ' ').split()
    out = []
    for i, tok in enumerate(tokens):
        for syn in get_synonyms(tok):
            new_tokens = tokens.copy()
            new_tokens[i] = syn
            slug = '-'.join(new_tokens)
            out.append((desc, f"{slug}.{tld}"))
    return out