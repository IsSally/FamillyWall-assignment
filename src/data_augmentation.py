# src/data_augmentation.py

import pandas as pd
from .utils import rule_variants

# We will augment data by using synonyms of the words
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

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
# Apply to every row and collect
def augment_data(initial_data_csv, aug_path_csv):
  df = pd.read_csv(initial_data_csv)
  augmented = []
  for _, row in df.iterrows():
      desc = row["business_description"]
      dom  = row["domain"]
      # keep the original
      augmented.append({"business_description": desc, "domain": dom})
      # add rule-based variants
      for d, new_dom in rule_variants(desc, dom):
          augmented.append({"business_description": d, "domain": new_dom})

  # Build a DataFrame
  aug_df = pd.DataFrame(augmented).drop_duplicates()
  print(f"Original size: {len(df)}, Augmented size: {len(aug_df)}")
  aug_df.to_csv(aug_path_csv, index=False)
  return aug_df



