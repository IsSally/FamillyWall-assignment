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

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def trainable_param_percentage(model: torch.nn.Module) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_params == 0:
        return 0.0  # Avoid division by zero
    return 100.0 * trainable_params / total_params

class EarlyStopNotifier(TrainerCallback):
    def __init__(self):
        self.triggered = False
        self.where = None

    def on_evaluate(self, args, state, control, **kwargs):
        # EarlyStoppingCallback sets this when patience is exceeded
        if control.should_training_stop and not self.triggered:
            self.triggered = True
            self.where = (state.global_step, state.epoch)

    def on_train_end(self, args, state, control, **kwargs):
        if self.triggered and state.is_local_process_zero:
            print(
                f"Early stopping at step {self.where[0]} (epoch {self.where[1]:.2f}). "
                f"Best '{args.metric_for_best_model}' = {state.best_metric} at {state.best_model_checkpoint}."
            )
