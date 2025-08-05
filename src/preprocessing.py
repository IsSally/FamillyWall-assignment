from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer
from functools import partial

def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction','output')
    Then concatenate them using newline characters
    :param sample: Sample dictionnary
    """

    BUSINESS_KEY = "Business: "
    DOMAIN_KEY = "Domain suggestions: "

    business = f"{BUSINESS_KEY}{sample['business_description']}"
    domain = f"{DOMAIN_KEY}{sample['domain']}"

    parts = [part for part in [business, domain] if part]

    formatted_prompt = "\n".join(parts)
    sample["text"] = formatted_prompt

    return sample

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """

    inputs = tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",#True,
    )
    inputs["labels"] = inputs["input_ids"]

    return inputs

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, csv_path):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    raw = load_dataset("csv", data_files=csv_path)["train"]
    split = raw.train_test_split(test_size=0.1, seed=42)

    # Add prompt to each sample
    print("Preprocessing dataset...")
    split = split.map(create_prompt_formats)#, batched=True)

    # Apply preprocessing to each batch of the dataset & and remove extra fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)

  
    dataset_train = split["train"].map(
        _preprocessing_function,
        batched=True,
        remove_columns=split["train"].column_names,#['complexity', 'industry'],
    )

    dataset_test = split["test"].map(
        _preprocessing_function,
        batched=True,
        remove_columns=split["test"].column_names,#['complexity', 'industry'],
    )
  
    # Shuffle dataset
    dataset_train = dataset_train.shuffle(seed=seed)
    dataset_test = dataset_test.shuffle(seed=seed)

    return DatasetDict({
        "train": dataset_train,
        "validation": dataset_test
    })


