# src/data_augmentation.py

import pandas as pd
from .utils import rule_variants

def augment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the original synthetic-data DataFrame and return
    an augmented version with rule-based variants.
    """
    augmented = []
    for _, row in df.iterrows():
        desc = row["business_description"]
        dom  = row["domain"]
        # Keep the original
        augmented.append({"business_description": desc, "domain": dom})
        # Add rule-based variants
        for d, new_dom in rule_variants(desc, dom):
            augmented.append({"business_description": d, "domain": new_dom})

    aug_df = pd.DataFrame(augmented).drop_duplicates().reset_index(drop=True)
    return aug_df

def augment_and_save(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Reads input_csv, augments it, writes to output_csv, and returns the DataFrame.
    """
    df = pd.read_csv(input_csv)
    aug_df = augment_dataframe(df)
    aug_df.to_csv(output_csv, index=False)
    print(f"Original size: {len(df)}, Augmented size: {len(aug_df)}")
    return aug_df
