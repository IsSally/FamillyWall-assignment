# src/__init__.py

__version__ = "0.1.0"

# top‐level imports to flatten the API
from .data_generation    import generate_synthetic_data
from .data_augmentation  import augment_and_save
from .preprocessing      import preprocess_dataset
