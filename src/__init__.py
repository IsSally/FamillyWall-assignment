# src/__init__.py

__version__ = "0.1.0"

# top‐level imports to flatten the API
from .data_generation    import generate_synthetic_data
from .data_augmentation  import augment_and_save
from .preprocessing      import preprocess_dataset

import pandas as pd
import statistics as stats
from random import choice, sample
from functools import partial
from transformers import (AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,HfArgumentParser,AutoTokenizer,TrainingArguments,Trainer,GenerationConfig)
import os
from openai import OpenAI
import torch
from random import choice, sample
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import re
import gc
import requests
import json
import re
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import time
from transformers import TrainerCallback
import pynvml
from transformers import set_seed
import time
import math
import pynvml
from transformers import TrainerCallback
import json, time, platform, sys, subprocess, shutil
from pathlib import Path
import torch
from getpass import getpass
import optuna
from peft import PeftConfig, PeftModel
try:
    import importlib.metadata as md  # py3.8+
except Exception:
    import importlib_metadata as md   # backport
seed = 42
set_seed(seed)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
