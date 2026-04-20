import os
import re
import json
import time
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Core filesystem paths.
DATA_DIR = "data"
SYNTHEA_DIR = os.path.join(DATA_DIR, "synthea")
EBM_DIR = os.path.join(DATA_DIR, "ebm_nlp")
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OPENAI_MODEL = "gpt-5.4-mini"
