# Central configuration for the healthcare chatbot core project.

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
SYNTHEA_DIR = DATA_DIR / "synthea"
EBM_DIR = DATA_DIR / "ebm_nlp"
EBM_CORPUS_DIR = EBM_DIR / "ebm_nlp_2_00"
EBM_DOC_DIR = EBM_CORPUS_DIR / "documents"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
TOP_N_LABELS = int(os.environ.get("TOP_N_LABELS", "8"))
TEST_SIZE = 0.2
EXPERIMENT_CASES = int(os.environ.get("EXPERIMENT_CASES", "20"))
FEW_SHOT_EXAMPLES = int(os.environ.get("FEW_SHOT_EXAMPLES", "4"))
KNOWLEDGE_TOP_K = int(os.environ.get("KNOWLEDGE_TOP_K", "2"))
SELF_CONSISTENCY_RUNS = int(os.environ.get("SELF_CONSISTENCY_RUNS", "3"))
REBALANCE_TOP_LABELS = os.environ.get("REBALANCE_TOP_LABELS", "true").lower() == "true"
