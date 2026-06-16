"""
config.py
---------
Single source of truth for every path, model name, and tunable parameter
in the pipeline. Every other module imports from here — nothing is hardcoded
elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        "Add it to your .env file: GROQ_API_KEY=your_key_here"
    )

ROOT_DIR: Path = Path(__file__).resolve().parent

DATA_DIR            : Path = ROOT_DIR / "data"
RAW_DATA_DIR        : Path = DATA_DIR / "raw"
RESULTS_DIR         : Path = DATA_DIR / "results"
OUTPUTS_DIR         : Path = ROOT_DIR / "outputs"
CHARTS_DIR          : Path = OUTPUTS_DIR / "charts"
REPORTS_DIR         : Path = OUTPUTS_DIR / "reports"
TRUTHFULQA_CSV      : Path = RAW_DATA_DIR / "truthfulqa_mc1.csv"

FLASH_RESPONSES_JSON: Path = RESULTS_DIR / "llama_8b_responses.json"
PRO_RESPONSES_JSON  : Path = RESULTS_DIR / "llama_70b_responses.json"

SCORED_RESULTS_JSON : Path = RESULTS_DIR / "scored_results.json"

HF_DATASET_NAME     : str  = "truthful_qa"
HF_MC_CONFIG        : str  = "multiple_choice"
HF_GEN_CONFIG       : str  = "generation"
HF_SPLIT            : str  = "validation"

MODEL_FLASH         : str  = "llama-3.1-8b-instant"
MODEL_PRO           : str  = "llama-3.3-70b-versatile"

MODELS: dict[str, str] = {
    "flash" : MODEL_FLASH,
    "pro"   : MODEL_PRO,
}

TEMPERATURE             : float = 0.0
MAX_OUTPUT_TOKENS       : int   = 10      
REQUEST_DELAY_SEC       : float = 0.5     
HALLUCINATION_THRESHOLD : float = 0.5

HIGH_RISK_KEYWORDS: list[str] = [
    "health", "law", "finance", "medical", "nutrition",
    "drug", "vaccine", "legal", "money", "politics",
]

for _dir in [RAW_DATA_DIR, RESULTS_DIR, CHARTS_DIR, REPORTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)