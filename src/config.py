import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QA_DIR = DATA_DIR / "qa"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(PROCESSED_DIR / "faiss_index"))
if not Path(VECTOR_STORE_PATH).is_absolute():
    VECTOR_STORE_PATH = str(BASE_DIR / VECTOR_STORE_PATH)

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

SOURCE_URL = "https://ai-2027.com/"
