from pathlib import Path

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"

ARTIFACT_DIR = ROOT_DIR / "artifacts"

QUBIT_COUNTS = [4, 6, 8]