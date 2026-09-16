from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# RAW DATA FILES
# ============================================================

PROPERTYFINDER_FILE = RAW_DATA_DIR / "propertyfinder.csv"


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_project_directories() -> None:
    """
    Ensure required project directories exist.
    """

    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        MODELS_DIR,
        SRC_DIR,
        TESTS_DIR,
        NOTEBOOKS_DIR,
        SCRIPTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)