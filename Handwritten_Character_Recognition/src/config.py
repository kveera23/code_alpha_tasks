from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Directories
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "emnist"

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

PLOT_DIR = OUTPUT_DIR / "plots"

REPORT_DIR = OUTPUT_DIR / "reports"

ASSET_DIR = PROJECT_ROOT / "assets"

# ==========================================================
# Files
# ==========================================================

MODEL_PATH = MODEL_DIR / "cnn_model.keras"

HISTORY_PATH = OUTPUT_DIR / "training_history.npy"

CSS_PATH = ASSET_DIR / "styles.css"