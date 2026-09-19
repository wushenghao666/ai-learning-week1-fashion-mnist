from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
SEED = 42
IMAGE_SIZE = 28
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 1e-3
VAL_RATIO = 0.1
MODEL_NAME = "mlp"
