import os

DEFAULT_CONFIG = """
from pathlib import Path
import os 

# --- PATHS ---
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

PATHS = {
    "DATA_READY": DATA_DIR / "processed" / "training_ready.csv",
    "RAW_DATA": DATA_DIR / "raw" / "training_data.csv",
    "FINAL_FEATURES": DATA_DIR / "processed" / "final_features.csv",
    "GOLDEN_DATA": DATA_DIR / "processed" / "golden_dataset.csv",
    "CLUSTERS_DIR": DATA_DIR / "cluster_visuals",
    "MODELS_DIR": DATA_DIR / "checkpoints",
}

# --- SETTINGS ---
GESTURE_LABELS = [
    "0. NOISE_RANDOM", "1. THE_POINT", "2. THE_CLICK", "3. RIGHT_CLICK",
    "4. SCROLL", "5. FIST", "6. PREVIOUS", "7. NEXT", "8. TAB",
    "9. WIN", "10. ZOOM", "11. PALM", "12. CUT", "13. VOLUME",
    "14. DELETE", "15. CTRL_PINKY", "16. THE_SHHH", "17. GEN_Z"
]

os.makedirs(DATA_DIR / "raw", exist_ok=True)
os.makedirs(DATA_DIR / "processed", exist_ok=True)
os.makedirs(PATHS["CLUSTERS_DIR"], exist_ok=True)
os.makedirs(PATHS["MODELS_DIR"], exist_ok=True)

COLORS = {
    "neon_green": (57, 255, 20),
    "electric_blue": (255, 255, 0),
    "alert_red": (0, 0, 255),
    "dark_bg": (20, 20, 20),
    "text": (200, 200, 200),
    "lock_fill": (0, 100, 255)
}

# --- HYPER PARAMETERS (FACTORY RESET) ---
CONFIG = {
    "TARGET_FPS": 20,
    "OPEN_THRESH": 0.20,
    "PINCH_START": 0.10,
    "PINCH_STOP": 0.25,
    "THUMB_LOCK_THRESH": 1.2,
    "SMOOTHING": 5.0,
    "DPI_SENSITIVITY": 2.5,
    "CLICK_DEADZONE": 35,
    "X_OFFSET": 0.0,
    "Y_OFFSET": 0.0,
    "LOCK_TIME": 1.5,
    "SCROLL_SPEED_BASE": 5,
    "SCROLL_ACCELERATION": 3.0,
    "DRAG_BUFFER": 0.15,
    "DOUBLE_CLICK_GAP": 0.25,
    "RIGHT_HOLD_MIDDLE": 0.5,
    "COPY_PASTE_HOLD": 0.6,
    "SELECT_ALL_HOLD": 0.8,
    "SELECT_ALL_DIAMETER": 0.20,
    "DELETE_LOOPS": 4,
    "NAV_LOOPS": 6,
}
"""

with open("config_backup.py", "w") as f:
    f.write(DEFAULT_CONFIG)
print("✅ Created 'config_backup.py'. If src/config.py breaks, copy content from here.")