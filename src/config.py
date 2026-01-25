"""
NeuroCursor Configuration Management.
=====================================

This module defines the hyperparameter space for the NeuroCursor application.
The parameters are organized into an architectural "Layer Cake" model.

! WARNING !
Changing `GESTURE_LABELS` requires retraining the ONNX model.
Changing Physics Layers affects the "feel" of the cursor immediately.
"""

from pathlib import Path
import os 

# --- SYSTEM PATHS ---
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

PATHS = {
    "DATA_READY": DATA_DIR / "processed" / "training_ready.csv",
    "RAW_DATA": DATA_DIR / "raw" / "training_data.csv",
    "FINAL_FEATURES": DATA_DIR / "processed" / "final_features.csv",
    "GOLDEN_DATA": DATA_DIR / "processed" / "golden_dataset.csv",
    "CLUSTERS_DIR": DATA_DIR / "cluster_visuals",
    "MODELS_DIR": PROJECT_ROOT / "models"
}

# --- LABEL DEFINITIONS (CRITICAL) ---
# TensorFlow/Keras sorts classes alphabetically by default during training.
# This list MUST match that sorted order. If you add a label, you must
# retrain the model and verify this list matches the folder structure.
GESTURE_LABELS = [
    "CTRL_PINKY",       # C - toggle ctrl
    "CUT",              # C - CUT
    "DELETE_CLOSED",    # D - Phase 2 of Delete
    "DELETE_OPENED",    # D - Phase 1 of Delete
    "FIST",             # F - Scroll Mode / Drag Anchor
    "GEN_Z",            # G - Locking Mechanism
    "NEXT_CLOSED",      # N - Phase 1 of Browser Forward
    "NEXT_OPENED",      # N - Phase 2 of Browser Forward
    "NOISE_RANDOM",     # N - Null State
    "PALM",             # P - Phase 1 of copy and cut and select all and wave goodbye
    "PREVIOUS_CLOSED",  # P - Phase 1 of Browser Back
    "PREVIOUS_OPENED",  # P - Phase 2 of Browser Back
    "RESTING",          # R - resting position
    "RIGHT_CLICK",      # R - right click
    "SCROLL",           # S - scroll
    "TAB_L",            # T - Alt Tab Left
    "TAB_M",            # T - Alt Tab Menu
    "TAB_R",            # T - Alt Tab Right
    "THE_CLICK",        # T - click
    "THE_POINT",        # T - hover
    "THE_SHHH",         # T - Mute
    "VOLUME",           # V - volume controll
    "WIN",              # W - Win Key / Task View
    "ZOOM"              # Z - zoom controll
]

# --- MASTER CONFIGURATION ---
CONFIG = {
    # =========================================================
    # LAYER 1: INPUT SIGNAL (The Stabilizer)
    # =========================================================
    "TARGET_FPS": 30,               # Hardware limit for Camera
    "SKELETON_BUFFER_SIZE": 7,      # Frames to average for skeleton viz
    "OUTLIER_REJECTION_PX": 150,    # Ignore sudden jumps > 150px (glitches)
    "STABILIZER_THRESHOLD": 0.008,  # Movement < 0.8% of screen is ignored (Anchor)
    "STABILIZER_ALPHA": 0.77,       # Smoothing factor (Higher = More Lag, Smoother)

    # =========================================================
    # LAYER 2: CURSOR FEEL (Motion Physics)
    # =========================================================
    "VELOCITY_GATE": 0.002,         # Minimum speed to start moving cursor
    "BREAKOUT_VELOCITY": 0.02,      # Speed to transition from Precision to Fast
    "FRICTION_LOW": 0.32,           # High resistance (Precision Mode)
    "FRICTION_HIGH": 0.27,          # Low resistance (Travel Mode)

    # =========================================================
    # LAYER 3: CLICK PHYSICS (Intention)
    # =========================================================
    "PINCH_START": 0.034,           # Distance to trigger click
    "PINCH_STOP": 0.050,            # Distance to release click (Hysteresis)
    "PINCH_DYNAMIC_SCALE": 1.16,    # "Sticky Drag": How much harder to release when moving fast
    "PINCH_STOP_MAX": 0.253,        # Max release distance allowed
    "CLICK_FREEZE_SENSITIVITY": 0.015, # Buffer for "Bunker" deadzone trigger

    # --- DYNAMIC DEADZONE (The Bunker) ---
    "DEADZONE_MAX": 77,             # Pixel radius when stationary
    "DEADZONE_MIN": 77,             # Pixel radius when moving fast (Currently static)
    "DEADZONE_DECAY_VELOCITY": 0.02,# How fast the deadzone shrinks
    
    # =========================================================
    # LAYER 3.5: Z-AXIS PHYSICS (Push to Click)
    # =========================================================
    "PUSH_FORCE_THRESHOLD": 0.04,   # Z-Velocity required to trigger click
    "PUSH_COOLDOWN": 0.5,           # Seconds between push clicks
    "PUSH_RETREAT_THRESHOLD": 0.02, # Reset threshold

    # =========================================================
    # LAYER 4: GESTURE ENGINE RULES (The Referee)
    # =========================================================
    "CTRL_C_THRESHOLD": 1.3,        # Pinky extension ratio
    "DELETE_THUMB_THRESHOLD": 0.35, # Thumb tuck ratio
    "PALM_THUMB_THRESHOLD": 0.45,   # Thumb extension ratio

    # =========================================================
    # LAYER 5: TIMING & OS INTEGRATION
    # =========================================================
    "LOCK_TIME": 1.5,               # Time to hold GEN_Z to lock/unlock
    "DRAG_START_DELAY": 0.35,       # Time to hold pinch before drag starts
    "DOUBLE_CLICK_GAP": 0.2,        # Max time between clicks for DBL Click
    "GESTURE_HISTORY": 7,           # Frames for voting system

    # --- ANALOG CONTROLS ---
    "SCROLL_SPEED_BASE": 3,
    "SCROLL_ACCELERATION": 2.8,
    "SCROLL_DEADZONE": 0.082,

    "VOLUME_DEADZONE": 0.083,
    "VOLUME_SENSITIVITY": 26.0,
    "ZOOM_STEP": 0.094,
    "ZOOM_BRAKE": 0.105,

    # --- SHORTCUTS ---
    "SHORTCUT_SPEED_LIMIT": 0.03,   # Hand must be slow to trigger shortcuts
    "SHORTCUT_COOLDOWN": 2.33,      # Global cooldown
    "COPY_WINDOW": 0.8,             # Max time for Palm -> Fist transition
    "PASTE_WINDOW": 0.6,            # Max time for Fist -> Palm transition
    "SELECT_ALL_WINDOW": 2.08,      # Max time to draw circle
    "SELECT_ALL_DIAMETER": 0.135,   # Min diameter for circle

    # =========================================================
    # WORKSPACE & CALIBRATION
    # =========================================================
    "BOX_WIDTH": 0.62,              # % of camera width usable
    "BOX_HEIGHT": 0.52,             # % of camera height usable
    "BOX_CENTER_X": 0.50,           # Center offset X
    "BOX_CENTER_Y": 0.40,           # Center offset Y (Higher = Lower on screen)
    
    "LEFT_HAND_MODE": False,        # Mirrors X-axis logic
    "CAMERA_INDEX": 0,              # OpenCV device ID
}

def init_environment():
    """
    Creates necessary directories safely at runtime.
    """
    os.makedirs(DATA_DIR / "raw", exist_ok=True)
    os.makedirs(DATA_DIR / "processed", exist_ok=True)
    os.makedirs(PATHS["CLUSTERS_DIR"], exist_ok=True)
    os.makedirs(PATHS["MODELS_DIR"], exist_ok=True)