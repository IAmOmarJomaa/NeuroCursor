"""
NeuroCursor Cognition Engine (The Brain).
========================================

This module handles the mapping from Raw Skeletal Data to Semantic Intents.
It implements a "Hybrid Architecture":
1. **Deep Learning (ONNX):** Probabilistic classification of complex shapes (Fist, Open Hand).
2. **The Referee (Geometric Rules):** Hard-coded constraints to override the AI.

Why The Referee?
Pure AI is "fuzzy". It might confuse a "Point" with a "Click" if the training data is imperfect.
The Referee uses strict Euclidean distance checks to force accuracy (e.g., "If thumb and index
are touching, it IS a click, no matter what the AI says").
"""

import onnxruntime as ort
import numpy as np
import joblib
import math
import os
import logging
from collections import deque, Counter
from typing import Tuple, List, Any, Optional

from src.config import CONFIG, PATHS
from src.hand_utils import pre_process_landmark

class NeuroCursorBrain:
    """
    The Inference Engine.
    
    Attributes:
        session (ort.InferenceSession): The ONNX runtime session.
        history (deque): Rolling buffer for temporal smoothing (Voting system).
        label_map (dict): Int -> String mapping for gesture classes.
    """
    def __init__(self):
        self._load_resources()
        self.history_len = CONFIG.get("GESTURE_HISTORY", 5)
        self.history = deque(maxlen=self.history_len)
        self.confidence_threshold = 0.6

    def _load_resources(self):
        """Loads the ONNX model and Label Map from disk."""
        # 1. Load the Label Map (The Rosetta Stone)
        label_map_path = str(PATHS["MODELS_DIR"] / "label_map.pkl")
        
        if not os.path.exists(label_map_path):
            logging.error(f"❌ Label Map missing: {label_map_path}")
            # Fallback to alphabetical if pickle is missing (Emergency Mode)
            self.label_map = {i: label for i, label in enumerate(CONFIG.get("GESTURE_LABELS", []))}
        else:
            print(f"📂 LOADING LABEL MAP: {label_map_path}")
            self.label_map = joblib.load(label_map_path)

        # 2. Load the Model
        model_path = str(PATHS["MODELS_DIR"] / "neurocursor_model.onnx")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"❌ ONNX Model not found at: {model_path}")
            
        print(f"🧠 LOADING ONNX MODEL: {model_path}")
        # CPU Provider is sufficient for this lightweight model
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _apply_rules(self, ml_label: str, landmarks: List[Any]) -> str:
        """
        The Referee: Applies geometric constraints to vet the AI's decision.
        
        Args:
            ml_label: The raw guess from the Neural Network.
            landmarks: The 21 hand skeleton points.
            
        Returns:
            The final, authoritative gesture label.
        """
        lm = landmarks 
        
        # Calculate Palm Scale (Wrist to Middle Finger MCP) to make thresholds hand-size invariant
        palm_size = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y) or 1.0

        # --- RULE 1: CLICK ACCURACY (Pinch vs Point) ---
        # If the ML says "Point" or "Ctrl", but fingers are touching -> Force CLICK.
        if "CTRL" in ml_label or "CLICK" in ml_label:
            # Thumb Tip (4) to Index Tip (8)
            pinch_dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y) / palm_size
            
            if pinch_dist < 0.15: 
                return "THE_CLICK"

            # Check Pinky Extension for CTRL Gesture
            mid_dist = math.hypot(lm[12].x - lm[0].x, lm[12].y - lm[0].y)
            ring_dist = math.hypot(lm[16].x - lm[0].x, lm[16].y - lm[0].y)
            pinky_dist = math.hypot(lm[20].x - lm[0].x, lm[20].y - lm[0].y)
            
            avg_ext = (mid_dist + ring_dist + pinky_dist) / 3.0
            norm_ext = avg_ext / palm_size
            
            if norm_ext > CONFIG["CTRL_C_THRESHOLD"]: 
                return "CTRL_PINKY"
            else: 
                return "THE_CLICK"

        # --- RULE 2: PALM vs DELETE (Thumb Position) ---
        # Delete gesture usually has the thumb tucked in. Palm has it out.
        thumb_tip = lm[4]
        index_mcp = lm[5]
        thumb_dist = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y)
        norm_thumb = thumb_dist / palm_size
        
        if "PALM" in ml_label and norm_thumb < CONFIG["DELETE_THUMB_THRESHOLD"]:
            return "DELETE_OPENED"
            
        if "DELETE" in ml_label and norm_thumb > CONFIG["PALM_THUMB_THRESHOLD"]:
            return "PALM"

        return ml_label

    def predict(self, raw_landmarks: Any) -> Tuple[str, float]:
        """
        Pipeline: Preprocess -> Infer -> Decode -> Vet -> Smooth.
        
        Args:
            raw_landmarks: MediaPipe NormalizedLandmarkList.
            
        Returns:
            (Label, Confidence Score)
        """
        if self.session is None or not raw_landmarks:
            self.history.clear()
            return "NOISE_RANDOM", 0.0

        # 1. PREPROCESS (Normalize relative to wrist)
        is_left = CONFIG.get("LEFT_HAND_MODE", False)
        # Convert MediaPipe object to flattened float vector (63 floats)
        norm_feats = pre_process_landmark(raw_landmarks, flip_x=is_left)
        
        # 2. INFERENCE (Run ONNX)
        input_data = np.array([norm_feats], dtype=np.float32)
        pred_scores = self.session.run([self.output_name], {self.input_name: input_data})[0][0]
        
        # 3. DECODE (Argmax)
        best_idx = np.argmax(pred_scores)
        conf = float(np.max(pred_scores))
        
        raw_label = self.label_map.get(best_idx, "UNKNOWN")
        
        # Clean up legacy label artifacts (e.g., "1. THE_POINT" -> "THE_POINT")
        if "." in raw_label: 
            raw_label = raw_label.split(".")[-1].strip()
        
        # 4. REFEREE (Geometric Override)
        final_label = self._apply_rules(raw_label, raw_landmarks)
        
        # 5. TEMPORAL SMOOTHING (Voting)
        # Prevents "flickering" between two similar gestures
        self.history.append(final_label)
        smoothed_label = Counter(self.history).most_common(1)[0][0]

        if conf > self.confidence_threshold:
            return smoothed_label, conf
        else:
            return "NOISE_RANDOM", 0.0