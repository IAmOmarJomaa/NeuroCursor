"""
NeuroCursor Types (God Tier).
Central definition of Data Contracts to prevent circular imports.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Tuple
import re

# --- PHYSICS TYPES ---
@dataclass
class Point2D:
    x: float
    y: float

class DragTrigger(Enum):
    NONE = auto()
    TIMED = auto()  # Triggered by holding position for X seconds
    MOVED = auto()  # Triggered by moving pixel distance > deadzone

@dataclass
class PhysicsData:
    raw_x: float = 0.0
    raw_y: float = 0.0
    raw_z: float = 0.0
    speed: float = 0.0
    smooth_speed: float = 0.0
    z_velocity: float = 0.0

# --- GESTURE TYPES ---
class Gesture(Enum):
    NOISE = "NOISE_RANDOM"
    POINT = "THE_POINT"
    CLICK = "THE_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    SCROLL = "SCROLL"
    FIST = "FIST"
    PALM = "PALM"
    GEN_Z = "GEN_Z"
    NEXT_OPEN = "NEXT_OPENED"
    NEXT_CLOSED = "NEXT_CLOSED"
    PREV_OPEN = "PREVIOUS_OPENED"
    PREV_CLOSED = "PREVIOUS_CLOSED"
    TAB_M = "TAB_M"
    TAB_L = "TAB_L"
    TAB_R = "TAB_R"
    WIN = "WIN"
    CUT = "CUT"
    CTRL = "CTRL_PINKY"
    ZOOM = "ZOOM"
    VOLUME = "VOLUME"
    THE_SHHH = "THE_SHHH"
    RESTING = "RESTING"
    DELETE_OPEN = "DELETE_OPENED"   
    DELETE_CLOSED = "DELETE_CLOSED" 
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_label(cls, raw_label: str) -> "Gesture":
        if not raw_label or not isinstance(raw_label, str):
            return cls.NOISE
        # Robust regex cleanup
        clean = re.sub(r'^\d+\.?\s*', '', raw_label).strip()
        for member in cls:
            if member.value == clean:
                return member
        return cls.UNKNOWN

@dataclass(frozen=True)
class HandData:
    gesture: Gesture
    confidence: float
    landmarks: Any # Still Any for now (MediaPipe object), but localized.
    
    @property
    def is_valid(self) -> bool:
        return self.gesture != Gesture.NOISE and self.confidence > 0.0