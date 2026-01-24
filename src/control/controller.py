"""
NeuroCursor Controller (God Tier).
Acts as the central nervous system. Now exposes physics state for the HUD.
"""

import math
from src.config import CONFIG
from src.core.state_manager import StateManager
from src.control.action_dispatcher import ActionDispatcher
from src.control.handlers import HandlerContext
from src.core.types import Gesture, HandData, PhysicsData

# HANDLERS
from src.control.handlers.system_handler import SystemHandler
from src.control.handlers.cursor_handler import CursorHandler
from src.control.handlers.scroll_handler import ScrollHandler
from src.control.handlers.nav_handler import NavigationHandler
from src.control.handlers.shortcut_handler import ShortcutHandler

class NeuroController:
    def __init__(self):
        self.state = StateManager()
        self.actions = ActionDispatcher()
        
        # Handlers
        self.system_handler = SystemHandler()
        self.cursor_handler = CursorHandler()
        self.scroll_handler = ScrollHandler()
        self.nav_handler = NavigationHandler()
        self.shortcut_handler = ShortcutHandler()
        
        # Physics State (Persisted for UI)
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_z = 0.0
        self.smoothed_speed = 0.0
        
        # [NEW] Expose physics for the HUD to render the 'Bunker'
        self.last_physics: PhysicsData = PhysicsData()
        
        # Visualizer Hooks
        self.show_box = self.cursor_handler.cursor.show_box
        self.BOX_WIDTH = self.cursor_handler.cursor.box_w
        self.BOX_HEIGHT = self.cursor_handler.cursor.box_h
        self.BOX_CENTER_X = self.cursor_handler.cursor.box_cx
        self.BOX_CENTER_Y = self.cursor_handler.cursor.box_cy

    def toggle_box(self):
        self.cursor_handler.cursor.toggle_visuals()
        self.show_box = self.cursor_handler.cursor.show_box

    def _pack_hand(self, raw_data) -> HandData:
        if not raw_data:
            return HandData(Gesture.NOISE, 0.0, None)
        lbl, conf, lms = raw_data
        return HandData(Gesture.from_label(lbl), conf, lms)

    def process(self, raw_h1, raw_h2=None):
        # 1. Normalize
        h1 = self._pack_hand(raw_h1)
        h2 = self._pack_hand(raw_h2)
        
        # 2. Smart Swap (Left hand control logic)
        if h2.is_valid:
            is_passive = h1.gesture in {Gesture.CTRL, Gesture.FIST}
            is_active = h2.gesture in {Gesture.POINT, Gesture.CLICK}
            if is_passive and is_active:
                h1, h2 = h2, h1 
        
        # 3. Physics Calculation
        if h1.landmarks:
            curr_x = h1.landmarks.landmark[8].x
            curr_y = h1.landmarks.landmark[8].y
            curr_z = h1.landmarks.landmark[8].z
        else:
            curr_x, curr_y, curr_z = 0, 0, 0
            
        inst_speed = math.hypot(curr_x - self.prev_x, curr_y - self.prev_y)
        z_delta = curr_z - self.prev_z
        self.smoothed_speed = (0.2 * inst_speed) + (0.8 * self.smoothed_speed)
        
        self.prev_x, self.prev_y, self.prev_z = curr_x, curr_y, curr_z
        
        # 4. Context Creation
        ctx = HandlerContext(h1, h2, self.state, self.actions, CONFIG)
        
        # Populate Physics
        ctx.physics.raw_x = curr_x
        ctx.physics.raw_y = curr_y
        ctx.physics.raw_z = curr_z
        ctx.physics.speed = inst_speed
        ctx.physics.smooth_speed = self.smoothed_speed
        ctx.physics.z_velocity = z_delta
        
        # [NEW] Save for HUD
        self.last_physics = ctx.physics
        
        # 5. Execution Pipeline
        self.system_handler.handle(ctx)
        
        if not self.state.is_locked:
            self.cursor_handler.handle(ctx)
            self.nav_handler.handle(ctx)
            self.scroll_handler.handle(ctx)
            self.shortcut_handler.handle(ctx)
            
        # 6. State Update
        self.state.update_gesture(h1.gesture)