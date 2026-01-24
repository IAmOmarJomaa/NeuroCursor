"""
NeuroCursor Shortcut Logic.
==========================

Handles complex, multi-stage gestures like Copy, Paste, and Select All.
Uses a state machine to track sequences (e.g., Palm -> Fist = Copy).

Optimization:
Uses `collections.deque` for the "Select All" circle path. Appending to a 
standard list is O(1), but popping from the front is O(N). Deque makes both O(1).
"""
import time
from collections import deque 
from src.control.handlers import HandlerContext
from src.core.types import Gesture
from src.core.kinematics import KinematicsEngine

class ShortcutHandler:
    def __init__(self):
        # Delete State Machine (Open -> Closed -> Open)
        self.delete_stage = 0 
        self.last_delete_stage_time = 0
        self.last_fire_delete = 0
        
        # Shortcut Timers
        self.last_palm_time = 0
        self.last_fist_time = 0
        
        # Circle Detection Buffer (Rolling window of 60 points)
        self.palm_path = deque(maxlen=60) 
        self.palm_start_time = 0
        
        # Cooldowns & Flags
        self.last_valid_palm = 0 
        self.last_copy_time = 0
        self.last_paste_time = 0
        self.last_select_all_time = 0
        self.fist_handled = False # Dirty flag to prevent Copy/Paste loops
        
        self.physics = KinematicsEngine()

    def handle(self, ctx: HandlerContext):
        now = time.time()
        g = ctx.hand.gesture
        cooldown = ctx.config["SHORTCUT_COOLDOWN"]

        # 1. DELETE CYCLE (ABSOLUTE PRIORITY)
        # Sequence: Open -> Closed -> Open (Grab and throw away)
        # Timeout: If stages take too long (>1.0s), reset.
        if self.delete_stage > 0 and (now - self.last_delete_stage_time) > 1.0: 
            self.delete_stage = 0

        if self.delete_stage == 0:
            if g == Gesture.DELETE_OPEN: 
                self.delete_stage = 1; self.last_delete_stage_time = now
        elif self.delete_stage == 1:
            if g == Gesture.DELETE_CLOSED: 
                self.delete_stage = 2; self.last_delete_stage_time = now
        elif self.delete_stage == 2:
            if g == Gesture.DELETE_OPEN:
                if (now - self.last_fire_delete) > 1:
                    ctx.actions.delete()
                    self.last_fire_delete = now; self.delete_stage = 0

        # 2. SELECT ALL / PASTE GROUP
        CIRCLE_GROUP = {Gesture.PALM, Gesture.DELETE_OPEN, Gesture.CTRL}
        
        if g in CIRCLE_GROUP:
            if g == Gesture.PALM: self.last_valid_palm = now 

            # PASTE LOGIC (Fist -> Palm)
            if g == Gesture.PALM and not ctx.state.is_dragging:
                if (now - self.last_fist_time) < ctx.config["PASTE_WINDOW"]:
                    if not self.fist_handled and (now - self.last_paste_time) > cooldown:
                        ctx.actions.paste()
                        self.last_paste_time = now; self.fist_handled = True 

            # SELECT ALL LOGIC (Circle Drawing)
            # Reset path if too old
            if not self.palm_path or (now - self.palm_start_time) > ctx.config["SELECT_ALL_WINDOW"]:
                self.palm_path.clear()
                self.palm_start_time = now
            
            self.palm_path.append((ctx.raw_x, ctx.raw_y))

            # Only check for circle if we have enough points (>15)
            if len(self.palm_path) > 15 and (now - self.palm_start_time) < ctx.config["SELECT_ALL_WINDOW"]:
                if self.physics.check_full_circle(list(self.palm_path)):
                     if (now - self.last_select_all_time) > cooldown:
                         ctx.actions.select_all()
                         self.last_select_all_time = now; self.palm_path.clear()
            
            self.last_palm_time = now
        else:
            # Clear path if hand is not in "Writing" mode
            if (now - self.last_palm_time) > 0.3: self.palm_path.clear()

        # 3. SAFETY BLOCK (Speed Limit)
        # Prevent accidental triggers during fast cursor movement
        if ctx.speed > ctx.config["SHORTCUT_SPEED_LIMIT"]: return
        if ctx.state.is_dragging or ctx.state.is_pinching: return

        # 4. COPY / CUT / RIGHT CLICK
        if g in {Gesture.FIST, Gesture.PREV_CLOSED}:
            # Reset handled flag if fist has been held for a while
            if (now - self.last_fist_time) > 0.5: self.fist_handled = False
            
            # COPY LOGIC (Palm -> Fist)
            if (now - self.last_valid_palm) < ctx.config["COPY_WINDOW"]: 
                if not self.fist_handled and (now - self.last_copy_time) > cooldown:
                    ctx.actions.copy()
                    self.last_copy_time = now; self.last_valid_palm = 0; self.fist_handled = True
            self.last_fist_time = now

        elif g == Gesture.CUT:
            if (now - self.last_valid_palm) < ctx.config["COPY_WINDOW"]:
                 if (now - self.last_copy_time) > cooldown:
                     ctx.actions.cut()
                     self.last_copy_time = now; self.last_valid_palm = 0
        
        elif g == Gesture.RIGHT_CLICK:
            if ctx.state.prev_gesture != Gesture.RIGHT_CLICK:
                ctx.actions.right_click()