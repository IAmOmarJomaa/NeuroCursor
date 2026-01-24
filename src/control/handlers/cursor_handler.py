"""
NeuroCursor Cursor Logic V2 (God Tier).
Aligned with Kinematics V2 (Squared Distances & Enums).
"""
import time
from src.control.handlers import HandlerContext
from src.core.kinematics import KinematicsEngine, DragTrigger, Point2D
from src.control.cursor_engine import CursorEngine
from src.core.types import Gesture

class CursorHandler:
    def __init__(self):
        self.physics = KinematicsEngine()
        self.cursor = CursorEngine()
        
        self.click_start_time = 0
        self.click_anchor = Point2D(0.0, 0.0)
        self.is_dragging = False
        
        self.last_click_time = 0 
        self.last_push_time = 0 
        self.last_open_hand_time = 0
        self.last_delete_gesture_time = 0

    def handle(self, ctx: HandlerContext):
        now = time.time()
        
        # --- 0. IRON DOME (Delete Protection) ---
        DELETE_GESTURES = {Gesture.DELETE_OPEN, Gesture.DELETE_CLOSED}
        if ctx.hand.gesture in DELETE_GESTURES:
            self.last_delete_gesture_time = now
            
        if (now - self.last_delete_gesture_time) < 0.5:
            if self.is_dragging: ctx.actions.drag_end()
            ctx.state.is_pinching = False
            ctx.state.is_dragging = False
            self.is_dragging = False
            self.click_start_time = 0
            return 

        # --- 1. CTRL GUARD ---
        if ctx.hand.gesture == Gesture.CTRL: return

        # --- 2. GHOST DRAG GUARD ---
        if ctx.hand.gesture == Gesture.PALM: self.last_open_hand_time = now
        
        is_transitioning = (now - self.last_open_hand_time) < 0.2
        
        # [OPTIMIZATION] Use Squared Distance (Fast)
        pinch_sq_dist = self.physics.get_pinch_sq_dist(ctx.lms)

        if is_transitioning and not self.is_dragging:
            ctx.state.is_pinching = False
        else:
            ctx.state.is_pinching = self.physics.check_pinch_hysteresis(
                pinch_sq_dist, ctx.state.is_pinching, velocity=ctx.speed
            )

        # --- 3. Z-AXIS PUSH ---
        if not ctx.state.is_pinching and not ctx.state.is_dragging:
            self._handle_push_click(ctx)

        # --- 4. THE BUNKER (Deadzone Stabilization) ---
        # "Anti-Jitter" for clicking. Forces stability when pinching starts.
        
        # [FIX] Square the threshold to match the new physics engine
        # (PINCH_START + sensitivity)^2
        freeze_sens = ctx.config.get("CLICK_FREEZE_SENSITIVITY", 0.015)
        pinch_threshold_sq = (ctx.config["PINCH_START"] + freeze_sens) ** 2
        
        # If we are pinching (or close to it) AND not yet dragging...
        if not self.is_dragging and pinch_sq_dist < pinch_threshold_sq:
             # Initialize Anchor if needed
             if self.click_anchor.x == 0.0 and self.click_anchor.y == 0.0: 
                 self.click_anchor = Point2D(ctx.raw_x, ctx.raw_y)
             
             # Calculate squared distance from anchor
             dist_sq = (ctx.raw_x - self.click_anchor.x)**2 + (ctx.raw_y - self.click_anchor.y)**2
             
             # Get Deadzone size for Speed=0 (Maximum Grip)
             deadzone_norm = self.physics.get_dynamic_deadzone(0.0)
             
             # If inside the Bunker...
             if dist_sq < (deadzone_norm ** 2):
                 # Process logic (click timer etc) but DO NOT MOVE MOUSE
                 self._handle_click_drag(ctx) 
                 return # <--- Exits before movement logic

        # --- 5. CLICK/DRAG MACHINE ---
        self._handle_click_drag(ctx)

        # --- 6. MOVEMENT RULES ---
        valid_moves = {Gesture.POINT, Gesture.CLICK}
        should_move = False
        
        if ctx.state.is_dragging:
            # Sticky Drag: Move even if pinch loosens slightly
            # [FIX] Square the config value
            if pinch_sq_dist < (ctx.config["PINCH_STOP"] ** 2): should_move = True
        elif ctx.hand.gesture in valid_moves:
            should_move = True

        if should_move:
            self.cursor.move_mouse(ctx.raw_x, ctx.raw_y, current_speed=ctx.speed)
        else:
            self.click_anchor = Point2D(0.0, 0.0) # Reset anchor

    def _handle_push_click(self, ctx):
        now = time.time()
        if (now - self.last_push_time) < ctx.config.get("PUSH_COOLDOWN", 0.5): return
        threshold = -ctx.config.get("PUSH_FORCE_THRESHOLD", 0.04)
        if ctx.z_velocity < threshold and ctx.smooth_speed < 0.02:
            ctx.actions.click()
            self.last_push_time = now

    def _handle_click_drag(self, ctx):
        now = time.time()
        # Start timer if new pinch
        if ctx.state.is_pinching and self.click_start_time == 0:
            self.click_start_time = now
            self.click_anchor = Point2D(ctx.raw_x, ctx.raw_y)

        # Logic to Transition from Click -> Drag
        if ctx.state.is_pinching and not self.is_dragging:
            # Don't allow drag immediately (debounce)
            calc_speed = ctx.smooth_speed
            if (now - self.click_start_time) < 0.2: calc_speed = 0.0
            
            # [FIX] Use new analyze_drag_intent API with Enums
            current_pos = Point2D(ctx.raw_x, ctx.raw_y)
            drag_state = self.physics.analyze_drag_intent(
                current_pos, self.click_anchor, self.click_start_time, velocity=calc_speed
            )
            
            if drag_state == DragTrigger.MOVED or drag_state == DragTrigger.TIMED:
                ctx.actions.drag_start()
                self.is_dragging = True; ctx.state.is_dragging = True 

        # Logic to Release Click/Drag
        if not ctx.state.is_pinching and self.click_start_time != 0:
            if self.is_dragging:
                ctx.actions.drag_end()
                self.is_dragging = False; ctx.state.is_dragging = False
            else:
                gap = ctx.config["DOUBLE_CLICK_GAP"]
                if (now - self.last_click_time) < gap:
                    ctx.actions.double_click(); self.last_click_time = 0 
                else:
                    ctx.actions.click(); self.last_click_time = now
            self.click_start_time = 0; self.click_anchor = Point2D(0.0, 0.0)