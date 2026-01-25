"""NeuroCursor System Handler (Locking/Exit)."""
import time
import sys
from src.control.handlers import HandlerContext
from src.core.types import Gesture

class SystemHandler:
    def __init__(self):
        self.lock_start_time = 0
        self.exit_timer = 0
        self.post_toggle_cooldown = 0
        self.prev_zoom_dist = None
        self.ctrl_timer_start = 0
        self.ctrl_gesture_handled = False
        self.last_mute_time = 0

    def handle(self, ctx: HandlerContext):
        now = time.time()
        
        # 1. MASTER TOGGLE (GEN_Z)
        if ctx.hand.gesture == Gesture.GEN_Z:
            if now > self.post_toggle_cooldown:
                if self.lock_start_time == 0: 
                    self.lock_start_time = now
                elif (now - self.lock_start_time) > ctx.config["LOCK_TIME"]:
                    ctx.state.is_locked = not ctx.state.is_locked
                    print(f"🔒 SYSTEM {'LOCKED' if ctx.state.is_locked else 'UNLOCKED'}")
                    self.post_toggle_cooldown = now + 1.5 
                    self.lock_start_time = 0
        else:
            self.lock_start_time = 0

        # 2. LOCKED STATE LOGIC
        if ctx.state.is_locked:
            if ctx.hand.gesture == Gesture.PALM:
                if self.exit_timer == 0: 
                    self.exit_timer = now
                elif (now - self.exit_timer) > 1.0:
                    print("👋 WAVE DETECTED - EXITING...")
                    sys.exit(0)
            else:
                self.exit_timer = 0
            return 

        # 3. UNLOCKED STATE LOGIC
        # Ctrl Modifier Toggle
        if ctx.hand.gesture == Gesture.CTRL:
            if self.ctrl_timer_start == 0: self.ctrl_timer_start = now
            elif (now - self.ctrl_timer_start) > 0.3 and not self.ctrl_gesture_handled:
                ctx.actions.set_ctrl_state(not ctx.actions.ctrl_held)
                self.ctrl_gesture_handled = True
        else:
            self.ctrl_timer_start = 0
            self.ctrl_gesture_handled = False

        # Dual Hand Zoom
        if ctx.hand_2 and ctx.hand_2.is_valid:
            ZOOM_TRIGGERS = {Gesture.ZOOM, Gesture.POINT}
            if ctx.hand.gesture in ZOOM_TRIGGERS and ctx.hand_2.gesture in ZOOM_TRIGGERS:
                x1, y1 = ctx.lms.landmark[8].x, ctx.lms.landmark[8].y
                x2, y2 = ctx.hand_2.landmarks.landmark[8].x, ctx.hand_2.landmarks.landmark[8].y
                dist = ((x2-x1)**2 + (y2-y1)**2)**0.5 # Euclidean
                
                if self.prev_zoom_dist is not None:
                    delta = dist - self.prev_zoom_dist
                    if abs(delta) > ctx.config["ZOOM_STEP"] and abs(delta) > ctx.config["ZOOM_BRAKE"]:
                        direction = 1 if delta > 0 else -1
                        ctx.actions.zoom(direction)
                        self.prev_zoom_dist = dist
                else:
                    self.prev_zoom_dist = dist
            else:
                self.prev_zoom_dist = None
        else:
            self.prev_zoom_dist = None
        # Mute Toggle (The Shhh)
        if ctx.hand.gesture == Gesture.THE_SHHH:
            # 1.0s Cooldown to prevent rapid toggling
            if (now - self.last_mute_time) > 3.0:
                print("🤫 MUTE TOGGLED")
                ctx.actions.toggle_mute()
                self.last_mute_time = now