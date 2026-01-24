"""
NeuroCursor Scroll & Volume Logic.
=================================

Handles continuous analog inputs. unlike discrete clicks, these handlers
map a continuous hand position to a continuous output (Scroll velocity).

Key Logic: "Anchor & Delta"
1. When the gesture starts, we save the Y-position as an "Anchor".
2. We calculate the `delta` (Current Y - Anchor Y).
3. If `delta` exceeds a Deadzone, we apply a gain factor to determine velocity.
"""
from src.control.handlers import HandlerContext
from src.core.types import Gesture

class ScrollHandler:
    def __init__(self):
        self.scroll_anchor_y = None
        self.vol_anchor_y = None

    def handle(self, ctx: HandlerContext):
        g = ctx.hand.gesture
        
        # --- SCROLL LOGIC (Fist / Scroll Gesture) ---
        if g == Gesture.SCROLL:
            # Initialize Anchor
            if ctx.state.prev_gesture != Gesture.SCROLL or self.scroll_anchor_y is None:
                self.scroll_anchor_y = ctx.raw_y
            else:
                delta = ctx.raw_y - self.scroll_anchor_y
                deadzone = ctx.config["SCROLL_DEADZONE"]
                
                if abs(delta) > deadzone:
                    # 1. Apply Deadzone Subtraction 
                    # This prevents the scroll from "jumping" the moment it crosses the threshold.
                    # It creates a smooth ramp-up from 0 velocity.
                    sign = 1 if delta > 0 else -1
                    adjusted_delta = delta - (sign * deadzone)
                    
                    # 2. Invert Direction
                    # Hand UP (Low Y) should mean Scroll UP (Positive Delta).
                    # Since Y decreases as we go up, delta is negative. We must invert it.
                    inverted = -adjusted_delta 

                    # 3. Apply Acceleration Gain
                    speed_mult = ctx.config.get("SCROLL_ACCELERATION", 2.8)
                    final_value = inverted * speed_mult
                    
                    ctx.actions.scroll(final_value)
        else:
            self.scroll_anchor_y = None

        # --- VOLUME LOGIC (Volume Gesture) ---
        if g == Gesture.VOLUME:
            if ctx.state.prev_gesture != Gesture.VOLUME or self.vol_anchor_y is None:
                self.vol_anchor_y = ctx.raw_y
            else:
                delta = ctx.raw_y - self.vol_anchor_y
                deadzone = ctx.config["VOLUME_DEADZONE"]
                
                if abs(delta) > deadzone:
                    # High sensitivity for volume to allow quick adjustments
                    sens = ctx.config.get("VOLUME_SENSITIVITY", 26.0)
                    
                    # Invert: Hand UP = Volume UP
                    scaled_change = (-delta) * sens
                    
                    ctx.actions.change_volume(scaled_change)
        else:
            self.vol_anchor_y = None