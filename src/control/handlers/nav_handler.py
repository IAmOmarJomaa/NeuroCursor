"""NeuroCursor Navigation Handler (Browser/Alt-Tab)."""
import time
from src.control.handlers import HandlerContext
from src.core.types import Gesture

class NavigationHandler:
    def __init__(self):
        self.last_trigger_time = 0
        self.tab_active = False
        self.tab_cooldown = 0

    def handle(self, ctx: HandlerContext):
        if ctx.state.is_locked: return

        g = ctx.hand.gesture
        prev = ctx.state.prev_gesture
        now = time.time()
        cooldown = 0.5 

        # 1. BROWSER NAVIGATION (Back/Forward)
        if prev == Gesture.NEXT_CLOSED and g == Gesture.NEXT_OPEN:
            if (now - self.last_trigger_time) > cooldown:
                ctx.actions.nav_forward(); self.last_trigger_time = now

        elif prev == Gesture.PREV_CLOSED and g == Gesture.PREV_OPEN:
            if (now - self.last_trigger_time) > cooldown:
                ctx.actions.nav_back(); self.last_trigger_time = now
        
        # 2. TASK VIEW (Win+Tab)
        elif prev != Gesture.WIN and g == Gesture.WIN:
             if (now - self.last_trigger_time) > 1.0: 
                ctx.actions.open_history(); self.last_trigger_time = now

        # 3. ALT TAB SWITCHER
        if g == Gesture.TAB_M:
            if not self.tab_active:
                ctx.actions.alt_tab_start()
                self.tab_active = True; self.tab_cooldown = now + 0.5

        elif self.tab_active:
            if (now > self.tab_cooldown):
                if g == Gesture.TAB_R and prev != Gesture.TAB_R:
                    ctx.actions.alt_tab_right(); self.tab_cooldown = now + 0.3
                elif g == Gesture.TAB_L and prev != Gesture.TAB_L:
                    ctx.actions.alt_tab_left(); self.tab_cooldown = now + 0.3

            if g not in {Gesture.TAB_M, Gesture.TAB_L, Gesture.TAB_R}:
                ctx.actions.alt_tab_end(); self.tab_active = False