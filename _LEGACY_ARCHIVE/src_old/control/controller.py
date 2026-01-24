import math
import time
from src.config import CONFIG
from src.core.state_manager import StateManager
from src.core.kinematics import KinematicsEngine
from src.control.action_dispatcher import ActionDispatcher

# Handlers
from src.control.handlers import HandlerContext
from src.control.handlers.cursor_handler import CursorHandler
from src.control.handlers.scroll_handler import ScrollHandler
from src.control.handlers.nav_handler import NavigationHandler
from src.control.handlers.shortcut_handler import ShortcutHandler
from src.control.handlers.system_handler import SystemHandler

class NeuroController:
    def __init__(self):
        # 1. Systems
        self.state = StateManager()
        self.physics = KinematicsEngine()
        self.actions = ActionDispatcher()
        
        # 2. Handlers
        self.cursor_handler = CursorHandler()
        self.scroll_handler = ScrollHandler()
        self.nav_handler = NavigationHandler()
        self.shortcut_handler = ShortcutHandler()
        self.system_handler = SystemHandler()
        
        # Speed Memory
        self.prev_raw_x = 0
        self.prev_raw_y = 0

    # --- 🔧 FIX: DYNAMIC PROPERTIES FOR MAIN.PY ---
    # This prevents the "AttributeError" because it always pulls the REAL value from the engine.
    @property
    def show_box(self): return self.cursor_handler.cursor.show_box
    @property
    def BOX_WIDTH(self): return self.cursor_handler.cursor.box_w
    @property
    def BOX_HEIGHT(self): return self.cursor_handler.cursor.box_h
    @property
    def BOX_CENTER_X(self): return self.cursor_handler.cursor.box_cx
    @property
    def BOX_CENTER_Y(self): return self.cursor_handler.cursor.box_cy

    def toggle_box(self): self.cursor_handler.cursor.toggle_visuals()

    def process(self, hand_1, hand_2=None):
        ctx = HandlerContext(hand_1, hand_2, self.state, self.actions, CONFIG)
        
        # 1. CALCULATE SPEED (Crucial for Magnet/Shortcuts)
        if ctx.lms:
            ctx.raw_x = ctx.lms.landmark[8].x 
            ctx.raw_y = ctx.lms.landmark[8].y 
            
            dx = ctx.raw_x - self.prev_raw_x
            dy = ctx.raw_y - self.prev_raw_y
            ctx.speed = math.hypot(dx, dy)
            
            self.prev_raw_x = ctx.raw_x
            self.prev_raw_y = ctx.raw_y
        else:
            ctx.speed = 0

        # 2. HANDLER PIPELINE
        
        # A. SYSTEM (Lock/Unlock) - Highest Priority
        self.system_handler.handle(ctx)
        if self.state.is_locked:
            self.state.update_gesture(ctx.label) # Update history so we can unlock later
            return

        # B. CURSOR (Move/Click)
        self.cursor_handler.handle(ctx)
        
        # C. TOOLS (Scroll, Nav, Shortcuts)
        self.scroll_handler.handle(ctx)
        self.nav_handler.handle(ctx)
        self.shortcut_handler.handle(ctx)
        
        # D. HISTORY UPDATE
        self.state.update_gesture(ctx.label)