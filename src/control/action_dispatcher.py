"""
NeuroCursor Action Dispatcher (The Actuator).
============================================

This module implements the Command Pattern to decouple the high-level intent
("Copy this") from the low-level OS implementation ("Ctrl+C via win32api").

Features:
- **Platform Agnostic:** Detects OS at runtime.
- **Graceful Fallback:** Uses a Mock Dispatcher if `pywin32` is missing (CI/CD safe).
- **State Tracking:** Manages modifier key states (e.g., holding Ctrl).
"""

import platform
import sys
from typing import Optional
from src.core.interfaces import IOsActionDispatcher

# =============================================================================
# WINDOWS BACKEND (Production)
# =============================================================================
class WindowsActionDispatcher(IOsActionDispatcher):
    """
    Concrete implementation using the Windows API (pywin32).
    Directly injects hardware events into the OS input stream.
    """
    def __init__(self):
        try:
            import win32api
            import win32con
            self._w32 = win32api
            self._con = win32con
        except ImportError:
            print("❌ CRITICAL: 'pywin32' not found. Install it via pip.")
            sys.exit(1)
            
        self.ctrl_held = False
        self.vol_accumulator = 0.0

    # --- MOUSE PRIMITIVES ---
    def move_cursor(self, x: int, y: int) -> None:
        """Absolute cursor positioning."""
        self._w32.SetCursorPos((x, y))

    def click(self) -> None:
        """Synthesizes a full Left Click (Down + Up)."""
        self._w32.mouse_event(self._con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self._w32.mouse_event(self._con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def double_click(self) -> None:
        """Synthesizes a Double Click."""
        self.click()
        self.click()

    def right_click(self) -> None:
        """Synthesizes a Right Click."""
        self._w32.mouse_event(self._con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        self._w32.mouse_event(self._con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def drag_start(self) -> None:
        """Holds Left Button Down."""
        self._w32.mouse_event(self._con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def drag_end(self) -> None:
        """Releases Left Button."""
        self._w32.mouse_event(self._con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def scroll(self, delta: int) -> None:
        """
        Vertical Scroll.
        Args:
            delta: Positive = Up, Negative = Down.
                   120 units = 1 physical notch.
        """
        amount = int(delta * 120)
        self._w32.mouse_event(self._con.MOUSEEVENTF_WHEEL, 0, 0, amount, 0)

    def zoom(self, direction: int) -> None:
        """
        Zoom (Ctrl + Scroll).
        Temporarily injects CTRL if it's not already held.
        """
        need_ctrl = not self.ctrl_held
        if need_ctrl: 
            self._w32.keybd_event(self._con.VK_CONTROL, 0, 0, 0)
        
        self.scroll(direction)
        
        if need_ctrl: 
            self._w32.keybd_event(self._con.VK_CONTROL, 0, self._con.KEYEVENTF_KEYUP, 0)

    # --- KEYBOARD PRIMITIVES ---
    def send_key(self, key_name: str) -> None:
        """Press and Release a single key."""
        code = self._map_key(key_name)
        if code:
            self._w32.keybd_event(code, 0, 0, 0)
            self._w32.keybd_event(code, 0, self._con.KEYEVENTF_KEYUP, 0)

    def send_hotkey(self, modifier: str, key: str) -> None:
        """Executes a combo like Ctrl+C."""
        mod_code = self._map_key(modifier)
        key_code = self._map_key(key)
        if mod_code and key_code:
            self._w32.keybd_event(mod_code, 0, 0, 0)         # Mod Down
            self._w32.keybd_event(key_code, 0, 0, 0)         # Key Down
            self._w32.keybd_event(key_code, 0, self._con.KEYEVENTF_KEYUP, 0) # Key Up
            self._w32.keybd_event(mod_code, 0, self._con.KEYEVENTF_KEYUP, 0) # Mod Up

    def set_ctrl_state(self, active: bool) -> None:
        """Toggles the Control key for multi-select."""
        if active and not self.ctrl_held:
            self._w32.keybd_event(self._con.VK_CONTROL, 0, 0, 0)
            self.ctrl_held = True
        elif not active and self.ctrl_held:
            self._w32.keybd_event(self._con.VK_CONTROL, 0, self._con.KEYEVENTF_KEYUP, 0)
            self.ctrl_held = False

    # --- SEMANTIC COMMANDS (Higher Level) ---
    def copy(self): self.send_hotkey("ctrl", "c")
    def paste(self): self.send_hotkey("ctrl", "v")
    def cut(self): self.send_hotkey("ctrl", "x")
    def select_all(self): self.send_hotkey("ctrl", "a")
    def delete(self): self.send_key("delete")
    def nav_back(self): self.send_key("browser_back")
    def nav_forward(self): self.send_key("browser_forward")
    def press_win_key(self): self.send_key("win")
    def open_history(self): self.send_hotkey("win", "tab")
    def toggle_mute(self): self.send_key("mute")

    # --- COMPLEX STATES ---
    def alt_tab_start(self):
        """Holds ALT and taps TAB."""
        self._w32.keybd_event(self._con.VK_MENU, 0, 0, 0)
        self.send_key("tab")

    def alt_tab_right(self): self.send_key("right")
    def alt_tab_left(self): self.send_key("left")

    def alt_tab_end(self):
        """Releases ALT to select the window."""
        self._w32.keybd_event(self._con.VK_MENU, 0, self._con.KEYEVENTF_KEYUP, 0)

    # --- SYSTEM ---
    def change_volume(self, delta: float) -> None:
        """
        Changes volume using an accumulator for smooth analog control.
        Inputs < 1.0 are accumulated until they trigger a discrete 'step'.
        """
        self.vol_accumulator += delta
        MAX_STEPS = 5
        
        # Threshold check
        if self.vol_accumulator >= 1:
            steps = min(int(self.vol_accumulator), MAX_STEPS)
            for _ in range(steps): self.send_key("vol_up")
            self.vol_accumulator -= steps
        elif self.vol_accumulator <= -1:
            steps = min(int(abs(self.vol_accumulator)), MAX_STEPS)
            for _ in range(steps): self.send_key("vol_down")
            self.vol_accumulator += steps

    def _map_key(self, name: str) -> Optional[int]:
        """Maps string names to Windows Virtual Key Codes."""
        name = name.lower()
        mapping = {
            "enter": self._con.VK_RETURN,
            "esc": self._con.VK_ESCAPE,
            "delete": self._con.VK_DELETE,
            "mute": self._con.VK_VOLUME_MUTE,
            "vol_up": self._con.VK_VOLUME_UP,
            "vol_down": self._con.VK_VOLUME_DOWN,
            "win": self._con.VK_LWIN,
            "tab": self._con.VK_TAB,
            "alt": self._con.VK_MENU,
            "ctrl": self._con.VK_CONTROL,
            "left": self._con.VK_LEFT,
            "right": self._con.VK_RIGHT,
            "c": ord('C'), "v": ord('V'), "x": ord('X'), "a": ord('A'),
            "browser_back": 0xA6,
            "browser_forward": 0xA7,
        }
        if len(name) == 1: return ord(name.upper())
        return mapping.get(name)

# =============================================================================
# MOCK BACKEND (Testing / Non-Windows)
# =============================================================================
class MockActionDispatcher(IOsActionDispatcher):
    """
    Silent implementation for Unit Tests or Linux/Mac environments.
    Prints actions to stdout instead of executing them.
    """
    def move_cursor(self, x, y): pass
    def click(self): print("[MOCK] Click")
    def double_click(self): print("[MOCK] Double Click")
    def right_click(self): print("[MOCK] Right Click")
    def drag_start(self): print("[MOCK] Drag Start")
    def drag_end(self): print("[MOCK] Drag End")
    def scroll(self, delta): print(f"[MOCK] Scroll {delta}")
    def zoom(self, d): print(f"[MOCK] Zoom {d}")
    def send_key(self, k): print(f"[MOCK] Key {k}")
    def send_hotkey(self, m, k): print(f"[MOCK] Hotkey {m}+{k}")
    def set_ctrl_state(self, a): print(f"[MOCK] Ctrl {a}")
    def copy(self): print("[MOCK] Copy")
    def paste(self): print("[MOCK] Paste")
    def cut(self): print("[MOCK] Cut")
    def select_all(self): print("[MOCK] Select All")
    def delete(self): print("[MOCK] Delete")
    def nav_back(self): print("[MOCK] Back")
    def nav_forward(self): print("[MOCK] Fwd")
    def open_history(self): print("[MOCK] Task View")
    def press_win_key(self): print("[MOCK] Win Key")
    def alt_tab_start(self): print("[MOCK] Alt-Tab Start")
    def alt_tab_right(self): print("[MOCK] Alt-Tab Right")
    def alt_tab_left(self): print("[MOCK] Alt-Tab Left")
    def alt_tab_end(self): print("[MOCK] Alt-Tab End")
    def change_volume(self, d): pass
    def toggle_mute(self): print("[MOCK] Mute")

def ActionDispatcher() -> IOsActionDispatcher:
    """Factory method to return the correct Dispatcher for the current OS."""
    current_os = platform.system()
    if current_os == "Windows": 
        return WindowsActionDispatcher()
    else: 
        print(f"⚠️ OS '{current_os}' detected. Using MOCK Action Dispatcher.")
        return MockActionDispatcher()