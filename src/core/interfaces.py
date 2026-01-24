"""
NeuroCursor Core Interfaces.
Defines the abstract contracts for system interaction.
"""

from abc import ABC, abstractmethod

class IOsActionDispatcher(ABC):
    """
    Abstract Protocol for OS Input Injection.
    """
    
    # --- MOUSE ---
    @abstractmethod
    def move_cursor(self, x: int, y: int) -> None: pass
    @abstractmethod
    def click(self) -> None: pass
    @abstractmethod
    def double_click(self) -> None: pass
    @abstractmethod
    def right_click(self) -> None: pass
    @abstractmethod
    def drag_start(self) -> None: pass
    @abstractmethod
    def drag_end(self) -> None: pass
    @abstractmethod
    def scroll(self, delta: int) -> None: pass
    @abstractmethod
    def zoom(self, direction: int) -> None: pass

    # --- KEYBOARD PRIMITIVES ---
    @abstractmethod
    def send_key(self, key_name: str) -> None: pass
    @abstractmethod
    def send_hotkey(self, modifier: str, key: str) -> None: pass
    @abstractmethod
    def set_ctrl_state(self, active: bool) -> None: pass

    # --- SEMANTIC COMMANDS ---
    @abstractmethod
    def copy(self) -> None: pass
    @abstractmethod
    def paste(self) -> None: pass
    @abstractmethod
    def cut(self) -> None: pass
    @abstractmethod
    def select_all(self) -> None: pass
    @abstractmethod
    def delete(self) -> None: pass
    
    # --- NAVIGATION ---
    @abstractmethod
    def nav_back(self) -> None: pass
    @abstractmethod
    def nav_forward(self) -> None: pass
    @abstractmethod
    def open_history(self) -> None: pass 
    @abstractmethod
    def press_win_key(self) -> None: pass

    # --- COMPLEX STATES ---
    @abstractmethod
    def alt_tab_start(self) -> None: pass
    @abstractmethod
    def alt_tab_right(self) -> None: pass
    @abstractmethod
    def alt_tab_left(self) -> None: pass
    @abstractmethod
    def alt_tab_end(self) -> None: pass

    # --- SYSTEM ---
    # [FIX] Renamed to match ScrollHandler
    @abstractmethod
    def change_volume(self, delta: float) -> None: pass
    @abstractmethod
    def toggle_mute(self) -> None: pass