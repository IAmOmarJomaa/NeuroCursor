import ctypes

class WindowsMouse:
    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.screen_w = self.user32.GetSystemMetrics(0)
        self.screen_h = self.user32.GetSystemMetrics(1)
        
    def move(self, x, y):
        # Clip to screen bounds to prevent crashes
        x = max(0, min(x, self.screen_w))
        y = max(0, min(y, self.screen_h))
        self.user32.SetCursorPos(int(x), int(y))

    def down(self): self.user32.mouse_event(0x0002, 0, 0, 0, 0) # Left Down
    def up(self): self.user32.mouse_event(0x0004, 0, 0, 0, 0)   # Left Up
    def right(self): 
        self.user32.mouse_event(0x0008, 0, 0, 0, 0) # Right Down
        self.user32.mouse_event(0x0010, 0, 0, 0, 0) # Right Up
    def scroll(self, val): self.user32.mouse_event(0x0800, 0, 0, val, 0)