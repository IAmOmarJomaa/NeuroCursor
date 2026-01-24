import unittest
import time
from src.core.state_manager import StateManager
from src.core.types import Gesture

class TestStateManager(unittest.TestCase):
    def setUp(self):
        """Runs before every test."""
        self.state = StateManager()

    def test_initial_state(self):
        """Verify the system starts in a safe, locked state."""
        self.assertTrue(self.state.is_locked, "System should start LOCKED")
        self.assertEqual(self.state.curr_gesture, Gesture.RESTING)

    def test_gesture_update(self):
        """Verify gesture history updates correctly."""
        self.state.update_gesture(Gesture.POINT)
        self.assertEqual(self.state.curr_gesture, Gesture.POINT)
        self.assertEqual(self.state.prev_gesture, Gesture.RESTING) # History check

        self.state.update_gesture(Gesture.CLICK)
        self.assertEqual(self.state.curr_gesture, Gesture.CLICK)
        self.assertEqual(self.state.prev_gesture, Gesture.POINT)

    def test_click_state_reset(self):
        """Verify click anchors are cleared properly."""
        self.state.click_start_time = 123456789
        self.state.click_anchor = (100, 200)
        
        self.state.reset_click_state()
        
        self.assertEqual(self.state.click_start_time, 0)
        self.assertEqual(self.state.click_anchor, (0, 0))

if __name__ == '__main__':
    unittest.main()