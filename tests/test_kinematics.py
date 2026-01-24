import unittest
import time
from src.core.kinematics import KinematicsEngine, DragTrigger, Point2D
from src.config import CONFIG

# Mock for MediaPipe Landmark structure
class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class MockHand:
    def __init__(self, t_x, t_y, i_x, i_y):
        self.landmark = [None] * 21
        self.landmark[4] = MockLandmark(t_x, t_y) # Thumb
        self.landmark[8] = MockLandmark(i_x, i_y) # Index

class TestKinematics(unittest.TestCase):
    def setUp(self):
        self.engine = KinematicsEngine()
        # Enforce known config for deterministic testing
        CONFIG["PINCH_START"] = 0.05
        CONFIG["DRAG_START_DELAY"] = 0.5
        CONFIG["DEADZONE_MAX"] = 50
        CONFIG["DEADZONE_MIN"] = 10
        # Assume screen ref is 1920 for normalized conversion
        self.px_to_norm = 1.0 / 1920.0

    def test_pinch_distance_sq(self):
        """Verify squared distance calculation."""
        # Distance of 0.1 on X axis -> Sq Dist should be 0.01
        hand = MockHand(0.0, 0.0, 0.1, 0.0)
        sq_dist = self.engine.get_pinch_sq_dist(hand)
        self.assertAlmostEqual(sq_dist, 0.01)

    def test_drag_trigger_deadzone(self):
        """Verify Deadzone logic returns MOVED only when boundary crossed."""
        # [FIXED] Use CURRENT time so we don't accidentally trigger 'TIMED'
        start_time = time.time() 
        
        # Anchor at 0.5, 0.5
        anchor = Point2D(0.5, 0.5)
        
        # 1. Test INSIDE Deadzone (Should be NONE)
        # Move 10 pixels away (approx 0.005 norm)
        current = Point2D(0.5 + (10 * self.px_to_norm), 0.5)
        
        result = self.engine.analyze_drag_intent(current, anchor, start_time, velocity=0.0)
        self.assertEqual(result, DragTrigger.NONE)

        # 2. Test OUTSIDE Deadzone (Should be MOVED)
        # Move 60 pixels away (Max is 50px)
        current = Point2D(0.5 + (60 * self.px_to_norm), 0.5)
        
        result = self.engine.analyze_drag_intent(current, anchor, start_time, velocity=0.0)
        self.assertEqual(result, DragTrigger.MOVED)

    def test_drag_trigger_timed(self):
        """Verify holding still for too long triggers TIMED."""
        # Set start time to 1 second ago (Greater than 0.5s delay)
        start_time = time.time() - 1.0
        anchor = Point2D(0.5, 0.5)
        current = Point2D(0.5, 0.5) # Still inside deadzone

        result = self.engine.analyze_drag_intent(current, anchor, start_time, velocity=0.0)
        self.assertEqual(result, DragTrigger.TIMED)

    def test_dynamic_deadzone_shrink(self):
        """Verify deadzone gets smaller as velocity increases."""
        # At velocity 0, deadzone should be MAX (50px)
        dz_slow = self.engine.get_dynamic_deadzone(0.0)
        
        # At high velocity, deadzone should be MIN (10px)
        dz_fast = self.engine.get_dynamic_deadzone(1.0) 
        
        self.assertGreater(dz_slow, dz_fast)

if __name__ == '__main__':
    unittest.main()