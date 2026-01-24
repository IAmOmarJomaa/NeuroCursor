"""
Handler Context Definition.
Defines the Data Transfer Object (DTO) for the Control Layer.
"""

from src.core.types import HandData, PhysicsData

class HandlerContext:
    """
    A unified context object containing all data required for a Handler to make decisions.
    Wraps Hand Data (Typed), Physics, System State, Action Dispatcher, and Configuration.
    """
    def __init__(self, hand_1: HandData, hand_2: HandData, state, actions, config):
        # 1. Primary Hand Data (The "Active" Hand)
        self.hand = hand_1
        self.label = hand_1.gesture # Convenience alias
        self.lms = hand_1.landmarks # Convenience alias

        # 2. Secondary Hand Data (The "Context" Hand)
        self.hand_2 = hand_2

        # 3. Physics Data (Populated by Controller)
        self.physics = PhysicsData()

        # 4. Global Resources
        self.state = state     # Shared StateManager
        self.actions = actions # ActionDispatcher
        self.config = config   # Master Config Dict
        
    # Properties for Backward Compatibility with old handler logic
    @property
    def raw_x(self): return self.physics.raw_x
    @property
    def raw_y(self): return self.physics.raw_y
    @property
    def speed(self): return self.physics.speed
    @property
    def smooth_speed(self): return self.physics.smooth_speed
    @property
    def z_velocity(self): return self.physics.z_velocity