import math
import time

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: Decrease this to remove more jitter (slower).
        beta: Increase this to reduce lag during fast movement.
        """
        self.frequency = 0.0
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """
        t: Current timestamp
        x: Current raw value (float)
        Returns: Smoothed value
        """
        t_e = t - self.t_prev
        
        # Prevent division by zero or negative time
        if t_e <= 0.0:
            return self.x_prev

        # 1. Estimate the Velocity (Jittery derivative)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 2. Calculate the Cutoff Frequency (The "Magic")
        # As speed (dx_hat) increases, cutoff increases -> Less lag
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 3. Filter the Signal
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # 4. Update memory
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

class PointFilter:
    """Helper to filter (x,y) or (x,y,z) points easily."""
    def __init__(self, min_cutoff=1.0, beta=0.0):
        self.filters = []
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.first_frame = True

    def process(self, coords):
        # coords is a list [x, y] or [x, y, z]
        t = time.time()
        
        if self.first_frame:
            # Initialize one filter per dimension
            self.filters = [OneEuroFilter(t, x, self.min_cutoff, self.beta) for x in coords]
            self.first_frame = False
            return coords
        
        return [f(t, x) for f, x in zip(self.filters, coords)]