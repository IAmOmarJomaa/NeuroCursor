import math
import numpy as np

class HandGeometry:
    
    @staticmethod
    def get_distance(p1, p2):
        """Calculates Euclidean distance between two 3D points."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    @staticmethod
    def get_vec_distance(v1, v2):
        """Calculates Euclidean distance between two numpy arrays."""
        return np.linalg.norm(np.array(v1) - np.array(v2))

    @staticmethod
    def get_angle(a, b, c):
        """
        Calculates the angle at point 'b' formed by 'a-b-c'.
        Returns degrees (0-180).
        Used for checking if a finger is curled or straight.
        """
        # Convert to numpy arrays for vector math
        va = np.array([a.x, a.y, a.z])
        vb = np.array([b.x, b.y, b.z])
        vc = np.array([c.x, c.y, c.z])
        
        # Create vectors BA and BC
        ba = va - vb
        bc = vc - vb
        
        # Calculate Cosine similarity
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Clamp to handle floating point errors (e.g., 1.0000001)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    @staticmethod
    def get_finger_state(hand_landmarks, finger_name):
        """
        Returns the bending angle of a specific finger.
        < 90 degrees usually means 'BENT' (Fist).
        > 160 degrees usually means 'STRAIGHT' (Open).
        """
        mp_indices = {
            "thumb":  [2, 3, 4],
            "index":  [5, 6, 8], # Using knuckle, pip, tip
            "middle": [9, 10, 12],
            "ring":   [13, 14, 16],
            "pinky":  [17, 18, 20]
        }
        
        idx = mp_indices[finger_name]
        p1 = hand_landmarks.landmark[idx[0]]
        p2 = hand_landmarks.landmark[idx[1]]
        p3 = hand_landmarks.landmark[idx[2]]
        
        return HandGeometry.get_angle(p1, p2, p3)

    @staticmethod
    def get_all_finger_states(hand_landmarks):
        return {
            "thumb": HandGeometry.get_finger_state(hand_landmarks, "thumb"),
            "index": HandGeometry.get_finger_state(hand_landmarks, "index"),
            "middle": HandGeometry.get_finger_state(hand_landmarks, "middle"),
            "ring":   HandGeometry.get_finger_state(hand_landmarks, "ring"),
            "pinky":  HandGeometry.get_finger_state(hand_landmarks, "pinky")
        }