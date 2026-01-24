# THE GOLDEN RULES
# STRICTLY derived from Deep Formula Hunter Output

RULES = {
    # --- GLOBAL BASICS ---
    "OPEN_EXT": 1.0,         # General threshold for "Finger is Up"
    "PINCH_START": 0.21,     # Click Down
    "PINCH_STOP": 0.35,      # Click Up
    
    # --- WAR ZONE 1: POINTER vs ZOOM vs SHHH ---
    # Decision: Ring Tip to Ring Knuckle distance checks if hand is "Tight Fist" or "Loose"
    "TIGHT_FIST_RING": 0.54, # Dist(RingTip-RingKnuckle) <= 0.54 is Point/Zoom
    "ZOOM_THUMB": 0.81,      # Dist(ThumbTip-MidKnuckle) > 0.81 is Zoom
    
    # --- WAR ZONE 2: V-SHAPES (Scroll/Right/Tab/Win) ---
    "SCROLL_GAP": 0.55,      # Dist(MidTip-RingTip) <= 0.55 is Scroll
    "WIN_GAP": 0.36,         # Dist(RingTip-PinkyTip) > 0.36 is Win
    "RIGHT_CLICK_WRIST": 0.98, # Dist(Wrist-RingKnuckle) <= 0.98 is Right Click
    
    # --- WAR ZONE 3: PALM vs DELETE ---
    "PALM_THUMB_X": 0.25,    # Thumb X Offset > 0.25 is Palm

    # --- WAR ZONE 4: THE FIST WARS (Crucial) ---
    "CUT_PINKY": 1.63,       # Pinky Ext > 1.63 is Cut
    "SIDE_GESTURE_X": 0.54,  # Pinky X Offset > 0.54 is Ctrl/Next
    "CTRL_SQUEEZE": 0.72,    # Dist(IndexTip-PinkyKnuckle) <= 0.72 is Ctrl
    "PREV_SQUEEZE": 0.48,    # Dist(IndexTip-MidTip) <= 0.48 is Previous
}