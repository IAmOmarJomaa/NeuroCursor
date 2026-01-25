"""
NeuroCursor Grid Generator.
==========================
Stitches individual gesture images into a professional 'Cheat Sheet' for the README.
Adds labels and action descriptions automatically.
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
IMAGE_DIR = os.path.join("assets", "gestures")
OUTPUT_FILE = os.path.join("assets", "gesture_grid.png")

# Layout Settings
COLUMNS = 4
THUMB_SIZE = (400, 300)  # Width, Height of each image
TEXT_HEIGHT = 110        # Height of the text area below image
BG_COLOR = (30, 30, 30)  # Dark Grey Background
TEXT_COLOR = (255, 255, 255)
ACCENT_COLOR = (0, 255, 255) # Cyan for Titles

# Descriptions map (Filename -> (Pretty Name, Description))
GESTURE_META = {
    "GEN_Z": ("GEN_Z (Lock)", "Index+Thumb Heart -> Hold to Lock"),
    "THE_POINT": ("THE_POINT", "Index Out -> Move Cursor"),
    "THE_CLICK": ("THE_CLICK", "Pinch -> Left Click / Drag"),
    "RIGHT_CLICK": ("RIGHT_CLICK", "Middle Finger Up -> Right Click"),
    "SCROLL": ("SCROLL", "Spiderman Hand -> Scroll Up/Down"),
    "VOLUME": ("VOLUME", "Phone Hand -> Tilt for Volume"),
    "ZOOM": ("ZOOM", "L-Shape -> Two Hands to Zoom"),
    "TAB_M": ("TAB_M", "Fingers Straight -> Alt-Tab Menu"),
    "TAB_L": ("TAB_L", "Tilt Left -> Select Left Window"),
    "TAB_R": ("TAB_R", "Tilt Right -> Select Right Window"),
    "CTRL_PINKY": ("CTRL_PINKY", "C-Shape -> Toggle Ctrl Key"),
    "FIST": ("FIST", "Closed Fist -> Copy (after Palm)"),
    "PALM": ("PALM", "Open Hand -> Paste / Reset"),
    "CUT": ("CUT", "Fist + Pinky -> Cut (after Palm)"),
    "DELETE_CLOSED": ("DELETE (End)", "Fist (Thumb In) -> Finish Delete"),
    "DELETE_OPENED": ("DELETE (Start)", "Palm (Thumb In) -> Start Delete"),
    "NEXT_CLOSED": ("NEXT (Start)", "Fist (Thumb Rest) -> Prep Forward"),
    "NEXT_OPENED": ("NEXT (End)", "Thumb Flick -> Go Forward"),
    "PREVIOUS_CLOSED": ("PREV (Start)", "Fist (Nails Fwd) -> Prep Back"),
    "PREVIOUS_OPENED": ("PREV (End)", "Thumb Flick -> Go Back"),
    "WIN": ("WIN KEY", "3 Fingers Up -> Task View"),
    # Fallbacks for any others found
}

def create_card(img_path, label, desc):
    """Creates a single card with image and text."""
    # 1. Load and Resize Image
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"⚠️ Could not load {img_path}")
        return None

    # Crop/Resize to fit thumbnail area (Aspect Ratio Preserve usually better, but we force fit for grid)
    img = img.resize(THUMB_SIZE, Image.Resampling.LANCZOS)

    # 2. Create Canvas for Card
    card_h = THUMB_SIZE[1] + TEXT_HEIGHT
    card = Image.new("RGB", (THUMB_SIZE[0], card_h), BG_COLOR)
    
    # 3. Paste Image
    card.paste(img, (0, 0))
    
    # 4. Draw Text
    draw = ImageDraw.Draw(card)
    
    # Attempt to load a nice font, fallback to default
    try:
        # standard windows font
        font_title = ImageFont.truetype("arialbd.ttf", 28)
        font_desc = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font_title = ImageFont.load_default()
        font_desc = ImageFont.load_default()

    # Draw Label (Title)
    # Centered logic
    draw.text((20, THUMB_SIZE[1] + 15), label, font=font_title, fill=ACCENT_COLOR)
    
    # Draw Description
    draw.text((20, THUMB_SIZE[1] + 55), desc, font=font_desc, fill=TEXT_COLOR)
    
    # Border
    draw.rectangle([(0,0), (THUMB_SIZE[0]-1, card_h-1)], outline=(60,60,60), width=2)
    
    return card

def main():
    print("🎨 STARTING GRID GENERATION...")
    
    # 1. Find Images
    files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")])
    if not files:
        print("❌ No images found in assets/gestures/")
        return

    cards = []
    
    # 2. Process Each Image
    for f in files:
        key = os.path.splitext(f)[0]
        meta = GESTURE_META.get(key, (key, "Gesture Action"))
        
        path = os.path.join(IMAGE_DIR, f)
        print(f"   Processing: {key}...")
        card = create_card(path, meta[0], meta[1])
        if card:
            cards.append(card)

    if not cards: return

    # 3. Create Master Grid
    num_cards = len(cards)
    rows = math.ceil(num_cards / COLUMNS)
    
    grid_w = COLUMNS * THUMB_SIZE[0]
    grid_h = rows * (THUMB_SIZE[1] + TEXT_HEIGHT)
    
    master = Image.new("RGB", (grid_w, grid_h), BG_COLOR)
    
    for idx, card in enumerate(cards):
        col = idx % COLUMNS
        row = idx // COLUMNS
        
        x = col * THUMB_SIZE[0]
        y = row * (THUMB_SIZE[1] + TEXT_HEIGHT)
        
        master.paste(card, (x, y))

    # 4. Save
    print(f"💾 Saving Grid to: {OUTPUT_FILE}")
    master.save(OUTPUT_FILE)
    print("✅ DONE. Check assets/gesture_grid.png")

if __name__ == "__main__":
    main()