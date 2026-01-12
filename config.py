import sys
from pathlib import Path

# 1. ANCHOR THE SYSTEM
ROOT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(ROOT_DIR))

# 2. DEFINE PATHS
PATHS = {
    "ROOT": ROOT_DIR,
    "RULES": ROOT_DIR / "data" / "heuristic_rules.json",
}

# 3. ENSURE DATA DIR EXISTS
(ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)