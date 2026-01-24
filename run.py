# run.py
import sys
import os

# Add the project root to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import main

if __name__ == "__main__":
    main()