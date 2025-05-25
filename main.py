#!/usr/bin/env python3
"""
Wrapper script for SampleGen Hip-Hop Producer AI.
This allows running the main producer from the project root.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the main function
from src.main_producer import main, demo_with_example

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments provided, run the demo
        demo_with_example()
    else:
        main() 