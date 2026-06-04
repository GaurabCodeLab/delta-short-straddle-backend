from __future__ import annotations

# import asyncio
# import logging
import os
import sys
import uvicorn

if __name__ == "__main__" and __package__ is None:
    # Allow running `python main.py` from inside `src/` by adding project root.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, log_level="info")
