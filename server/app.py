"""
server/app.py — OpenEnv multi-mode deployment entry point.
This re-exports the FastAPI app from the root app.py so the
OpenEnv validator can find it at the standard server/app.py path.
"""
import sys
import os

# Allow importing from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # re-export

__all__ = ["app"]