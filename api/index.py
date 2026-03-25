"""Vercel serverless function entry point."""

import sys
import os

# Add the project root to Python path so 'src.app' imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.api import app
