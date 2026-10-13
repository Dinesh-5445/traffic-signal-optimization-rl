"""
conftest.py - Adds the project root to sys.path so that tests can import
project packages (agents, environment, training, communication) without
requiring an editable install or manual PYTHONPATH setup.
"""
import sys
import os

# Insert the repo root at index 0 so local packages take priority over installed ones
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

