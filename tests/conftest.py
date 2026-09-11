"""Shared test fixtures — initializes config and Agents SDK for imports."""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load config first (uses defaults if no config.json)
from core.config import load as load_config

try:
    load_config()
except Exception:
    pass

# Initialize the Agents SDK so agent modules can be imported
from core.agents_init import initialize

try:
    initialize()
except Exception:
    pass  # Tests that need the SDK will fail with clear errors
