#!/usr/bin/env python3
"""Convenience runner script for credential encryption and key generation."""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.security.cli import main

if __name__ == "__main__":
    main()
