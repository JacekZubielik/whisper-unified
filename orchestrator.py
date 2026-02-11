#!/usr/bin/env python3
"""
Backwards-compatible entry point.

Use `python -m src.whisper_unified` instead.
This file is kept for Docker compatibility during migration.
"""

from src.whisper_unified.api.app import create_app

app = create_app()

if __name__ == "__main__":
    from src.whisper_unified.__main__ import main

    main()
