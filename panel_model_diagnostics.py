"""Compatibility shim for the refactored diagnostics module.

New notebook code should import from ``src.panel_diagnostics``.
"""

from src.panel_diagnostics import *  # noqa: F401,F403
