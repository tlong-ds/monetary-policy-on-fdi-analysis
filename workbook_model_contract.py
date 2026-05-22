"""Compatibility shim for the refactored workbook contract module.

New notebook code should import from ``src.model_contract``.
"""

from src.model_contract import *  # noqa: F401,F403
