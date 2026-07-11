"""Helpers for Textual/Rich markup in TUI status strings."""

from __future__ import annotations


def escape_brackets(text: str) -> str:
    """Escape ``[`` so keyboard hints and ``[n/m]`` indices render literally."""
    return text.replace("[", r"\[")
