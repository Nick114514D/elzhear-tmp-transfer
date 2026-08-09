"""
engine/config.py — project-wide constants.

Audio spec locked 2026-06-23: 48 kHz / 24-bit, stereo masters.
Every render in the assemblage inherits these values; import from here rather
than hard-coding numbers in agents or scripts.
"""
from __future__ import annotations
from pathlib import Path

# ---- audio spec (locked) ----
SR: int = 48_000          # sample rate, Hz
BIT_DEPTH: int = 24       # render bit depth
SUBTYPE: str = "PCM_24"   # soundfile subtype for 24-bit PCM WAV
CHANNELS: int = 2         # stereo masters

# ---- reference levels ----
HEADROOM_DB: float = -1.0     # default normalization ceiling for written files
MIX_TARGET_DB: float = -14.0  # loose RMS target for fragments (guidance, not enforced)

# ---- repo paths ----
ROOT: Path = Path(__file__).resolve().parent.parent
CORPUS: Path = ROOT / "corpus"
POOL: Path = CORPUS / "pool"     # candidates awaiting judgment
CANON: Path = CORPUS / "canon"   # promoted material
LEDGER: Path = CORPUS / "ledger.json"
BRIEF: Path = ROOT / "brief.md"
