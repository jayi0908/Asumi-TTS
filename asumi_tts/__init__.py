"""Asumi (亜澄) TTS — application-facing package."""
from .engine import AsumiTTSEngine, get_engine

__all__ = ["AsumiTTSEngine", "get_engine"]
__version__ = "1.0.0"
