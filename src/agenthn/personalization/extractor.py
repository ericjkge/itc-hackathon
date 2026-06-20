"""Preference extractor: conversation turns -> structured profile updates.

Emits updates of the form {category, value, action: add|update|remove} so that
e.g. "I went vegan" overwrites "loves steak" instead of contradicting it. These
feed both the profile doc and the UI diff view.

TODO: implement against the chosen extractor LLM (local gemma vs Claude API).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ProfileUpdate:
    category: str
    value: str
    action: Literal["add", "update", "remove"]


def extract_updates(conversation: list[dict]) -> list[ProfileUpdate]:
    """Inspect recent turns and propose structured profile updates."""
    raise NotImplementedError
