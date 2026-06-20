"""Personalization track: running profile doc -> LoRA adapter, per user.

Design (see project memory `personalization-track-design`):
  conversation -> structured extractor -> profile doc (never enters the prompt)
  -> manual re-personalize -> internalize into ONE adapter -> cache + swap per user.
We do NOT stack/rank-concat adapters here (that's the memory track's complexity).
"""
