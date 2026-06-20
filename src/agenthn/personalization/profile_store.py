"""PersonalizationStore: per-user profile docs + cached LoRA adapters.

Lifecycle per user:
  update_profile(uid, updates)  -> merge structured updates into the profile doc
                                   (dedup, resolve conflicts, recompact to fit ctx)
  repersonalize(uid)            -> internalize the doc into ONE adapter, cache it
  chat(uid, message)            -> swap that user's adapter in, then generate

The profile doc is the human-readable source of truth and NEVER enters the prompt.
"""

from __future__ import annotations

from .extractor import ProfileUpdate
from ..core.model import D2LModel


class PersonalizationStore:
    def __init__(self, model: D2LModel):
        self.model = model
        self._profiles: dict[str, str] = {}      # uid -> profile doc text
        self._adapters: dict[str, object] = {}    # uid -> cached generated adapter

    # --- profile doc maintenance -----------------------------------------
    def update_profile(self, uid: str, updates: list[ProfileUpdate]) -> str:
        """Apply structured updates to the user's profile doc; return new doc."""
        raise NotImplementedError

    def recompact(self, uid: str) -> str:
        """LLM-summarize the doc to stay coherent and within the ctx window."""
        raise NotImplementedError

    # --- adapter cache / swap --------------------------------------------
    def repersonalize(self, uid: str) -> None:
        """Internalize the user's current doc into an adapter and cache it."""
        doc = self._profiles[uid]
        self.model.reset()
        self.model.internalize(doc)
        self._adapters[uid] = self.model.snapshot()

    def chat(self, uid: str, message: str, **kw) -> str:
        """Swap in the user's cached adapter, then generate."""
        adapter = self._adapters.get(uid)
        if adapter is None:
            self.model.reset()           # no profile yet -> base behavior
        else:
            self.model.restore(adapter)
        return self.model.chat(message, **kw)
