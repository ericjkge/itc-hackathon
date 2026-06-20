"""Shared model runtime for the web app.

The D2L model is heavy (a few GB of VRAM) and NOT thread-safe — its active LoRA
adapter is mutated on every call. Both demos (personalization + memory) therefore
share ONE loaded model behind ONE lock, so they can't stomp on each other's
adapter state. Load is lazy so the server starts instantly.
"""

from __future__ import annotations

import threading
import time

_model = None
_load_lock = threading.Lock()


class ModelBusyError(RuntimeError):
    """Raised instead of leaving a disconnected HTTP request queued forever."""


class ModelLock:
    """Observable, bounded lock around the single mutable GPU model."""

    def __init__(self, wait_timeout: float = 3.0) -> None:
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._wait_timeout = wait_timeout
        self._waiting = 0
        self._held_since: float | None = None

    def __enter__(self):
        with self._state_lock:
            self._waiting += 1
        try:
            acquired = self._lock.acquire(timeout=self._wait_timeout)
        finally:
            with self._state_lock:
                self._waiting -= 1
        if not acquired:
            raise ModelBusyError("GPU is busy; retry this request shortly")
        with self._state_lock:
            self._held_since = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        with self._state_lock:
            self._held_since = None
        self._lock.release()

    def status(self) -> dict:
        with self._state_lock:
            held_for = (
                round(time.monotonic() - self._held_since, 1)
                if self._held_since is not None else 0.0
            )
            return {
                "busy": self._held_since is not None,
                "queued": self._waiting,
                "held_for_seconds": held_for,
            }


# Generation mutates the active adapter, so only one request may use the model.
# Contenders fail quickly rather than surviving a browser/tunnel disconnect in
# an unbounded queue and later running expensive work for an abandoned request.
MODEL_LOCK = ModelLock()


def get_model():
    """Return the shared, lazily-loaded D2LModel."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from ..core.model import D2LModel

                _model = D2LModel.load()
    return _model


def model_loaded() -> bool:
    return _model is not None


def model_status() -> dict:
    return {"model_loaded": model_loaded(), **MODEL_LOCK.status()}
