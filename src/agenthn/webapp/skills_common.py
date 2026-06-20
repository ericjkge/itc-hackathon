"""Shared generation helper for the live skills demos (physics, formatting, ...).

D2LModel.chat() has no assistant-prefill hook, so this talks to the
tokenizer/model directly instead of changing the shared wrapper's API for
one demo family. Used by skills_service_physics.py and
skills_service_formatting.py (mirrors generate_text() in
doc-to-lora/scripts/skill-acquisition-physics-timing.py).
"""

from __future__ import annotations

import time

import torch

DEFAULT_PREFILL = "Let's think step by step. "
DEFAULT_MAX_NEW_TOKENS = 200


def generate_with_prefill(
    model,
    message: str,
    prefill: str = DEFAULT_PREFILL,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> tuple[str, float]:
    chat_str = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = model.tokenizer(
        chat_str + prefill, return_tensors="pt", add_special_tokens=False
    ).to(model.model.device)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    elapsed = time.perf_counter() - start
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return (prefill + text).strip(), elapsed


__all__ = ["generate_with_prefill", "DEFAULT_PREFILL", "DEFAULT_MAX_NEW_TOKENS"]
