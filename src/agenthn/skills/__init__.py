"""Skills track (David): self-improving skills via Text-to-LoRA (T2L).

Agent fails a task -> writes a better task-specific prompt -> convert to a LoRA
adapter -> retry. Uses Sakana's T2L (separate from the D2L code in core).
"""
