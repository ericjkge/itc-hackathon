"""Memory track: long-horizon agent memory stored in LoRA blocks, not context.

An agent journals observations over a long session. Each observation is
internalized into a LoRA adapter and rank-concatenated onto a running memory
adapter (ctx_to_lora.modeling.lora_merger.combine_lora) — then evicted from the
context window. At recall time the prompt holds only the question, so per-query
context cost is ~constant no matter how much the agent has remembered.

Key API:
  WeightMemory(model, mode="doc")  - the store: remember(text) / recall(question)
  generate_session(n, seed)        - synthetic agentic memory task + recall probes

Requires the chunk-trained checkpoint (config.CHUNK_CHECKPOINT). The gemma_demo
checkpoint is single-chunk only and its combine_lora produces garbage — see
scripts/memory_chunkpath_diag.py and the track README.
"""

from .memory_store import WeightMemory, encode_doc_chunks
from .tasks import MemoryEntry, Probe, generate_session

__all__ = [
    "WeightMemory",
    "encode_doc_chunks",
    "MemoryEntry",
    "Probe",
    "generate_session",
]
