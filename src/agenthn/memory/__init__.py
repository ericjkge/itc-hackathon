"""Memory track (Bryan, Nikash): long-horizon memory via stacked LoRA adapters.

Every K steps, convert oldest turns into an adapter and rank-concat with the
running memory adapter (see ctx_to_lora.modeling.lora_merger.combine_lora), then
evict those turns from context.
"""
