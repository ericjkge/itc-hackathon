"""Cost model: $ per query (prefix-cache hit vs miss) + one-time creation cost.

"Memory used at inference" isn't the same as $ cost once prefix caching exists.
This model prices each memory strategy at representative frontier-API rates and
separates two costs:

  creation  one-time, per corpus/history (write the memory)
  query     per question (read the memory + answer), with a cold (uncached) and a
            warm (prefix-cache hit) variant

Methods (per-query context taken from our measured runs where applicable):
  vanilla     whole history in the prompt every query   (H tokens)
  RAG         retrieve top-1 chunk as text              (~120 tokens, varies/query)
  Cartridges  trained KV cache prefix (HazyResearch)    (ICL/38.6 tokens, cached)
  NapLoRA     question only; memory is in the adapter   (~10 tokens + weight swap)

Token costs are a proxy for compute. NapLoRA's adapter apply + Cartridges'/NapLoRA's
creation are compute, not API tokens — noted, and creation is estimated.

  /root/doc-to-lora/.venv/bin/python scripts/cost_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "cost.json"

# Representative frontier pricing ($ per 1M tokens), Sonnet-class, with Anthropic-
# style prompt caching (cache write = 1.25x input, cache read = 0.1x input).
PRICE = {"in": 3.00, "cache_write": 3.75, "cache_read": 0.30, "out": 15.00}
M = 1_000_000

# Scenario: a long agent history / corpus, queried many times.
H = 48_000          # history / corpus tokens
Q_TOK = 10          # question tokens
OUT_TOK = 30        # answer tokens
N_QUERIES = 1_000   # queries to amortize creation over

# Measured per-query context (tokens the model attends to), from our eval runs.
RAG_CTX = 116
NAPORA_CTX = 7
CARTRIDGE_KV = round(H / 38.6)   # paper: 38.6x less memory than ICL


def q_cost(ctx_uncached, ctx_cached=0):
    """$ for one query: uncached input + cached input (read) + output."""
    return (ctx_uncached * PRICE["in"] + ctx_cached * PRICE["cache_read"]
            + OUT_TOK * PRICE["out"]) / M


def main():
    methods = {}

    # vanilla: whole history every query
    methods["vanilla"] = {
        "query_cold": q_cost(H + Q_TOK),                 # all uncached
        "query_warm": q_cost(Q_TOK, H),                  # history prefix-cached
        "creation": 0.0,
        "online": True, "quality_note": "0% once history > window",
        "ctx": H,
    }
    # RAG: retrieved chunk varies per query -> not a stable cache prefix
    methods["rag"] = {
        "query_cold": q_cost(RAG_CTX + Q_TOK),
        "query_warm": q_cost(RAG_CTX + Q_TOK),           # chunk changes -> no prefix hit
        "creation": (H * PRICE["in"]) / M * 0.1,         # build embedding index (~cheap)
        "online": True, "quality_note": "~100% (lossless text)",
        "ctx": RAG_CTX,
    }
    # Cartridges: trained KV cache prefix -> cache hit every query; expensive offline train
    methods["cartridges"] = {
        "query_cold": q_cost(Q_TOK, CARTRIDGE_KV),       # cartridge loaded as cached prefix
        "query_warm": q_cost(Q_TOK, CARTRIDGE_KV),
        # offline self-study: synth-gen + gradient training. Estimated as ~read the
        # corpus hundreds of times (fwd+bwd). Order-of-magnitude, labeled estimate.
        "creation": (H * PRICE["in"]) / M * 300,
        "online": False, "quality_note": "matches ICL (paper)",
        "ctx": CARTRIDGE_KV,
    }
    # NapLoRA: question only; memory in the adapter (one fwd pass/segment to create)
    methods["napora"] = {
        "query_cold": q_cost(NAPORA_CTX),
        "query_warm": q_cost(NAPORA_CTX),
        "creation": (H * PRICE["in"]) / M,               # one hypernet pass over the corpus
        "online": True, "quality_note": "~87% (lossy encoding)",
        "ctx": NAPORA_CTX,
    }

    # total $ over N queries (amortize creation), warm-cache regime
    for m in methods.values():
        m["total_warm"] = m["creation"] + m["query_warm"] * N_QUERIES
        m["total_cold"] = m["creation"] + m["query_cold"] * N_QUERIES
        for k in ("query_cold", "query_warm", "creation", "total_warm", "total_cold"):
            m[k] = round(m[k], 6)

    out = {"price_per_Mtok": PRICE, "scenario": {"history_tokens": H, "query_tokens": Q_TOK,
            "output_tokens": OUT_TOK, "n_queries": N_QUERIES, "cartridge_kv_tokens": CARTRIDGE_KV},
           "methods": methods}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))

    print(f"Scenario: {H:,}-token history, {N_QUERIES:,} queries, prices $/Mtok={PRICE}\n")
    print(f"{'method':<11} {'ctx':>7} {'$/query cold':>13} {'$/query warm':>13} {'creation $':>11} {'$/1k warm':>10}  online")
    for name, m in methods.items():
        print(f"{name:<11} {m['ctx']:>7} {m['query_cold']*1000:>11.4f}m {m['query_warm']*1000:>11.4f}m "
              f"{m['creation']:>10.4f} {m['total_warm']:>9.3f}  {'yes' if m['online'] else 'OFFLINE'}")
    print(f"\n(m = milli-dollars per query; $/1k = total $ for {N_QUERIES} queries incl. creation)")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
