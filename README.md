# itc-hackathon — AgentHN

**AgentHN** (Agent + Hypernetwork): agents that write to their own **weights** at
inference time, built on [Doc-to-LoRA](https://github.com/SakanaAI/doc-to-lora).

Tracks live in their own folders under `src/agenthn/` so the team can work in
parallel:

| Folder | Owner | What |
|---|---|---|
| `core/` | shared | D2L model wrapper + config (keep stable) |
| `personalization/` | Eric | running profile doc → per-user adapter |
| `memory/` | Bryan, Nikash | long-horizon memory via stacked adapters |
| `skills/` | David | self-improving skills (T2L) |
| `webapp/` | Eric | demo UI |

## Setup

This package reuses the working `doc-to-lora` venv (which already has the full
GPU stack + the `ctx_to_lora` editable install). It is installed editable into
that venv — no separate environment needed.

```bash
export PATH="$HOME/.local/bin:$PATH"
uv pip install -e /home/ubuntu/itc-hackathon --python /home/ubuntu/doc-to-lora/.venv
```

Requires HF login for the gated `google/gemma-2-2b-it` base model
(`uv run --no-sync huggingface-cli login` in the doc-to-lora repo).

## Smoke test

```bash
/home/ubuntu/doc-to-lora/.venv/bin/python scripts/smoke_test.py
```

## Web app (cache-only demo UI)

The deployed browser loads only committed files from `webapp/static/fixtures`.
It never health-checks or sends requests to the Prime Intellect/GPU backend.
Captured streams replay one frame every 200 ms; personalization and skills use
recorded example choices so every enabled action has a genuine cached result.

The FastAPI/GPU code remains available only for regenerating fixtures. See
`scripts/README_GPU.md`, including the command for capturing Large memory runs.

## Layout

```
src/agenthn/
  core/
    config.py               # paths (D2L repo, checkpoint), device
    model.py                # D2LModel: load / internalize / snapshot / restore / chat
  personalization/
    extractor.py            # turns -> {category, value, action} updates
    profile_store.py        # per-user profile docs + cached adapters (swap)
  webapp/
    app.py                  # FastAPI: serves the SPA + personalization API
    service.py              # live (D2L) + mock services behind the API
    static/                 # index.html, app.js, styles.css (the demo page)
  memory/ skills/           # teammates' tracks
scripts/smoke_test.py       # load checkpoint, internalize, generate
```
