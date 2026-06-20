"""AgentHN — Agent + Hypernetwork.

Agents that write to their own weights at inference time, built on Doc-to-LoRA.

Tracks live in their own subpackages so teammates can work in parallel:
  core/            shared model wrapper + config (stable; change with care)
  personalization/ Eric  — running profile doc -> per-user adapter
  memory/          Bryan, Nikash — long-horizon memory (stacked adapters)
  skills/          David — self-improving skills (T2L)
  webapp/          Eric  — demo UI
"""

__version__ = "0.0.1"
