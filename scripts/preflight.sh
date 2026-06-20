#!/usr/bin/env bash
#
# Preflight for scripts/run_demo_server.sh — checks everything the demo run
# needs, non-destructively, and reports PASS/FAIL/WARN. Run it before showtime:
#
#     bash scripts/preflight.sh
#
# Honors the same env knobs as the orchestrator (PY, PORT, BRANCH, SWITCH_AT,
# KILL_AT). Exits non-zero if any hard check FAILs.
set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

PY="${PY:-/home/ubuntu/doc-to-lora/.venv/bin/python}"
PORT="${PORT:-8000}"
BRANCH="${BRANCH:-main}"
SWITCH_AT="${SWITCH_AT:-2026-06-20T20:55:00-07:00}"
KILL_AT="${KILL_AT:-2026-06-20T21:00:00-07:00}"

fails=0; warns=0
pass() { printf "  \033[32mPASS\033[0m  %s\n" "$1"; }
warn() { printf "  \033[33mWARN\033[0m  %s\n" "$1"; warns=$((warns+1)); }
fail() { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; fails=$((fails+1)); }

echo "== preflight for run_demo_server.sh =="

# 1. git repo + branch
if git rev-parse --git-dir >/dev/null 2>&1; then
  pass "git repo (branch $(git rev-parse --abbrev-ref HEAD))"
else
  fail "not inside a git repository"
fi

# 2. python interpreter with the GPU stack
if [ ! -x "$PY" ]; then
  for c in /root/doc-to-lora/.venv/bin/python "$(command -v python3)"; do
    [ -x "$c" ] && PY="$c" && break
  done
fi
if [ -x "$PY" ]; then pass "python: $PY"; else fail "no python interpreter found (set PY=...)"; fi

# 3. heavy deps import (torch + ctx_to_lora)
if [ -x "$PY" ] && "$PY" - <<'PYEOF' 2>/tmp/preflight_import.log
import torch, ctx_to_lora  # noqa: F401
assert torch.cuda.is_available(), "CUDA not available"
print("cuda devices:", torch.cuda.device_count())
PYEOF
then
  pass "torch + ctx_to_lora import; CUDA available"
else
  fail "torch/ctx_to_lora import or CUDA check failed — see /tmp/preflight_import.log"
fi

# 4. agenthn package importable in that venv
if [ -x "$PY" ] && PYTHONPATH=src "$PY" -c "import agenthn.webapp.app" >/tmp/preflight_app.log 2>&1; then
  pass "agenthn.webapp.app imports"
else
  warn "agenthn.webapp.app import failed (model load is lazy; check /tmp/preflight_app.log)"
fi

# 5. Hugging Face auth (token present)
if [ -f "$HOME/.cache/huggingface/token" ] || [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  pass "Hugging Face token present"
else
  warn "no HF token found — run 'huggingface-cli login' if the checkpoint is gated"
fi

# 6. git push access (dry-run, writes nothing)
if git ls-remote origin >/dev/null 2>&1; then
  if git push --dry-run origin "HEAD:$BRANCH" >/tmp/preflight_push.log 2>&1; then
    pass "git push access to origin/$BRANCH (dry-run)"
  else
    fail "git push dry-run failed — fix credentials (see /tmp/preflight_push.log)"
  fi
else
  fail "cannot reach git remote 'origin' — check auth/URL"
fi
git config user.email >/dev/null 2>&1 || warn "git user.email unset (orchestrator sets a default)"

# 7. tunnel binary (or downloadable)
if command -v cloudflared >/dev/null 2>&1 || [ -x ./bin/cloudflared ]; then
  pass "cloudflared available"
elif curl -fsI https://github.com/cloudflare/cloudflared/releases/latest >/dev/null 2>&1; then
  warn "cloudflared missing but reachable for auto-download at run time"
else
  fail "cloudflared missing and GitHub releases unreachable — install it manually"
fi

# 8. outbound HTTPS
if curl -fsS -o /dev/null https://api.github.com 2>/dev/null; then
  pass "outbound HTTPS works"
else
  fail "no outbound HTTPS — tunnel + git push will fail"
fi

# 9. port free
if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  warn "port $PORT already in use — orchestrator will collide (set PORT=...)"
else
  pass "port $PORT free"
fi

# 10. schedule sanity
epoch() { date -d "$1" +%s 2>/dev/null || \
  python3 -c 'import sys,datetime;print(int(datetime.datetime.fromisoformat(sys.argv[1]).timestamp()))' "$1"; }
now=$(date +%s); sw=$(epoch "$SWITCH_AT"); kl=$(epoch "$KILL_AT")
if [ -n "$sw" ] && [ -n "$kl" ] && [ "$now" -lt "$sw" ] && [ "$sw" -lt "$kl" ]; then
  pass "schedule ok: now < switch($SWITCH_AT) < kill($KILL_AT)"
else
  warn "schedule looks off — now=$(date -u +%FT%TZ), switch=$SWITCH_AT, kill=$KILL_AT (override with SWITCH_AT/KILL_AT)"
fi

# 11. sudo for the final poweroff
if [ "${DO_SHUTDOWN:-1}" = "1" ]; then
  if sudo -n true 2>/dev/null; then
    pass "passwordless sudo (poweroff at 9pm will work)"
  else
    warn "no passwordless sudo — auto-poweroff may fail; stop the instance from the dashboard, or set DO_SHUTDOWN=0"
  fi
fi

echo "== summary: $fails fail, $warns warn =="
[ "$fails" -eq 0 ] && { echo "ready — run: bash scripts/run_demo_server.sh"; exit 0; } || \
  { echo "fix the FAILs above before running."; exit 1; }
