#!/usr/bin/env bash
set -euo pipefail

# ------------ SETTINGS ------------
JULIA=${JULIA:-julia}          # which julia (can override env JULIA)
SCRIPT="phi4gsvumps.jl"             # your Julia script

# What to commit: you can track a directory, a specific file, or patterns.
# Examples:
#   TRACK_PATHS=("phi4gs.png")
#   TRACK_PATHS=("plots/" "phi4gs.png")
TRACK_PATHS=("phi4EEgsinfinite_d30.png")

NAME="Auto Plot Bot"
EMAIL="ujjwalbasumatary@gmail.com"

# SSH key
KEY="${KEY:-$HOME/.ssh/id_ed25519}"

# ------------ SSH-AGENT SETUP ------------
AGENT_ENV="$HOME/.ssh/agent_env"

start_agent () {
  echo "[ssh-agent] starting..."
  eval "$(ssh-agent -s)" > /dev/null
  echo "export SSH_AUTH_SOCK=$SSH_AUTH_SOCK" > "$AGENT_ENV"
  echo "export SSH_AGENT_PID=$SSH_AGENT_PID" >> "$AGENT_ENV"
}

if [[ -f "$AGENT_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$AGENT_ENV"
  if ! ssh-add -l >/dev/null 2>&1; then
    start_agent
  fi
else
  start_agent
fi

# Load the key if not present
if ! ssh-add -l | grep -q "$(ssh-keygen -lf "$KEY" | awk '{print $2}')" 2>/dev/null; then
  echo "[ssh-agent] adding key: $KEY"
  ssh-add "$KEY"
fi

# ------------ GIT IDENTITY & REMOTE CHECK ------------
git config user.name  "$NAME"
git config user.email "$EMAIL"

REMOTE_URL="$(git remote get-url origin)"
if [[ "$REMOTE_URL" != git@* ]]; then
  echo "Warning: origin is not SSH (current: $REMOTE_URL)"
  echo "Run: git remote set-url origin git@github.com:<USER>/<REPO>.git"
fi

# Fail fast if SSH auth fails (no interactive prompts for cron/systemd)
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

# ------------ RUN YOUR JOB ------------
echo "[run] $JULIA $SCRIPT"
"$JULIA" "$SCRIPT"

# ------------ COMMIT & PUSH ------------
for p in "${TRACK_PATHS[@]}"; do
  git add "$p" 2>/dev/null || true
done

if git diff --cached --quiet; then
  echo "[git] no PNG/plot changes to commit."
  exit 0
fi

HOST="${HOSTNAME:-unknown-host}"
WHEN="$(date -u +'%Y-%m-%d %H:%M:%S UTC')"
BASE_REV="$(git rev-parse --short HEAD || true)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

git commit -m "Auto: update phi4 plots after run on ${HOST} at ${WHEN} (base ${BASE_REV})"

# First push attempt
if ! git push origin "$BRANCH"; then
  echo "[git] push failed; rebase & retry…"
  git pull --rebase --autostash origin "$BRANCH"
  git push origin "$BRANCH"
fi

echo "[done] auto-push complete."
