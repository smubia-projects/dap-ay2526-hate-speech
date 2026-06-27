#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STREAMLIT_BIN="${STREAMLIT_BIN:-.venv/bin/streamlit}"
APP_PATH="${APP_PATH:-demo/app.py}"
PORT="${PORT:-8501}"
FILE_WATCHER_TYPE="${FILE_WATCHER_TYPE:-none}"
LOG_DIR="${LOG_DIR:-tmp/streamlit-demo}"

BACKENDS=(
  "sg|SG SingBERT local HF model"
  "llmguard|LLM Guard Toxicity scanner"
  "lionguard|LionGuard 2.0, needs OPENAI_API_KEY"
  "lionguard-2.1|LionGuard 2.1, needs GEMINI_KEY"
  "sea-guard|SEA-Guard, needs SEA_LION_API_KEY"
)

select_backend() {
  if [[ -n "${GUARD_MODEL:-}" ]]; then
    echo "$GUARD_MODEL"
    return
  fi

  echo "Select guard backend:"
  local i=1
  local item backend desc
  for item in "${BACKENDS[@]}"; do
    backend="${item%%|*}"
    desc="${item#*|}"
    printf "  %d) %-13s %s\n" "$i" "$backend" "$desc"
    i=$((i + 1))
  done

  local choice
  while true; do
    read -r -p "Backend [1]: " choice
    choice="${choice:-1}"
    if [[ "$choice" =~ ^[1-5]$ ]]; then
      item="${BACKENDS[$((choice - 1))]}"
      echo "${item%%|*}"
      return
    fi
    echo "Please enter a number from 1 to 5."
  done
}

if [[ ! -x "$STREAMLIT_BIN" ]]; then
  echo "Streamlit binary not found at $STREAMLIT_BIN"
  echo "Run: uv sync"
  exit 1
fi

if [[ ! -f "$APP_PATH" ]]; then
  echo "Streamlit app not found at $APP_PATH"
  exit 1
fi

BACKEND="$(select_backend)"
mkdir -p "$LOG_DIR"
STAMP="$(date +"%Y%m%d-%H%M%S")"
LOG_FILE="$LOG_DIR/${BACKEND}-${STAMP}.log"
LATEST_LOG="$LOG_DIR/latest.log"

cat <<EOF

Starting demo
  backend:      $BACKEND
  app:          $APP_PATH
  url:          http://localhost:$PORT
  log file:     $LOG_FILE
  file watcher: $FILE_WATCHER_TYPE

Press Ctrl+C to stop.

EOF

ln -sf "$(basename "$LOG_FILE")" "$LATEST_LOG"

GUARD_MODEL="$BACKEND" "$STREAMLIT_BIN" run "$APP_PATH" \
  --server.port "$PORT" \
  --server.fileWatcherType "$FILE_WATCHER_TYPE" \
  2>&1 | tee "$LOG_FILE"
