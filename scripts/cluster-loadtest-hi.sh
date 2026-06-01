#!/usr/bin/env bash
# Short-prompt variant of cluster-loadtest.sh. Fires N concurrent "hi <i>"
# requests (default 30) to exercise the dispatcher under burst load without
# spending real generation time per request. Same node-tally output.
#
# Usage:
#   CLUSTER_SECRET=... ENTRY=http://srv1:42069 MODEL=gpt-oss-20b ./scripts/cluster-loadtest-hi.sh [N]
#
# See cluster-loadtest.sh for the full rules on ENTRY port choice and what
# balancing actually requires (Share queue ON, Max Concurrent > 0, >1 node).

SECRET="YOUR_API_KEY_HERE"
ENTRY="${ENTRY:-http://localhost:42069}"

MODEL="gpt-oss-20b-Q4_K_M"


N="${1:-${N:-30}}"
ENTRY="${ENTRY%/}"

[[ -n "$SECRET" && -n "$MODEL" ]] || {
  echo "set CLUSTER_SECRET, MODEL (and ENTRY). e.g.:" >&2
  echo "  CLUSTER_SECRET=xxx ENTRY=http://srv1:5000 MODEL=${MODEL} $0 30" >&2
  exit 1
}

run() {  # $1 = label, $2 = path
  echo "── $1 ($ENTRY$2)  $N concurrent ──"
  local tmp; tmp="$(mktemp)"
  for i in $(seq 1 "$N"); do
    (
      out="$(curl -s -D - -o /dev/null -w 'HTTPSTATUS:%{http_code}' \
        -H "Authorization: Bearer $SECRET" -H "Content-Type: application/json" \
        "$ENTRY$2" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi $i\"}],\"stream\":false}" )"
      node="$(printf '%s' "$out" | awk 'tolower($0) ~ /^x-llamaman-node:/ {print $2}' | tr -d '\r')"
      entry="$(printf '%s' "$out" | awk 'tolower($0) ~ /^x-llamaman-entry:/ {print $2}' | tr -d '\r')"
      code="$(printf '%s' "$out" | sed -n 's/.*HTTPSTATUS:\([0-9][0-9]*\).*/\1/p')"
      echo "${node:-UNKNOWN}" >> "$tmp"
      echo "  req $i -> served=${node:-UNKNOWN} entry=${entry:-?} (HTTP ${code:-?})"
    ) &
  done
  wait
  echo "  tally:"; sort "$tmp" | uniq -c | sed 's/^/   /'
  rm -f "$tmp"
}

run "Ollama" "/api/chat"
echo
run "OpenAI" "/v1/chat/completions"
