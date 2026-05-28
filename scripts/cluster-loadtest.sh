#!/usr/bin/env bash
# Quick cluster distribution check: fire N concurrent requests at the Ollama
# (/api/chat) and OpenAI (/v1/chat/completions) proxy APIs and show which node
# served each, via the X-Llamaman-Node response header.
#
# Usage:
#   CLUSTER_SECRET=... ENTRY=http://srv1:42069 MODEL=gpt-oss-20b ./scripts/cluster-loadtest.sh [N]
#
# ENTRY must be an APP port - the management port (container 5000) or the proxy
# port 42069. BOTH serve the compat APIs and run cluster dispatch. Do NOT use a
# per-instance port (e.g. the 12020-12025 -> 8000-8005 range): those hit a
# single llama.cpp instance directly and bypass the cluster router (no balancing).
#
# Balancing only spreads when the model is loaded on >1 node with "Share queue"
# ON and Max Concurrent > 0; the requests must overlap (that's why they run
# concurrently). A single sequential request stays on the entry node by design.

SECRET="llm-V8XJNoHTBAYw-mRJ0-1wVgwtvd2l_IFvKhzQP-DJivI"
ENTRY="${ENTRY:-http://localhost:42069}"

MODEL="gpt-oss-20b-Q4_K_M"


N="${1:-${N:-8}}"
ENTRY="${ENTRY%/}"

[[ -n "$SECRET" && -n "$MODEL" ]] || {
  echo "set CLUSTER_SECRET, MODEL (and ENTRY). e.g.:" >&2
  echo "  CLUSTER_SECRET=xxx ENTRY=http://srv1:5000 MODEL=Qwen3.6-35B-A3B-UD-IQ1_M $0 8" >&2
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
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"write ~150 words #$i\"}],\"stream\":false}" )"
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
