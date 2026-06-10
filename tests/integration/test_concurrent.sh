#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Integration test: concurrent request handling
#
# Fires N parallel chat completion requests at any OpenAI-compatible endpoint
# (/v1/chat/completions). Works against both the llamaman proxy and direct
# llama-server instances.
#
# Usage:
#   ./test_concurrent.sh HOST PORT MODEL                     # required args
#   ./test_concurrent.sh http://nullsrv 11434 omnicoder-9b-q8_0
#   ./test_concurrent.sh http://nullsrv 42069 omnicoder-9b-q8_0
#
# Environment:
#   NUM_REQUESTS - Number of parallel requests to send (default: 10)
#   API_KEY      - Bearer token if require_auth is enabled
#   FORMAT       - "openai" (default), "ollama", or "both"
#                  Ollama format only works on the llamaman proxy, not direct instances
# ---------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: ${0} HOST PORT MODEL"
  echo "  e.g. ${0} http://nullsrv 11434 omnicoder-9b-q8_0"
  echo "  e.g. ${0} http://nullsrv 42069 omnicoder-9b-q8_0"
  exit 1
fi

HOST="${1%/}"
PORT="${2}"
MODEL="${3}"

NUM_REQUESTS="${NUM_REQUESTS:-10}"
FORMAT="${FORMAT:-openai}"
API_KEY="${API_KEY:-}"

BASE_URL="${HOST}:${PORT}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
TOTAL=0

AUTH_HEADER=()
if [[ -n "${API_KEY}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${API_KEY}")
fi

# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------
echo ""
printf "${CYAN}=== llamaMan concurrent request test ===${NC}\n"
printf "  Target:   %s\n" "${BASE_URL}"
printf "  Model:    %s\n" "${MODEL}"
printf "  Requests: %d\n" "${NUM_REQUESTS}"
printf "  Format:   %s\n" "${FORMAT}"

# Try /v1/models first (works on both proxy and direct instance), fall back to /health
if ! curl -sf -o /dev/null --max-time 5 "${AUTH_HEADER[@]}" "${BASE_URL}/v1/models" 2>/dev/null; then
  if ! curl -sf -o /dev/null --max-time 5 "${AUTH_HEADER[@]}" "${BASE_URL}/health" 2>/dev/null; then
    echo "ERROR: Cannot reach ${BASE_URL} - is the server running?"
    echo "       If require_auth is enabled, set API_KEY=llm-xxx"
    exit 1
  fi
fi

echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "${RESULTS_DIR}"' EXIT

function fireOllamaRequest {
  local reqIndex="${1}"
  local startTime endTime elapsed statusCode
  startTime=$(date +%s%N)

  local resp
  resp=$(curl -s -w "\n%{http_code}" --max-time 300 \
    "${AUTH_HEADER[@]}" \
    -X POST "${BASE_URL}/api/chat" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say the number ${reqIndex} and nothing else.\"}],\"stream\":false}" 2>/dev/null || echo -e "\n000")

  statusCode=$(echo "${resp}" | tail -1)
  local respBody
  respBody=$(echo "${resp}" | sed '$d')

  endTime=$(date +%s%N)
  elapsed=$(( (endTime - startTime) / 1000000 ))

  echo "${statusCode}|${elapsed}|ollama|${reqIndex}" > "${RESULTS_DIR}/ollama_${reqIndex}.result"

  if [[ "${statusCode}" -ge 200 ]] && [[ "${statusCode}" -lt 300 ]]; then
    local content
    content=$(echo "${respBody}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message',{}).get('content','(empty)'))" 2>/dev/null || echo "(parse error)")
    printf "${GREEN}  OK${NC}  [ollama #%02d] HTTP %s in %dms - %s\n" "${reqIndex}" "${statusCode}" "${elapsed}" "${content}"
  else
    printf "${RED}  ERR${NC} [ollama #%02d] HTTP %s in %dms\n" "${reqIndex}" "${statusCode}" "${elapsed}"
  fi
}

function fireOpenaiRequest {
  local reqIndex="${1}"
  local startTime endTime elapsed statusCode
  startTime=$(date +%s%N)

  local resp
  resp=$(curl -s -w "\n%{http_code}" --max-time 300 \
    "${AUTH_HEADER[@]}" \
    -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say the number ${reqIndex} and nothing else.\"}],\"stream\":false}" 2>/dev/null || echo -e "\n000")

  statusCode=$(echo "${resp}" | tail -1)
  local respBody
  respBody=$(echo "${resp}" | sed '$d')

  endTime=$(date +%s%N)
  elapsed=$(( (endTime - startTime) / 1000000 ))

  echo "${statusCode}|${elapsed}|openai|${reqIndex}" > "${RESULTS_DIR}/openai_${reqIndex}.result"

  if [[ "${statusCode}" -ge 200 ]] && [[ "${statusCode}" -lt 300 ]]; then
    local content
    content=$(echo "${respBody}" | python3 -c "import sys,json; c=json.load(sys.stdin)['choices'][0]['message']['content']; print(c)" 2>/dev/null || echo "(parse error)")
    local tokenUsage
    tokenUsage=$(echo "${respBody}" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(f\"{u.get('prompt_tokens','?')} in / {u.get('completion_tokens','?')} out\")" 2>/dev/null || echo "")
    printf "${GREEN}  OK${NC}  [openai #%02d] HTTP %s in %dms - %s [%s]\n" "${reqIndex}" "${statusCode}" "${elapsed}" "${content}" "${tokenUsage}"
  else
    printf "${RED}  ERR${NC} [openai #%02d] HTTP %s in %dms\n" "${reqIndex}" "${statusCode}" "${elapsed}"
  fi
}

# ---------------------------------------------------------------------------
# Fire requests
# ---------------------------------------------------------------------------
if [[ "${FORMAT}" == "ollama" ]] || [[ "${FORMAT}" == "both" ]]; then
  printf "${CYAN}--- Ollama format: /api/chat (%d parallel requests) ---${NC}\n" "${NUM_REQUESTS}"
  for i in $(seq 1 "${NUM_REQUESTS}"); do
    fireOllamaRequest "${i}" &
  done
  wait
  echo ""
fi

if [[ "${FORMAT}" == "openai" ]] || [[ "${FORMAT}" == "both" ]]; then
  printf "${CYAN}--- OpenAI format: /v1/chat/completions (%d parallel requests) ---${NC}\n" "${NUM_REQUESTS}"
  for i in $(seq 1 "${NUM_REQUESTS}"); do
    fireOpenaiRequest "${i}" &
  done
  wait
  echo ""
fi

# ---------------------------------------------------------------------------
# Tally results
# ---------------------------------------------------------------------------
printf "${CYAN}--- Results ---${NC}\n"

for resultFile in "${RESULTS_DIR}"/*.result; do
  [[ -f "${resultFile}" ]] || continue
  ((TOTAL++))
  statusCode=$(cut -d'|' -f1 < "${resultFile}")
  if [[ "${statusCode}" -ge 200 ]] && [[ "${statusCode}" -lt 300 ]]; then
    ((PASS++))
  else
    ((FAIL++))
  fi
done

# Timing stats
if [[ "${TOTAL}" -gt 0 ]]; then
  timings=()
  for resultFile in "${RESULTS_DIR}"/*.result; do
    [[ -f "${resultFile}" ]] || continue
    timings+=($(cut -d'|' -f2 < "${resultFile}"))
  done

  minTime=${timings[0]} maxTime=${timings[0]} sumTime=0
  for t in "${timings[@]}"; do
    sumTime=$((${sumTime} + ${t}))
    [[ "${t}" -lt "${minTime}" ]] && minTime=${t}
    [[ "${t}" -gt "${maxTime}" ]] && maxTime=${t}
  done
  avgTime=$((${sumTime} / ${#timings[@]}))

  printf "  Total:   %d requests\n" "${TOTAL}"
  printf "${GREEN}  Pass:    %d${NC}\n" "${PASS}"
  if [[ "${FAIL}" -gt 0 ]]; then
    printf "${RED}  Fail:    %d${NC}\n" "${FAIL}"
  fi
  printf "  Timing:  min=%dms  avg=%dms  max=%dms\n" "${minTime}" "${avgTime}" "${maxTime}"
fi

echo ""
if [[ "${FAIL}" -gt 0 ]]; then
  printf "${RED}SOME REQUESTS FAILED${NC}\n"
  exit 1
else
  printf "${GREEN}ALL REQUESTS SUCCEEDED${NC}\n"
  exit 0
fi
