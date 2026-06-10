#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Integration test: bearer-token auth enforcement on ALL ports
#
# Tests endpoints on:
#   - Main Flask port (default 5000)      - management UI + API
#   - Llamaman proxy port (default 42069) - Ollama-compatible API (OpenWebUI)
#   - Per-instance proxy ports (8000+)    - direct model access
#
# Usage:
#   ./test_auth.sh                                            # all defaults
#   ./test_auth.sh http://192.168.1.50                        # custom host
#   MAIN_PORT=5000 PROXY_PORT=42069 ./test_auth.sh            # custom ports
#   API_KEY=llm-xxx ./test_auth.sh                            # existing key
#
# Environment:
#   TEST_USER / TEST_PASS  - skip interactive login prompt
#   MAIN_PORT              - Flask/gunicorn port (default: 5000)
#   PROXY_PORT             - Llamaman proxy port (default: 42069)
#   INSTANCE_PORT          - A running instance proxy port (optional, skipped if unset)
#   API_KEY                - Use an existing API key instead of creating one
# ---------------------------------------------------------------------------
set -euo pipefail

HOST="${1:-http://localhost}"
# Strip trailing slash
HOST="${HOST%/}"

MAIN_PORT="${MAIN_PORT:-5000}"
PROXY_PORT="${PROXY_PORT:-42069}"
INSTANCE_PORT="${INSTANCE_PORT:-}"

MAIN_URL="${HOST}:${MAIN_PORT}"
PROXY_URL="${HOST}:${PROXY_PORT}"
INSTANCE_URL=""
[[ -n "${INSTANCE_PORT}" ]] && INSTANCE_URL="${HOST}:${INSTANCE_PORT}"

PASS=0
FAIL=0
SKIP=0
FAILURES=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function assertStatus {
  local testName="${1}" expected="${2}" actual="${3}"
  if [[ "${actual}" -eq "${expected}" ]]; then
    printf "${GREEN}  PASS${NC} %s (HTTP %s)\n" "${testName}" "${actual}"
    ((PASS++))
  else
    printf "${RED}  FAIL${NC} %s - expected %s, got %s\n" "${testName}" "${expected}" "${actual}"
    ((FAIL++))
    FAILURES="${FAILURES}\n  - ${testName} (expected ${expected}, got ${actual})"
  fi
}

function assertNotStatus {
  local testName="${1}" notExpected="${2}" actual="${3}"
  if [[ "${actual}" -ne "${notExpected}" ]]; then
    printf "${GREEN}  PASS${NC} %s (HTTP %s)\n" "${testName}" "${actual}"
    ((PASS++))
  else
    printf "${RED}  FAIL${NC} %s - expected NOT %s, got %s\n" "${testName}" "${notExpected}" "${actual}"
    ((FAIL++))
    FAILURES="${FAILURES}\n  - ${testName} (expected NOT ${notExpected}, got ${actual})"
  fi
}

function skipTest {
  local testName="${1}" reason="${2}"
  printf "${YELLOW}  SKIP${NC} %s - %s\n" "${testName}" "${reason}"
  ((SKIP++))
}

function httpStatus {
  local reqMethod="${1}" reqUrl="${2}"
  shift 2
  curl -sf -o /dev/null -w '%{http_code}' -X "${reqMethod}" "$@" "${reqUrl}" 2>/dev/null || \
  curl -s -o /dev/null -w '%{http_code}' -X "${reqMethod}" "$@" "${reqUrl}" 2>/dev/null
}

# ---------------------------------------------------------------------------
# Session login
# ---------------------------------------------------------------------------
COOKIE_JAR=$(mktemp)
trap 'rm -f "${COOKIE_JAR}"' EXIT

function sessionLogin {
  local userName userPass
  if [[ -z "${TEST_USER:-}" ]] || [[ -z "${TEST_PASS:-}" ]]; then
    read -rp "Admin username: " userName
    read -rsp "Admin password: " userPass
    echo
  else
    userName="${TEST_USER}"
    userPass="${TEST_PASS}"
  fi

  local statusCode
  statusCode=$(curl -s -o /dev/null -w '%{http_code}' \
    -X POST "${MAIN_URL}/login" \
    -c "${COOKIE_JAR}" -b "${COOKIE_JAR}" -L \
    -d "username=${userName}&password=${userPass}")

  if [[ "${statusCode}" -ne 200 ]]; then
    echo "ERROR: Login failed (HTTP ${statusCode}). Check credentials."
    exit 1
  fi
  echo "Logged in as ${userName}"
}

# ---------------------------------------------------------------------------
# API key management (via session)
# ---------------------------------------------------------------------------
function createApiKey {
  local resp
  resp=$(curl -s -X POST "${MAIN_URL}/api/api-keys" \
    -b "${COOKIE_JAR}" \
    -H "Content-Type: application/json" \
    -d '{"name":"integration-test"}')
  echo "${resp}" | jq -r '.key // empty'
}

function deleteApiKeyByName {
  local keyName="${1}"
  local keysList
  keysList=$(curl -s "${MAIN_URL}/api/api-keys" -b "${COOKIE_JAR}")
  local keyId
  keyId=$(echo "${keysList}" | jq -r ".[] | select(.name==\"${keyName}\") | .id" | head -1)
  if [[ -n "${keyId}" ]]; then
    curl -s -o /dev/null -X DELETE "${MAIN_URL}/api/api-keys/${keyId}" -b "${COOKIE_JAR}"
  fi
}

# ---------------------------------------------------------------------------
# Settings toggle
# ---------------------------------------------------------------------------
function setRequireAuth {
  local authValue="${1}"  # true or false
  local currentSettings
  currentSettings=$(curl -s "${MAIN_URL}/api/settings" -b "${COOKIE_JAR}")
  local updatedSettings
  updatedSettings=$(echo "${currentSettings}" | jq ". + {require_auth: ${authValue}}")
  curl -s -o /dev/null -X POST "${MAIN_URL}/api/settings" \
    -b "${COOKIE_JAR}" \
    -H "Content-Type: application/json" \
    -d "${updatedSettings}"
}

# ---------------------------------------------------------------------------
# Test an endpoint with no token, bad token, and valid token
# ---------------------------------------------------------------------------
function testEndpointProtected {
  local label="${1}" method="${2}" url="${3}"
  local bodyArgs=()
  if [[ "${method}" == "POST" ]]; then
    bodyArgs=(-H "Content-Type: application/json" -d '{}')
  fi

  local statusCode
  # No token -> 401
  statusCode=$(httpStatus "${method}" "${url}" "${bodyArgs[@]}")
  assertStatus "[${label}] ${method} ${url} - no token => 401" 401 "${statusCode}"

  # Bad token -> 401
  statusCode=$(httpStatus "${method}" "${url}" -H "Authorization: Bearer bad-token-12345" "${bodyArgs[@]}")
  assertStatus "[${label}] ${method} ${url} - bad token => 401" 401 "${statusCode}"

  # Valid token -> NOT 401
  statusCode=$(httpStatus "${method}" "${url}" -H "Authorization: Bearer ${API_KEY}" "${bodyArgs[@]}")
  assertNotStatus "[${label}] ${method} ${url} - valid token => not 401" 401 "${statusCode}"
}

function testEndpointOpen {
  local label="${1}" method="${2}" url="${3}"
  local bodyArgs=()
  if [[ "${method}" == "POST" ]]; then
    bodyArgs=(-H "Content-Type: application/json" -d '{}')
  fi

  local statusCode
  statusCode=$(httpStatus "${method}" "${url}" "${bodyArgs[@]}")
  assertNotStatus "[${label}] ${method} ${url} - no token => not 401" 401 "${statusCode}"
}

# ---------------------------------------------------------------------------
# Endpoint lists
# ---------------------------------------------------------------------------
LLAMAMAN_PATHS=(
  "GET|/api/tags"
  "GET|/api/version"
  "POST|/api/show"
  "GET|/api/ps"
  "POST|/api/chat"
  "POST|/api/generate"
  "GET|/v1/models"
  "POST|/v1/chat/completions"
)

MANAGEMENT_PATHS=(
  "GET|/api/models"
  "GET|/api/instances"
  "GET|/api/downloads"
  "GET|/api/presets"
  "GET|/api/settings"
  "GET|/api/system-info"
  "GET|/api/gpu-info"
  "GET|/api/api-keys"
  "GET|/api/next-port"
  "GET|/api/disk-space"
  "GET|/api/model-layers"
)

ALWAYS_OPEN_PATHS=(
  "GET|/health"
  "GET|/login"
  "GET|/setup"
)

# Proxy port paths (llama.cpp server compatible)
PROXY_PATHS=(
  "GET|/v1/models"
  "POST|/v1/chat/completions"
  "GET|/api/tags"
  "POST|/api/chat"
  "POST|/api/generate"
  "GET|/api/ps"
)

# Instance proxy paths (raw llama-server endpoints)
INSTANCE_PROXY_PATHS=(
  "GET|/health"
  "POST|/v1/chat/completions"
  "POST|/completion"
  "GET|/v1/models"
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo ""
printf "${CYAN}=== llamaMan auth integration tests ===${NC}\n"
printf "  Main:     %s\n" "${MAIN_URL}"
printf "  Proxy:    %s\n" "${PROXY_URL}"
if [[ -n "${INSTANCE_URL}" ]]; then
  printf "  Instance: %s\n" "${INSTANCE_URL}"
else
  printf "  Instance: ${YELLOW}(skipped - set INSTANCE_PORT to test)${NC}\n"
fi
echo ""

# Check connectivity
if ! curl -sf -o /dev/null "${MAIN_URL}/health"; then
  echo "ERROR: Cannot reach ${MAIN_URL}/health - is the server running?"
  exit 1
fi

PROXY_REACHABLE=1
if ! curl -sf -o /dev/null --max-time 3 "${PROXY_URL}/health" 2>/dev/null; then
  printf "${YELLOW}WARNING: Cannot reach proxy at %s - proxy port tests will be skipped${NC}\n" "${PROXY_URL}"
  PROXY_REACHABLE=0
fi

INSTANCE_REACHABLE=0
if [[ -n "${INSTANCE_URL}" ]]; then
  if curl -sf -o /dev/null --max-time 3 "${INSTANCE_URL}/health" 2>/dev/null; then
    INSTANCE_REACHABLE=1
  else
    printf "${YELLOW}WARNING: Cannot reach instance at %s - instance proxy tests will be skipped${NC}\n" "${INSTANCE_URL}"
  fi
fi

# Login & key setup
sessionLogin

if [[ -z "${API_KEY:-}" ]]; then
  echo "Creating test API key..."
  API_KEY=$(createApiKey)
  if [[ -z "${API_KEY}" ]]; then
    echo "ERROR: Could not create API key."
    exit 1
  fi
  CREATED_KEY=1
  echo "Test key: ${API_KEY:0:12}..."
else
  CREATED_KEY=0
  echo "Using provided API_KEY: ${API_KEY:0:12}..."
fi

echo ""

# =============================================
# PHASE 1: require_auth = true (default)
# =============================================
printf "${CYAN}--- Phase 1: require_auth = true ---${NC}\n"
setRequireAuth true
sleep 0.5

# -- Main port: llamaman endpoints --
echo ""
echo "  [Port ${MAIN_PORT}] Llamaman endpoints (should require auth):"
for ep in "${LLAMAMAN_PATHS[@]}"; do
  IFS='|' read -r method path <<< "${ep}"
  testEndpointProtected "auth=ON,main" "${method}" "${MAIN_URL}${path}"
done

# -- Main port: management endpoints --
echo ""
echo "  [Port ${MAIN_PORT}] Management endpoints (should require auth):"
for ep in "${MANAGEMENT_PATHS[@]}"; do
  IFS='|' read -r method path <<< "${ep}"
  testEndpointProtected "auth=ON,main" "${method}" "${MAIN_URL}${path}"
done

# -- Main port: always-open endpoints --
echo ""
echo "  [Port ${MAIN_PORT}] Always-open endpoints:"
for ep in "${ALWAYS_OPEN_PATHS[@]}"; do
  IFS='|' read -r method path <<< "${ep}"
  testEndpointOpen "auth=ON,main" "${method}" "${MAIN_URL}${path}"
done

# -- Proxy port (42069) --
echo ""
echo "  [Port ${PROXY_PORT}] Proxy port - Ollama-compatible API (should require auth):"
if [[ "${PROXY_REACHABLE}" -eq 1 ]]; then
  for ep in "${PROXY_PATHS[@]}"; do
    IFS='|' read -r method path <<< "${ep}"
    testEndpointProtected "auth=ON,proxy" "${method}" "${PROXY_URL}${path}"
  done
else
  skipTest "[auth=ON,proxy] all proxy endpoints" "proxy port unreachable"
fi

# -- Instance proxy port --
echo ""
echo "  [Port ${INSTANCE_PORT:-N/A}] Per-instance proxy (should require auth):"
if [[ "${INSTANCE_REACHABLE}" -eq 1 ]]; then
  for ep in "${INSTANCE_PROXY_PATHS[@]}"; do
    IFS='|' read -r method path <<< "${ep}"
    testEndpointProtected "auth=ON,instance" "${method}" "${INSTANCE_URL}${path}"
  done
else
  skipTest "[auth=ON,instance] all instance proxy endpoints" "no instance port or unreachable"
fi

# =============================================
# PHASE 2: require_auth = false
# =============================================
echo ""
printf "${CYAN}--- Phase 2: require_auth = false ---${NC}\n"
setRequireAuth false
sleep 0.5

# -- Main port: llamaman endpoints should be OPEN --
echo ""
echo "  [Port ${MAIN_PORT}] Llamaman endpoints (should be OPEN):"
for ep in "${LLAMAMAN_PATHS[@]}"; do
  IFS='|' read -r method path <<< "${ep}"
  testEndpointOpen "auth=OFF,main" "${method}" "${MAIN_URL}${path}"
done

# -- Main port: management endpoints still require auth --
echo ""
echo "  [Port ${MAIN_PORT}] Management endpoints (should STILL require auth):"
for ep in "${MANAGEMENT_PATHS[@]}"; do
  IFS='|' read -r method path <<< "${ep}"

  bodyArgs=()
  if [[ "${method}" == "POST" ]]; then
    bodyArgs=(-H "Content-Type: application/json" -d '{}')
  fi

  statusCode=$(httpStatus "${method}" "${MAIN_URL}${path}" "${bodyArgs[@]}")
  assertStatus "[auth=OFF,main] ${method} ${MAIN_URL}${path} - no token => 401" 401 "${statusCode}"

  statusCode=$(httpStatus "${method}" "${MAIN_URL}${path}" -H "Authorization: Bearer ${API_KEY}" "${bodyArgs[@]}")
  assertNotStatus "[auth=OFF,main] ${method} ${MAIN_URL}${path} - valid token => not 401" 401 "${statusCode}"
done

# -- Proxy port should be OPEN --
echo ""
echo "  [Port ${PROXY_PORT}] Proxy port (should be OPEN):"
if [[ "${PROXY_REACHABLE}" -eq 1 ]]; then
  for ep in "${PROXY_PATHS[@]}"; do
    IFS='|' read -r method path <<< "${ep}"
    testEndpointOpen "auth=OFF,proxy" "${method}" "${PROXY_URL}${path}"
  done
else
  skipTest "[auth=OFF,proxy] all proxy endpoints" "proxy port unreachable"
fi

# -- Instance proxy should be OPEN --
echo ""
echo "  [Port ${INSTANCE_PORT:-N/A}] Per-instance proxy (should be OPEN):"
if [[ "${INSTANCE_REACHABLE}" -eq 1 ]]; then
  for ep in "${INSTANCE_PROXY_PATHS[@]}"; do
    IFS='|' read -r method path <<< "${ep}"
    testEndpointOpen "auth=OFF,instance" "${method}" "${INSTANCE_URL}${path}"
  done
else
  skipTest "[auth=OFF,instance] all instance proxy endpoints" "no instance port or unreachable"
fi

# -- Always-open still open --
echo ""
echo "  [Port ${MAIN_PORT}] Always-open endpoints (should still be open):"
for ep in "${ALWAYS_OPEN_PATHS[@]}"; do
  IFS='|' read -r method path <<< "${ep}"
  testEndpointOpen "auth=OFF,main" "${method}" "${MAIN_URL}${path}"
done

# =============================================
# PHASE 3: Edge cases
# =============================================
echo ""
printf "${CYAN}--- Phase 3: Edge cases ---${NC}\n"
setRequireAuth true
sleep 0.5

echo ""
echo "  Malformed auth headers on main port:"
statusCode=$(httpStatus "GET" "${MAIN_URL}/api/tags" -H "Authorization: Bearer ")
assertStatus "[edge] empty bearer value => 401" 401 "${statusCode}"

statusCode=$(httpStatus "GET" "${MAIN_URL}/api/tags" -H "Authorization: Basic dXNlcjpwYXNz")
assertStatus "[edge] Basic auth header => 401" 401 "${statusCode}"

statusCode=$(httpStatus "GET" "${MAIN_URL}/api/tags" -H "Authorization: token ${API_KEY}")
assertStatus "[edge] 'token' prefix instead of 'Bearer' => 401" 401 "${statusCode}"

if [[ "${PROXY_REACHABLE}" -eq 1 ]]; then
  echo ""
  echo "  Malformed auth headers on proxy port:"
  statusCode=$(httpStatus "GET" "${PROXY_URL}/api/tags" -H "Authorization: Bearer ")
  assertStatus "[edge,proxy] empty bearer value => 401" 401 "${statusCode}"

  statusCode=$(httpStatus "GET" "${PROXY_URL}/api/tags" -H "Authorization: Basic dXNlcjpwYXNz")
  assertStatus "[edge,proxy] Basic auth header => 401" 401 "${statusCode}"

  statusCode=$(httpStatus "GET" "${PROXY_URL}/v1/models" -H "Authorization: token ${API_KEY}")
  assertStatus "[edge,proxy] 'token' prefix instead of 'Bearer' => 401" 401 "${statusCode}"
fi

if [[ "${INSTANCE_REACHABLE}" -eq 1 ]]; then
  echo ""
  echo "  Malformed auth headers on instance proxy port:"
  statusCode=$(httpStatus "GET" "${INSTANCE_URL}/health" -H "Authorization: Bearer ")
  assertStatus "[edge,instance] empty bearer value => 401" 401 "${statusCode}"

  statusCode=$(httpStatus "GET" "${INSTANCE_URL}/health" -H "Authorization: Basic dXNlcjpwYXNz")
  assertStatus "[edge,instance] Basic auth header => 401" 401 "${statusCode}"
fi

# =============================================
# Cleanup
# =============================================
echo ""
printf "${CYAN}--- Cleanup ---${NC}\n"

setRequireAuth true
echo "  Restored require_auth = true"

if [[ "${CREATED_KEY}" -eq 1 ]]; then
  deleteApiKeyByName "integration-test"
  echo "  Deleted test API key"
fi

# =============================================
# Summary
# =============================================
echo ""
printf "${CYAN}=== Results ===${NC}\n"
printf "${GREEN}  PASS: %d${NC}\n" "${PASS}"
if [[ "${SKIP}" -gt 0 ]]; then
  printf "${YELLOW}  SKIP: %d${NC}\n" "${SKIP}"
fi
if [[ "${FAIL}" -gt 0 ]]; then
  printf "${RED}  FAIL: %d${NC}\n" "${FAIL}"
  printf "${RED}  Failures:${NC}"
  printf "${FAILURES}\n"
else
  printf "${GREEN}  FAIL: 0${NC}\n"
fi

echo ""
if [[ "${FAIL}" -gt 0 ]]; then
  printf "${RED}FAILED${NC}\n"
  exit 1
else
  printf "${GREEN}ALL TESTS PASSED${NC}\n"
  exit 0
fi
