#!/bin/bash
# Protocol-level test harness for the llamaman builtin MCP server.
BASE="${BASE:-http://localhost:42069/mcp/knowledge}"
if [ -z "$1" ]; then
  echo "usage: $0 <api-key>   (BASE=... to override the endpoint)" >&2
  exit 2
fi
AUTH="Bearer $1"
PASS=0; FAIL=0
say() { printf '%-58s %s\n' "$1" "$2"; }
check() { # name expected actual
  if [ "$2" = "$3" ]; then PASS=$((PASS+1)); say "$1" "PASS ($2)";
  else FAIL=$((FAIL+1)); say "$1" "FAIL (want=$2 got=$3)"; fi
}

# 1. initialize
INIT=$(curl -s -D /tmp/mcp_h.txt -X POST "$BASE" -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","clientInfo":{"name":"curl-harness","version":"0"},"capabilities":{}}}')
SID=$(grep -i '^Mcp-Session-Id:' /tmp/mcp_h.txt | tr -d '\r' | awk '{print $2}')
check "initialize returns session id" "yes" "$([ -n "$SID" ] && echo yes || echo no)"
check "initialize negotiates 2025-06-18" "yes" "$(echo "$INIT" | grep -q '2025-06-18' && echo yes || echo no)"
check "initialize has instructions" "yes" "$(echo "$INIT" | grep -q '"instructions"' && echo yes || echo no)"
check "initialize advertises listChanged" "yes" "$(echo "$INIT" | grep -q '"listChanged":true' && echo yes || echo no)"

call() { # body [extra curl args...] -> sets CODE BODY
  local body="$1"; shift
  local out; out=$(curl -s -w '\n%{http_code}' -X POST "$BASE" -H "Authorization: $AUTH" \
    -H 'Content-Type: application/json' -H "Mcp-Session-Id: $SID" \
    -H 'MCP-Protocol-Version: 2025-06-18' "$@" --data-binary "$body")
  CODE=$(echo "$out" | tail -1); BODY=$(echo "$out" | head -n -1)
}

# 2. initialized notification -> 202 empty
out=$(curl -s -w '%{http_code}' -o /tmp/mcp_n.txt -X POST "$BASE" -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' -H "Mcp-Session-Id: $SID" \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}')
check "notifications/initialized -> 202" "202" "$out"
check "notification body empty" "0" "$(wc -c < /tmp/mcp_n.txt | tr -d ' ')"

# 3. tools/list -> 7 tools
call '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
check "tools/list -> 200" "200" "$CODE"
NTOOLS=$(echo "$BODY" | grep -o '"name":"kb_[a-z_]*"' | sort -u | wc -l)
check "tools/list tool count" "7" "$NTOOLS"

# 4. ping
call '{"jsonrpc":"2.0","id":3,"method":"ping"}'
check "ping -> 200 empty result" "yes" "$( [ "$CODE" = 200 ] && echo "$BODY" | grep -q '"result":{}' && echo yes || echo no)"

# 5. unknown method -> -32601 at 200
call '{"jsonrpc":"2.0","id":4,"method":"tools/whatever"}'
check "unknown method -> -32601 @200" "yes" "$( [ "$CODE" = 200 ] && echo "$BODY" | grep -q '"code":-32601' && echo yes || echo no)"

# 6. unknown tool -> -32602
call '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"kb_nope","arguments":{}}}'
check "unknown tool -> -32602" "yes" "$(echo "$BODY" | grep -q '"code":-32602' && echo yes || echo no)"

# 7. bad JSON -> -32700 @400
call 'this is not json'
check "malformed JSON -> -32700 @400" "yes" "$( [ "$CODE" = 400 ] && echo "$BODY" | grep -q '"code":-32700' && echo yes || echo no)"

# 8. batch -> -32600 @400
call '[{"jsonrpc":"2.0","id":1,"method":"ping"}]'
check "batch request -> -32600 @400" "yes" "$( [ "$CODE" = 400 ] && echo "$BODY" | grep -q '"code":-32600' && echo yes || echo no)"

# 9. wrong envelope
call '{"jsonrpc":"1.0","id":1,"method":"ping"}'
check "non-2.0 envelope -> -32600 @400" "yes" "$( [ "$CODE" = 400 ] && echo "$BODY" | grep -q '"code":-32600' && echo yes || echo no)"

# 10. missing protocol-version header (negotiated 2025-06-18) -> 400
out=$(curl -s -w '%{http_code}' -o /tmp/mcp_x.txt -X POST "$BASE" -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","id":6,"method":"ping"}')
check "missing MCP-Protocol-Version -> 400" "400" "$out"

# 11. bogus protocol-version header -> 400
out=$(curl -s -w '%{http_code}' -o /dev/null -X POST "$BASE" -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' -H "Mcp-Session-Id: $SID" \
  -H 'MCP-Protocol-Version: 1999-01-01' -d '{"jsonrpc":"2.0","id":7,"method":"ping"}')
check "bogus MCP-Protocol-Version -> 400" "400" "$out"

# 12. unknown session id -> 404 with -32001
out=$(curl -s -w '\n%{http_code}' -X POST "$BASE" -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' -H "Mcp-Session-Id: deadbeefdeadbeefdeadbeefdeadbeef" \
  -H 'MCP-Protocol-Version: 2025-06-18' -d '{"jsonrpc":"2.0","id":8,"method":"ping"}')
CODE=$(echo "$out" | tail -1); BODY=$(echo "$out" | head -n -1)
check "unknown session -> 404 -32001" "yes" "$( [ "$CODE" = 404 ] && echo "$BODY" | grep -q '"code":-32001' && echo yes || echo no)"

# 13. GET -> 405
out=$(curl -s -o /dev/null -w '%{http_code}' -X GET "$BASE" -H "Authorization: $AUTH" -H "Mcp-Session-Id: $SID")
check "GET /mcp/knowledge -> 405" "405" "$out"

# 14. OPTIONS preflight -> 204 + reflected origin
out=$(curl -s -D - -o /dev/null -X OPTIONS "$BASE" -H "Authorization: $AUTH" -H 'Origin: http://example.test')
check "OPTIONS -> 204" "204" "$(echo "$out" | head -1 | grep -o '[0-9]\{3\}')"
check "CORS reflects origin" "yes" "$(echo "$out" | grep -qi 'Access-Control-Allow-Origin: http://example.test' && echo yes || echo no)"

# 15. tool-level errors are isError @200 (not RPC errors)
call '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"kb_get_document","arguments":{"document_id":99999999}}}'
check "get missing doc -> isError @200" "yes" "$( [ "$CODE" = 200 ] && echo "$BODY" | grep -q '"isError":true' && echo yes || echo no)"
call '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"kb_search","arguments":{"query":"   "}}}'
check "blank query -> isError @200" "yes" "$( [ "$CODE" = 200 ] && echo "$BODY" | grep -q '"isError":true' && echo yes || echo no)"
call '{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"kb_get_document","arguments":{"document_id":"not-an-int"}}}'
check "non-int doc id -> isError @200" "yes" "$( [ "$CODE" = 200 ] && echo "$BODY" | grep -q '"isError":true' && echo yes || echo no)"

# 16. ingest over 200KB cap -> isError @200 (payload built in-process; too big for argv)
python3 - "$BASE" "$AUTH" "$SID" <<'EOF' > /tmp/mcp_big.txt
import json, sys, urllib.request
base, auth, sid = sys.argv[1:4]
big = 'x' * 210000
req = urllib.request.Request(base, data=json.dumps({
    "jsonrpc": "2.0", "id": 12, "method": "tools/call",
    "params": {"name": "kb_ingest_document", "arguments": {
        "topic": "mcp-e2e-test", "title": "too big", "content": big}}}).encode(),
    headers={"Authorization": auth, "Content-Type": "application/json",
             "Mcp-Session-Id": sid, "MCP-Protocol-Version": "2025-06-18"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print(r.status); print(r.read().decode())
except urllib.error.HTTPError as e:
    print(e.code); print(e.read().decode())
EOF
BIGCODE=$(head -1 /tmp/mcp_big.txt); BIGBODY=$(tail -n +2 /tmp/mcp_big.txt)
check "oversize ingest -> isError @200" "yes" "$( [ "$BIGCODE" = 200 ] && echo "$BIGBODY" | grep -q '"isError":true' && echo yes || echo no)"
say "oversize ingest message" "$(echo "$BIGBODY" | head -c 160)"

# 17. real tool call over raw protocol
call '{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"kb_list_topics","arguments":{}}}'
check "tools/call kb_list_topics -> 200" "200" "$CODE"

# 18. legacy protocol negotiation: 2024-11-05 exempt from header requirement
INIT2=$(curl -s -D /tmp/mcp_h2.txt -X POST "$BASE" -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"curl-h2","version":"0"},"capabilities":{}}}')
SID2=$(grep -i '^Mcp-Session-Id:' /tmp/mcp_h2.txt | tr -d '\r' | awk '{print $2}')
check "initialize 2024-11-05 negotiated" "yes" "$(echo "$INIT2" | grep -q '"protocolVersion":"2024-11-05"' && echo yes || echo no)"
out=$(curl -s -w '%{http_code}' -o /dev/null -X POST "$BASE" -H "Authorization: $AUTH" \
  -H 'Content-Type: application/json' -H "Mcp-Session-Id: $SID2" \
  -d '{"jsonrpc":"2.0","id":2,"method":"ping"}')
check "2024-11-05 session works w/o version header" "200" "$out"

# 19. unsupported requested version falls back to latest
INIT3=$(curl -s -X POST "$BASE" -H "Authorization: $AUTH" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2099-01-01","clientInfo":{},"capabilities":{}}}')
check "unsupported version -> latest fallback" "yes" "$(echo "$INIT3" | grep -q '"protocolVersion":"2025-06-18"' && echo yes || echo no)"

# 20. no auth -> 401
out=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE" -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}')
check "no bearer token -> 401" "401" "$out"

echo; echo "RESULT: $PASS passed, $FAIL failed"
