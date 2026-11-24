#!/usr/bin/env bash
# Verify a running ORCA deployment is healthy AND cannot reach a broker.
# The safety assertions matter more than the liveness ones: a staging box that
# has quietly been pointed at a live gateway is worse than one that is down.
#
#   ./scripts/staging_smoke.sh [base_url]
set -uo pipefail

BASE="${1:-http://127.0.0.1:8000}"
PASS=0; FAIL=0

ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
check(){ if [ "$2" = "$3" ]; then ok "$1 = $3"; else bad "$1: expected '$3', got '$2'"; fi; }

jqf() { # extract a top-level-ish json value without requiring jq
  python3 -c "
import json,sys,functools
d=json.load(sys.stdin)
try:
    print(functools.reduce(lambda o,k: o[k], '$1'.split('.'), d))
except Exception:
    print('<missing>')"
}

echo "ORCA smoke test -> $BASE"
echo

echo "liveness"
root=$(curl -fsS --max-time 5 "$BASE/" 2>/dev/null) || { bad "GET / unreachable"; echo; echo "  is it running?  make staging-up"; exit 1; }
ok "GET / reachable"
curl -fsS --max-time 5 "$BASE/docs"         >/dev/null 2>&1 && ok "GET /docs"         || bad "GET /docs"
curl -fsS --max-time 5 "$BASE/openapi.json" >/dev/null 2>&1 && ok "GET /openapi.json" || bad "GET /openapi.json"

health=$(curl -fsS --max-time 10 "$BASE/api/health" 2>/dev/null) || { bad "GET /api/health"; exit 1; }
ok "GET /api/health"

echo
echo "safety - nothing may reach a broker"
check "trading_enabled"  "$(echo "$health" | jqf trading_enabled)" "False"
check "gateway"          "$(echo "$health" | jqf gateway.gateway)" "paper"
check "gateway.live"     "$(echo "$health" | jqf gateway.live)"    "False"

echo
echo "wiring"
check "env"              "$(echo "$health" | jqf env)"             "staging"
check "feed ok"          "$(echo "$health" | jqf feed.ok)"         "True"
printf '  \033[36mINFO\033[0m  feed provider = %s\n' "$(echo "$health" | jqf feed.provider)"
printf '  \033[36mINFO\033[0m  market_open   = %s\n' "$(echo "$health" | jqf market_open)"

model_fitted=$(echo "$health" | jqf model.is_fitted)
synthetic=$(echo "$health" | jqf model.synthetic)
check "model loaded"     "$model_fitted" "True"
if [ "$synthetic" = "True" ]; then
  printf '  \033[33mWARN\033[0m  model is SYNTHETIC - ranking plumbing only, do not trade off it\n'
else
  ok "model is not synthetic"
fi

echo
echo "read endpoints"
for p in /api/market/underlyings /api/signals /api/portfolio /api/model; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE$p")
  [ "$code" = "200" ] && ok "GET $p -> 200" || bad "GET $p -> $code"
done

echo
echo "scan produces candidates"
scan=$(curl -fsS --max-time 30 -X POST "$BASE/api/signals/scan" \
  -H 'Content-Type: application/json' \
  -d '{"symbols":["SENSEX"],"lots":1,"limit":5,"persist":true}' 2>/dev/null)
if [ -n "$scan" ]; then
  n=$(echo "$scan" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')
  if [ "$n" -gt 0 ]; then
    ok "POST /api/signals/scan -> $n candidates"
    echo "$scan" | python3 -c "
import json,sys
for c in json.load(sys.stdin)[:3]:
    legs=' '.join(f\"{l['side'][0]}{int(l['strike'])}{l['option_type']}\" for l in c['legs'])
    print(f\"        {c['score']}  {c['strategy_id']:<24} credit={c['net_credit']:>8,.0f}  {legs}\")"
  else
    printf '  \033[33mWARN\033[0m  scan returned 0 candidates (expiry-day / credit floor / market closed)\n'
  fi
else
  bad "POST /api/signals/scan"
fi

echo
echo "-------------------------------------------"
printf '  %d passed, %d failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
echo "  staging OK"
