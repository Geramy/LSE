#!/usr/bin/env bash
# End-to-end check: the engine still answers, and the HTTP surface still
# speaks the wire format. Run by CI and by hand:
#
#   .github/scripts/smoke.sh [build-dir]
#
# LSE_SMOKE_MODEL picks the model; it must be in the HF cache already, since
# this asserts on what it says.
set -euo pipefail

BUILD="${1:-build}"
MODEL="${LSE_SMOKE_MODEL:-mlx-community/Qwen3.5-0.8B-4bit}"
PORT="${LSE_SMOKE_PORT:-8171}"
HOST=127.0.0.1
fails=0

pass() { printf '  ok    %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }

# A greedy answer is deterministic, so this asserts on the text rather than on
# the exit code. Anything else means the engine is wrong, not merely slow.
echo "== engine =="
out=$("$BUILD/lse" -m "$MODEL" -n 16 -t 0 "The capital of France is" 2>/dev/null || true)
if grep -qi 'paris' <<<"$out"; then
  pass "greedy decode answers with the fact"
else
  fail "greedy decode: expected Paris, got: $(head -c 120 <<<"$out")"
fi

# Recall and arithmetic fail differently. A model whose weights are bound to
# the wrong tensors still recites a fact it saw a thousand times; getting a sum
# right needs the whole forward pass to be correct.
out=$("$BUILD/lse" -m "$MODEL" -n 40 -t 0 "What is 4 + 4? Answer with just the number." 2>/dev/null || true)
if grep -q '8' <<<"$out"; then
  pass "greedy decode does arithmetic"
else
  fail "greedy decode: expected 8, got: $(head -c 120 <<<"$out")"
fi

echo "== server =="
"$BUILD/lse-server" -m "$MODEL" --host "$HOST" --port "$PORT" >/tmp/lse-smoke-server.log 2>&1 &
server_pid=$!
# The server is up when it answers, not when the process exists: loading the
# weights takes longer than starting the listener.
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf -m 2 "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$server_pid" 2>/dev/null || { echo "server died:"; tail -20 /tmp/lse-smoke-server.log; exit 1; }
  sleep 2
done

curl -sf -m 5 "http://$HOST:$PORT/health" >/dev/null && pass "GET /health" || fail "GET /health"

if curl -sf -m 5 "http://$HOST:$PORT/v1/models" | grep -q '"object":"list"'; then
  pass "GET /v1/models"
else
  fail "GET /v1/models"
fi

body=$(curl -sf -m 300 "http://$HOST:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":32,"temperature":0}' || true)
if python3 - "$body" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
c = d["choices"][0]
assert d["object"] == "chat.completion", d["object"]
assert c["message"]["role"] == "assistant"
assert "paris" in c["message"]["content"].lower(), c["message"]["content"]
assert c["finish_reason"] in ("stop", "length")
u = d["usage"]
assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0
assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
PY
then pass "POST /v1/chat/completions"; else fail "POST /v1/chat/completions: $(head -c 200 <<<"$body")"; fi

body=$(curl -sf -m 300 "http://$HOST:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 4 + 4? Answer with just the number."}],"max_tokens":48,"temperature":0}' || true)
if python3 - "$body" <<'ARITH'
import json, sys
c = json.loads(sys.argv[1])["choices"][0]["message"]["content"]
assert "8" in c, c
ARITH
then pass "chat completions does arithmetic"; else fail "chat arithmetic: $(head -c 200 <<<\"$body\")"; fi

stream=$(curl -sN -m 300 "http://$HOST:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":24,"temperature":0,"stream":true}' || true)
printf '%s' "$stream" > /tmp/lse-smoke-stream.txt
if python3 - /tmp/lse-smoke-stream.txt <<'PY'
import json, sys
frames = [l[6:] for l in open(sys.argv[1]).read().splitlines() if l.startswith("data: ")]
assert frames, "no SSE frames"
assert frames[-1].strip() == "[DONE]", frames[-1]
chunks = [json.loads(f) for f in frames[:-1]]
assert chunks[0]["choices"][0]["delta"].get("role") == "assistant", "no opening role chunk"
assert any(c["choices"][0]["delta"].get("content") for c in chunks), "no content delta"
assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length"), "no finish_reason"
assert "usage" in chunks[-1], "final chunk carries no usage"
PY
then pass "POST /v1/chat/completions (stream)"; else fail "POST /v1/chat/completions (stream)"; fi

body=$(curl -sf -m 300 "http://$HOST:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":8,"temperature":0}' || true)
if python3 - "$body" <<'PY'
import json, sys
d = json.loads(sys.argv[1])
assert d["object"] == "text_completion", d["object"]
assert "paris" in d["choices"][0]["text"].lower(), d["choices"][0]["text"]
PY
then pass "POST /v1/completions"; else fail "POST /v1/completions: $(head -c 200 <<<"$body")"; fi

# A bad request must be refused in the format a client parses, not merely
# refused: an OpenAI client reads error.message and would show nothing.
code=$(curl -s -o /tmp/lse-smoke-err.json -w '%{http_code}' -m 30 \
  "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"n":2}')
if [ "$code" = "400" ] && python3 -c "import json;d=json.load(open('/tmp/lse-smoke-err.json'));assert d['error']['message']"; then
  pass "a refused request wears the error envelope"
else
  fail "expected 400 with an error envelope, got $code"
fi

code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 "http://$HOST:$PORT/v1/embeddings" -d '{}')
[ "$code" = "501" ] && pass "an unimplemented route says so" || fail "expected 501 from /v1/embeddings, got $code"

echo
if [ "$fails" -ne 0 ]; then
  echo "$fails check(s) failed"
  exit 1
fi
echo "all checks passed"
