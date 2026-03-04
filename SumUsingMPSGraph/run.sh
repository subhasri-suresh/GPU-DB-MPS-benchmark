#!/bin/bash

# ── 1. Build (make checks timestamps and recompiles only if needed) ───────────
make

# ── 2. Start metrics collection in background ─────────────────────────────────
bash metric.sh &
METRIC_PID=$!

# ── 3. Run binary ─────────────────────────────────────────────────────────────
./MPSSumUsingGraph

# ── 4. Stop metrics collection ────────────────────────────────────────────────
kill $METRIC_PID 2>/dev/null
wait $METRIC_PID 2>/dev/null
