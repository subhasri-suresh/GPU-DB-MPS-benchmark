#!/bin/bash

# ── 1. Build (make checks timestamps and recompiles only if needed) ───────────
make

# ── 2. Run binary ─────────────────────────────────────────────────────────────
./MetalVectorSum
