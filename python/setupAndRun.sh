#!/bin/bash

# ── 1. Clone and build TPC-H dbgen ──────────────────────────────────────────
if [ ! -d "../common/tpch-dbgen" ]; then
    git clone https://github.com/electrum/tpch-dbgen
    cd tpch-dbgen

    # Fix the warning by adding -Wno-string-plus-int
    sed -i '' 's/CFLAGS =/CFLAGS = -Wno-string-plus-int/' makefile

    make

    # Generate TPC-H data at scale factor 1
    ./dbgen -s 1
    cd ..
else
    echo "tpch-dbgen already exists, skipping clone and build..."
fi

# ── 2. Install Python dependencies ──────────────────────────────────────────
pip install torch pandas

# ── 3. Run the Python script ─────────────────────────────────────────────────
python3 TensorConverstion.py
