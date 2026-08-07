import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Load data ─────────────────────────────────────────────────────────────────
metal_csv = "results/metal_results.csv"
mps_csv   = "results/mps_results.csv"

metal_df = pd.read_csv(metal_csv)
mps_df   = pd.read_csv(mps_csv)

# ── Filter: latest run on Apple M1, only queries present in both ──────────────
QUERIES = ["Q1", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"]
SFS     = ["SF-1", "SF-10"]

metal_m1 = metal_df[metal_df["gpu_name"] == "Apple M1"]
metal_latest = (
    metal_m1.sort_values("timestamp")
             .groupby(["scale_factor", "query"], as_index=False)
             .last()
)
metal_latest = metal_latest[
    metal_latest["query"].isin(QUERIES) &
    metal_latest["scale_factor"].isin(SFS)
][["scale_factor", "query", "gpu_exec_ms"]].rename(columns={"gpu_exec_ms": "metal_ms"})

mps_latest = (
    mps_df.sort_values("timestamp")
          .groupby(["scale_factor", "query"], as_index=False)
          .last()
)
mps_latest = mps_latest[
    mps_latest["query"].isin(QUERIES) &
    mps_latest["scale_factor"].isin(SFS)
][["scale_factor", "query", "gpu_exec_ms"]].rename(columns={"gpu_exec_ms": "mps_ms"})

df = pd.merge(metal_latest, mps_latest, on=["scale_factor", "query"])
df["speedup"] = df["mps_ms"] / df["metal_ms"]
df = df.sort_values(["scale_factor", "query"])

# ── Print and save summary table ─────────────────────────────────────────────
header = f"{'SF':<8} {'Query':<6} {'Metal (ms)':>12} {'MPS (ms)':>12} {'Speedup':>10}"
divider = "-" * 52
rows = [f"{r.scale_factor:<8} {r.query:<6} {r.metal_ms:>12.2f} {r.mps_ms:>12.2f} {r.speedup:>9.1f}x"
        for _, r in df.iterrows()]

table_txt = "\n".join([header, divider] + rows)
print(table_txt)

with open("results/comparison_table.txt", "w") as f:
    f.write("Plain Metal vs MPSGraph — TPC-H on Apple M1\n")
    f.write("=" * 52 + "\n")
    f.write(table_txt + "\n")

df.rename(columns={"metal_ms": "metal_gpu_ms", "mps_ms": "mps_gpu_ms"}) \
  .to_csv("results/comparison_table.csv", index=False)

print("\nTable saved → results/comparison_table.txt")
print("Table saved → results/comparison_table.csv")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Plain Metal vs MPSGraph — TPC-H on Apple M1", fontsize=14, fontweight="bold")

colors = {"Metal": "#2196F3", "MPS": "#FF5722"}
x = np.arange(len(QUERIES))
width = 0.35

for ax, sf in zip(axes, SFS):
    sub = df[df["scale_factor"] == sf].set_index("query").reindex(QUERIES)
    bars_m = ax.bar(x - width/2, sub["metal_ms"], width, label="Plain Metal", color=colors["Metal"], zorder=3)
    bars_p = ax.bar(x + width/2, sub["mps_ms"],   width, label="MPSGraph",    color=colors["MPS"],   zorder=3)

    # Speedup labels above MPS bars
    for bar, sp in zip(bars_p, sub["speedup"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bar.get_height() * 0.02,
                f"{sp:.1f}×", ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_title(sf, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(QUERIES)
    ax.set_ylabel("GPU Execution Time (ms)")
    ax.set_xlabel("TPC-H Query")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f ms"))
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

plt.tight_layout()
out = "results/comparison_metal_vs_mps.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nChart saved → {out}")
plt.show()
