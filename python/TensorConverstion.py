import pandas as pd
import torch

df = pd.read_csv(
    "tpch-dbgen/orders.tbl",
    sep="|",
    header=None,
    names=["o_orderkey","o_custkey","o_orderstatus","o_totalprice",
           "o_orderdate","o_orderpriority","o_clerk","o_shippriority","o_comment"],
    usecols=range(9)
)

# Keep only numeric columns
numeric_cols = ["o_orderkey", "o_custkey", "o_totalprice", "o_shippriority"]
df_numeric = df[numeric_cols].astype("float32")

# Convert to tensor
tensor = torch.tensor(df_numeric.values, dtype=torch.float32)

print("Shape:", tensor.shape)
print("Sample:", tensor[:5])

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tensor = tensor.to(device)
print("Device:", tensor.device)


col = {name: i for i, name in enumerate(numeric_cols)}

total = tensor[:, col["o_totalprice"]].sum()
print("Total price sum:", total.item())
