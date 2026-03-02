
## Projects

| Directory | Description |
|---|---|
| `SumUsingGraph/` | Large vector sum via MPSGraph |
| `SumUsingMPSMatrix/` | Large vector sum via MPSMatrix |
| `NeuralNetworkWithMPS/` | Neural network inference with MPS |
| `python/` | Python baseline / tensor conversion utilities |

## Setup & Run

```bash
./setupAndRun.sh
```

Each project also has its own `run.sh` and `Makefile`.

## Output Files

- **`output.txt`** — Computation results: device info, row count, CPU sum, GPU sum, and relative error between CPU and GPU results.
- **`power_log.txt`** — Per-second power samples (CPU, GPU, ANE, combined in mW) captured via `powermetrics` during execution.

## Measuring Power

```bash
sudo powermetrics --samplers cpu_power,gpu_power -i 1000
```

Use `metric.sh` inside each project directory to automate power logging alongside a run.
