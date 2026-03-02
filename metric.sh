#!/bin/bash

OUTPUT_FILE="power_log.txt"

echo "Timestamp, CPU Power (mW), GPU Power (mW), ANE Power (mW), Combined Power (mW)" > "$OUTPUT_FILE"

sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 | while IFS= read -r line; do
    if [[ "$line" =~ ^CPU\ Power:\ ([0-9]+)\ mW ]]; then
        cpu="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^GPU\ Power:\ ([0-9]+)\ mW ]]; then
        gpu="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^ANE\ Power:\ ([0-9]+)\ mW ]]; then
        ane="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^Combined\ Power.*:\ ([0-9]+)\ mW ]]; then
        combined="${BASH_REMATCH[1]}"
        timestamp=$(date "+%Y-%m-%d %H:%M:%S")
        echo "$timestamp, $cpu, $gpu, $ane, $combined" | tee -a "$OUTPUT_FILE"
    fi
done
