#!/bin/bash

PROBLEM_DIR="/home/user/RLSolver_Contest_2025/Task2/Baseline/_graphs/benchmarks/EA_20x20"
CONFIG_FILE="mcpg_config.yaml"
PYTHON_SCRIPT="ising_mcpg_single_file.py"
LOG_DIR="run_logs"

if [ ! -d "$PROBLEM_DIR" ]; then
    echo "Error: Problem directory '${PROBLEM_DIR}' not found."
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "Starting automated runs for all .txt files in '${PROBLEM_DIR}'..."

for problem_file in "$PROBLEM_DIR"/*.txt; do
    if [ -f "$problem_file" ]; then
        base_name=$(basename "$problem_file" .txt)
        output_file="${LOG_DIR}/${base_name}.log"

        echo "==> Processing: ${base_name}.txt"
        python "${PYTHON_SCRIPT}" "${CONFIG_FILE}" "${problem_file}" > "${output_file}"
        echo "==> Finished. Output saved to ${output_file}"
        echo "--------------------------------------------------"
    else
        echo "Warning: No .txt files found in '${PROBLEM_DIR}'. Exiting."
        break
    fi
done

echo "All runs completed!"
