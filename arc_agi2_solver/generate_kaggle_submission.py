#!/usr/bin/env python3
"""
Generate Kaggle submission CSV for ARC Prize 2025.

Loads the official test set and runs the ARC-AGI-2 solver on each test input.
Outputs a CSV file in the required competition format.
"""

import json
import os
import csv
from tqdm import tqdm
from arc_solver import ARCSolver, Grid, Task

# Path to the test set
TEST_JSON = "arc-prize-2025-data/arc-agi_test_challenges.json"
SUBMISSION_CSV = "submission_kaggle.csv"


def main():
    # Load test set
    with open(TEST_JSON, "r") as f:
        test_data = json.load(f)

    solver = ARCSolver()
    results = []

    print(f"Loaded {len(test_data)} test tasks. Running solver on all tasks...")

    for task_id, task_obj in tqdm(test_data.items()):
        train_pairs = []
        for pair in task_obj["train"]:
            train_pairs.append((Grid(pair["input"]), Grid(pair["output"])))
        test_pairs = [(Grid(pair["input"]), Grid([[0]])) for pair in task_obj["test"]]  # Dummy output for type
        task = Task(task_id, train_pairs, test_pairs)
        try:
            outputs = solver.solve_task(task)
        except Exception as e:
            print(f"Error solving {task_id}: {e}")
            outputs = [Grid([[0]]) for _ in test_pairs]
        for i, output in enumerate(outputs):
            results.append({
                "task_id": task_id,
                "test_id": i,
                "output": json.dumps(output.to_list())
            })

    # Write CSV
    with open(SUBMISSION_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["task_id", "test_id", "output"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Submission CSV written to {SUBMISSION_CSV} with {len(results)} rows.")


if __name__ == "__main__":
    main() 