#!/usr/bin/env python3
"""
Submission module for ARC Prize 2025 Kaggle competition.

Formats solver output to match the expected submission format.
"""

import json
import csv
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from arc_solver import ARCSolver, Grid, Task
from arc_solver.parser import load_task_from_json, load_puzzle_from_directory


class SubmissionFormatter:
    """Formats solver output for Kaggle competition submission."""
    
    def __init__(self):
        self.solver = ARCSolver()
    
    def solve_and_format_task(self, task_path: str) -> Dict[str, Any]:
        """
        Solve a task and format the output for submission.
        
        Args:
            task_path: Path to task JSON file
            
        Returns:
            Dictionary with formatted submission data
        """
        # Load task
        task = load_task_from_json(task_path)
        task_id = task.task_id
        
        # Solve task
        results = self.solver.solve_task(task)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                'task_id': task_id,
                'test_id': i,
                'output': result.to_list()
            })
        
        return {
            'task_id': task_id,
            'results': formatted_results
        }
    
    def solve_multiple_tasks(self, task_directory: str) -> List[Dict[str, Any]]:
        """
        Solve multiple tasks and format for submission.
        
        Args:
            task_directory: Directory containing task JSON files
            
        Returns:
            List of formatted submission data
        """
        puzzle = load_puzzle_from_directory(task_directory)
        all_results = []
        
        for task in puzzle.tasks:
            try:
                results = self.solver.solve_task(task)
                
                for i, result in enumerate(results):
                    all_results.append({
                        'task_id': task.task_id,
                        'test_id': i,
                        'output': result.to_list()
                    })
                    
            except Exception as e:
                print(f"Error solving task {task.task_id}: {e}")
                # Add empty result for failed tasks
                for i in range(len(task.test_pairs)):
                    all_results.append({
                        'task_id': task.task_id,
                        'test_id': i,
                        'output': [[0]]  # Default empty output
                    })
        
        return all_results
    
    def save_submission_csv(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save results in CSV format for Kaggle submission.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save CSV file
        """
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['task_id', 'test_id', 'output']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'task_id': result['task_id'],
                    'test_id': result['test_id'],
                    'output': json.dumps(result['output'])
                })
    
    def save_submission_json(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save results in JSON format for submission.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save JSON file
        """
        submission_data = {
            'submission_format': 'arc_prize_2025',
            'version': '1.0',
            'results': results
        }
        
        with open(output_path, 'w') as jsonfile:
            json.dump(submission_data, jsonfile, indent=2)
    
    def create_sample_submission(self, task_directory: str, output_dir: str = "submission"):
        """
        Create a complete submission from task directory.
        
        Args:
            task_directory: Directory containing task JSON files
            output_dir: Directory to save submission files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Solve all tasks
        print("Solving tasks...")
        results = self.solve_multiple_tasks(task_directory)
        
        # Save in different formats
        csv_path = os.path.join(output_dir, "submission.csv")
        json_path = os.path.join(output_dir, "submission.json")
        
        self.save_submission_csv(results, csv_path)
        self.save_submission_json(results, json_path)
        
        print(f"Submission saved to {output_dir}/")
        print(f"  - CSV: {csv_path}")
        print(f"  - JSON: {json_path}")
        print(f"  - Total results: {len(results)}")
        
        return results


def create_submission_from_directory(task_dir: str, output_dir: str = "submission"):
    """
    Create a submission from a directory of task files.
    
    Args:
        task_dir: Directory containing task JSON files
        output_dir: Directory to save submission files
    """
    formatter = SubmissionFormatter()
    return formatter.create_sample_submission(task_dir, output_dir)


def create_submission_from_single_task(task_path: str, output_dir: str = "submission"):
    """
    Create a submission from a single task file.
    
    Args:
        task_path: Path to single task JSON file
        output_dir: Directory to save submission files
    """
    formatter = SubmissionFormatter()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Solve task
    result = formatter.solve_and_format_task(task_path)
    results = result['results']
    
    # Save in different formats
    csv_path = os.path.join(output_dir, "submission.csv")
    json_path = os.path.join(output_dir, "submission.json")
    
    formatter.save_submission_csv(results, csv_path)
    formatter.save_submission_json(results, json_path)
    
    print(f"Submission saved to {output_dir}/")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Total results: {len(results)}")
    
    return results


def validate_submission_format(results: List[Dict[str, Any]]) -> bool:
    """
    Validate that submission format is correct.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        True if format is valid
    """
    try:
        for result in results:
            # Check required fields
            required_fields = ['task_id', 'test_id', 'output']
            for field in required_fields:
                if field not in result:
                    print(f"Missing required field: {field}")
                    return False
            
            # Check output format
            output = result['output']
            if not isinstance(output, list):
                print(f"Output must be a list, got {type(output)}")
                return False
            
            if not all(isinstance(row, list) for row in output):
                print("Output must be a 2D list")
                return False
            
            # Check that all values are integers 0-9
            for row in output:
                for val in row:
                    if not isinstance(val, int) or val < 0 or val > 9:
                        print(f"Invalid color value: {val}")
                        return False
        
        print("Submission format validation passed!")
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def main():
    """Main function for creating submissions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create ARC Prize 2025 submission')
    parser.add_argument('--task_dir', type=str, help='Directory containing task JSON files')
    parser.add_argument('--task_file', type=str, help='Single task JSON file')
    parser.add_argument('--output_dir', type=str, default='submission', help='Output directory')
    
    args = parser.parse_args()
    
    if args.task_dir:
        print(f"Creating submission from directory: {args.task_dir}")
        results = create_submission_from_directory(args.task_dir, args.output_dir)
    elif args.task_file:
        print(f"Creating submission from file: {args.task_file}")
        results = create_submission_from_single_task(args.task_file, args.output_dir)
    else:
        print("Please provide either --task_dir or --task_file")
        return
    
    # Validate submission
    print("\nValidating submission format...")
    validate_submission_format(results)


if __name__ == "__main__":
    main() 