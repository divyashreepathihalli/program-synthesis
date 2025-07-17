#!/usr/bin/env python3
"""
Example submission script for ARC Prize 2025 Kaggle competition.

This script demonstrates the complete workflow from loading tasks
to creating a properly formatted submission file.
"""

import os
import json
from submission import SubmissionFormatter, validate_submission_format


def create_sample_tasks():
    """Create sample task files for demonstration."""
    
    # Task 1: Simple color swap (1 -> 2)
    task1 = {
        "train": [
            {
                "input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                "output": [[0, 2, 0], [2, 2, 2], [0, 2, 0]]
            }
        ],
        "test": [
            {
                "input": [[1, 1], [1, 1]],
                "output": [[2, 2], [2, 2]]
            }
        ]
    }
    
    # Task 2: Translation (move right by 1)
    task2 = {
        "train": [
            {
                "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
            }
        ],
        "test": [
            {
                "input": [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                "output": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
            }
        ]
    }
    
    # Create tasks directory
    os.makedirs("sample_tasks", exist_ok=True)
    
    # Save task files
    with open("sample_tasks/task_001.json", "w") as f:
        json.dump(task1, f, indent=2)
    
    with open("sample_tasks/task_002.json", "w") as f:
        json.dump(task2, f, indent=2)
    
    print("✅ Created sample tasks in sample_tasks/")
    return "sample_tasks"


def demonstrate_submission_workflow():
    """Demonstrate the complete submission workflow."""
    
    print("=== ARC Prize 2025 Submission Workflow ===")
    print()
    
    # Step 1: Create sample tasks
    print("Step 1: Creating sample tasks...")
    task_dir = create_sample_tasks()
    
    # Step 2: Initialize solver
    print("\nStep 2: Initializing solver...")
    formatter = SubmissionFormatter()
    
    # Step 3: Solve tasks
    print("\nStep 3: Solving tasks...")
    try:
        results = formatter.solve_multiple_tasks(task_dir)
        print(f"✅ Solved {len(results)} test cases")
    except Exception as e:
        print(f"❌ Error solving tasks: {e}")
        print("Creating dummy results for demonstration...")
        results = [
            {
                'task_id': 'task_001',
                'test_id': 0,
                'output': [[2, 2], [2, 2]]
            },
            {
                'task_id': 'task_002', 
                'test_id': 0,
                'output': [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
            }
        ]
    
    # Step 4: Validate format
    print("\nStep 4: Validating submission format...")
    is_valid = validate_submission_format(results)
    if is_valid:
        print("✅ Submission format is valid!")
    else:
        print("❌ Submission format validation failed!")
        return
    
    # Step 5: Save submission files
    print("\nStep 5: Saving submission files...")
    os.makedirs("submission", exist_ok=True)
    
    # Save CSV
    csv_path = "submission/submission.csv"
    formatter.save_submission_csv(results, csv_path)
    print(f"✅ CSV saved: {csv_path}")
    
    # Save JSON
    json_path = "submission/submission.json"
    formatter.save_submission_json(results, json_path)
    print(f"✅ JSON saved: {json_path}")
    
    # Step 6: Show results
    print("\nStep 6: Submission summary:")
    print(f"  - Total results: {len(results)}")
    print(f"  - Tasks: {len(set(r['task_id'] for r in results))}")
    print(f"  - Test cases: {len(set((r['task_id'], r['test_id']) for r in results))}")
    
    print("\nSample results:")
    for i, result in enumerate(results[:3]):  # Show first 3
        print(f"  {i+1}. Task {result['task_id']}, Test {result['test_id']}")
        print(f"     Output: {result['output']}")
    
    print("\n🎉 Submission workflow completed successfully!")
    print("📁 Check the 'submission/' directory for your files.")
    print("📤 Ready to upload to Kaggle!")


def show_submission_requirements():
    """Show the submission requirements."""
    
    print("\n=== Submission Requirements ===")
    print()
    print("📋 Required Format:")
    print("  - CSV file with columns: task_id, test_id, output")
    print("  - Output must be 2D arrays of integers (0-9)")
    print("  - All test cases must be included")
    print()
    print("🎨 Color Mapping:")
    print("  0: Black (background)")
    print("  1: Blue")
    print("  2: Red")
    print("  3: Green")
    print("  4: Yellow")
    print("  5: Gray")
    print("  6: Orange")
    print("  7: Purple")
    print("  8: Brown")
    print("  9: Pink")
    print()
    print("📊 Example CSV:")
    print("  task_id,test_id,output")
    print('  task_001,0,"[[0,0,0],[0,1,0],[0,0,0]]"')
    print('  task_001,1,"[[0,0,0],[0,2,0],[0,0,0]]"')


def main():
    """Main function."""
    
    print("ARC Prize 2025 - Submission Example")
    print("=" * 50)
    
    # Show requirements
    show_submission_requirements()
    
    # Run workflow
    demonstrate_submission_workflow()
    
    print("\n" + "=" * 50)
    print("📚 For detailed instructions, see SUBMISSION_GUIDE.md")
    print("🧪 For testing, run: python test_submission.py")
    print("🚀 For Kaggle upload, use the files in submission/")


if __name__ == "__main__":
    main() 