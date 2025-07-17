#!/usr/bin/env python3
"""
Test script for submission format.

Tests the submission system with a sample task to ensure
the output format is correct for Kaggle competition.
"""

import json
import os
from submission import SubmissionFormatter, validate_submission_format


def test_submission_format():
    """Test the submission format with sample task."""
    print("=== Testing Submission Format ===")
    
    # Create formatter
    formatter = SubmissionFormatter()
    
    # Test with sample task
    sample_task_path = "sample_task.json"
    
    if not os.path.exists(sample_task_path):
        print(f"Sample task file not found: {sample_task_path}")
        return False
    
    try:
        # Solve and format task
        print("Solving sample task...")
        result = formatter.solve_and_format_task(sample_task_path)
        
        print(f"Task ID: {result['task_id']}")
        print(f"Number of results: {len(result['results'])}")
        
        # Validate format
        print("\nValidating submission format...")
        is_valid = validate_submission_format(result['results'])
        
        if is_valid:
            print("✅ Submission format is valid!")
            
            # Show sample output
            print("\nSample output:")
            for i, res in enumerate(result['results']):
                print(f"  Test {res['test_id']}:")
                print(f"    Task ID: {res['task_id']}")
                print(f"    Output shape: {len(res['output'])}x{len(res['output'][0])}")
                print(f"    Output preview: {res['output'][:2]}...")  # Show first 2 rows
        else:
            print("❌ Submission format validation failed!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing submission: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_output():
    """Test CSV output format."""
    print("\n=== Testing CSV Output ===")
    
    formatter = SubmissionFormatter()
    sample_task_path = "sample_task.json"
    
    if not os.path.exists(sample_task_path):
        print(f"Sample task file not found: {sample_task_path}")
        return False
    
    try:
        # Create submission
        result = formatter.solve_and_format_task(sample_task_path)
        results = result['results']
        
        # Save CSV
        csv_path = "test_submission.csv"
        formatter.save_submission_csv(results, csv_path)
        
        print(f"✅ CSV saved to: {csv_path}")
        
        # Verify CSV content
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            print(f"CSV has {len(lines)} lines (including header)")
            if len(lines) > 1:
                print(f"First data line: {lines[1].strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing CSV output: {e}")
        return False


def test_json_output():
    """Test JSON output format."""
    print("\n=== Testing JSON Output ===")
    
    formatter = SubmissionFormatter()
    sample_task_path = "sample_task.json"
    
    if not os.path.exists(sample_task_path):
        print(f"Sample task file not found: {sample_task_path}")
        return False
    
    try:
        # Create submission
        result = formatter.solve_and_format_task(sample_task_path)
        results = result['results']
        
        # Save JSON
        json_path = "test_submission.json"
        formatter.save_submission_json(results, json_path)
        
        print(f"✅ JSON saved to: {json_path}")
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            data = json.load(f)
            print(f"JSON structure: {list(data.keys())}")
            print(f"Number of results: {len(data['results'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing JSON output: {e}")
        return False


def main():
    """Run all submission tests."""
    print("ARC Prize 2025 Submission Format Test")
    print("=" * 50)
    
    tests = [
        test_submission_format,
        test_csv_output,
        test_json_output
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error: {test.__name__} - {e}")
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Submission format is ready for Kaggle.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main() 