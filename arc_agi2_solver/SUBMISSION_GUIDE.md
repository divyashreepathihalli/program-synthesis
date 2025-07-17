# ARC Prize 2025 Submission Guide

This guide explains how to create submissions for the ARC Prize 2025 Kaggle competition using our ARC-AGI-2 solver.

## Submission Format

The competition expects submissions in the following format:

### CSV Format (Primary)
```csv
task_id,test_id,output
task_001,0,"[[0,0,0],[0,1,0],[0,0,0]]"
task_001,1,"[[0,0,0],[0,2,0],[0,0,0]]"
task_002,0,"[[1,1],[1,1]]"
```

### JSON Format (Alternative)
```json
{
  "submission_format": "arc_prize_2025",
  "version": "1.0",
  "results": [
    {
      "task_id": "task_001",
      "test_id": 0,
      "output": [[0,0,0],[0,1,0],[0,0,0]]
    },
    {
      "task_id": "task_001", 
      "test_id": 1,
      "output": [[0,0,0],[0,2,0],[0,0,0]]
    }
  ]
}
```

## Requirements

- **task_id**: Unique identifier for each task
- **test_id**: Index of the test case (0-based)
- **output**: 2D array of integers (0-9) representing the predicted output grid

## Color Mapping

The solver uses the following color mapping:
- 0: Black (background)
- 1: Blue
- 2: Red  
- 3: Green
- 4: Yellow
- 5: Gray
- 6: Orange
- 7: Purple
- 8: Brown
- 9: Pink

## Usage

### 1. Single Task Submission

```python
from submission import create_submission_from_single_task

# Create submission from a single task file
results = create_submission_from_single_task(
    task_path="path/to/task.json",
    output_dir="submission"
)
```

### 2. Multiple Tasks Submission

```python
from submission import create_submission_from_directory

# Create submission from a directory of task files
results = create_submission_from_directory(
    task_dir="path/to/tasks/",
    output_dir="submission"
)
```

### 3. Command Line Usage

```bash
# Single task
python submission.py --task_file sample_task.json --output_dir submission

# Multiple tasks
python submission.py --task_dir tasks/ --output_dir submission
```

## File Structure

```
arc_agi2_solver/
├── submission.py              # Main submission module
├── test_submission.py         # Test script
├── sample_task.json           # Sample task for testing
├── SUBMISSION_GUIDE.md        # This guide
└── submission/                # Output directory
    ├── submission.csv         # CSV submission file
    └── submission.json        # JSON submission file
```

## Testing Your Submission

Run the test script to validate your submission format:

```bash
python test_submission.py
```

This will:
1. Test with the sample task
2. Validate the output format
3. Generate CSV and JSON files
4. Verify all requirements are met

## Validation Rules

The submission validator checks:

1. **Required Fields**: All results must have `task_id`, `test_id`, and `output`
2. **Output Format**: Output must be a 2D list of integers
3. **Color Values**: All values must be integers 0-9
4. **Consistency**: All test cases for a task must be included

## Error Handling

The submission system includes robust error handling:

- **Failed Tasks**: If a task fails to solve, it returns a default output `[[0]]`
- **Missing Files**: Graceful handling of missing task files
- **Format Errors**: Detailed error messages for invalid formats

## Competition Integration

### Kaggle Notebook

```python
# In your Kaggle notebook
import sys
sys.path.append('/kaggle/input/arc-agi2-solver/arc_agi2_solver')

from submission import SubmissionFormatter

# Initialize solver
formatter = SubmissionFormatter()

# Solve all tasks in the competition dataset
results = formatter.solve_multiple_tasks('/kaggle/input/arc-prize-2025/tasks/')

# Save submission
formatter.save_submission_csv(results, 'submission.csv')
```

### Local Development

```python
# Test locally before submitting
from submission import create_submission_from_directory

# Create submission
results = create_submission_from_directory('local_tasks/', 'submission/')

# Validate format
from submission import validate_submission_format
is_valid = validate_submission_format(results)
print(f"Submission valid: {is_valid}")
```

## Best Practices

1. **Test Locally**: Always test your submission format before uploading
2. **Validate Output**: Use the validation function to check your results
3. **Handle Errors**: The system gracefully handles failed tasks
4. **Check Format**: Ensure your CSV/JSON matches the expected format
5. **Version Control**: Keep track of different solver versions

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **Format Errors**: Check that output arrays are 2D lists of integers
3. **Missing Tasks**: Ensure all test cases are included
4. **Color Values**: Verify all values are 0-9

### Debug Mode

Enable debug output by modifying the solver:

```python
# In submission.py, add debug=True
formatter = SubmissionFormatter(debug=True)
```

## Performance Tips

1. **Batch Processing**: Process multiple tasks together for efficiency
2. **Memory Management**: Clear intermediate results for large datasets
3. **Parallel Processing**: Use multiprocessing for large task sets
4. **Caching**: Cache solved tasks to avoid recomputation

## Submission Checklist

Before submitting to Kaggle:

- [ ] All required fields present (`task_id`, `test_id`, `output`)
- [ ] Output format is 2D list of integers 0-9
- [ ] All test cases included
- [ ] CSV file properly formatted
- [ ] File size within limits
- [ ] No syntax errors in output
- [ ] Validation script passes

## Support

For issues with the submission format:

1. Check the validation output
2. Review the sample task format
3. Test with the provided test script
4. Verify against the competition requirements

## Example Output

```
=== Testing Submission Format ===
Solving sample task...
Task ID: sample_task
Number of results: 1

Validating submission format...
✅ Submission format is valid!

Sample output:
  Test 0:
    Task ID: sample_task
    Output shape: 5x5
    Output preview: [[0, 0, 0, 0, 0], [0, 2, 0, 2, 0]]...

✅ CSV saved to: test_submission.csv
✅ JSON saved to: test_submission.json

🎉 All tests passed! Submission format is ready for Kaggle.
``` 