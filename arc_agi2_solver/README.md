# ARC-AGI-2 Solver

A comprehensive, modular solver for the ARC-AGI-2 benchmark that uses symbolic reasoning and compositional transformations to solve grid-based puzzles.

## Overview

The ARC-AGI-2 benchmark consists of grid-based puzzles where each task provides a few example input-output grid pairs and one or more test inputs (grids up to 30×30, using 10 colors). A solver must infer the hidden rule from the demos and apply it to the test inputs.

This implementation provides a modular, extensible architecture designed to tackle these challenging visual reasoning tasks.

## Architecture

### Core Components

- **`arc_solver/parser.py`** - Grid, Task, and Puzzle classes for loading ARC-AGI-2 tasks
- **`arc_solver/grid.py`** - Utility functions for grid operations (rotate, reflect, flood-fill, etc.)
- **`arc_solver/detectors/`** - Modules to detect primitive elements:
  - `color_detector.py` - Analyze color patterns and transformations
  - `shape_detector.py` - Identify objects via connected-component labeling
  - `pattern_detector.py` - Detect spatial patterns (lines, symmetry, motifs)
- **`arc_solver/transformations/`** - Elementary transformations:
  - `transform.py` - Base Transformation class and implementations (Translate, Rotate, Recolor, etc.)
- **`arc_solver/symbolic/`** - Symbolic reasoning engine:
  - `rule.py` - Symbolic rule representation
  - `inference.py` - Logic to match rules against example pairs
- **`arc_solver/pipeline.py`** - Main orchestrator that ties everything together

## Features

### Object Detection
- Connected component analysis for object identification
- Shape property analysis (area, perimeter, circularity, symmetry)
- Bounding box and centroid calculation
- Color-based object segmentation

### Pattern Recognition
- **Color patterns**: Color swaps, additions, removals, frequency analysis
- **Geometric patterns**: Translation, scaling, rotation detection
- **Spatial patterns**: Lines, symmetry axes, repeated motifs, alignment
- **Structural patterns**: Grid patterns, relative positioning

### Rule Inference
- Multi-modal transformation detection with confidence scoring
- Symbolic rule representation and composition
- Rule testing and validation against training pairs
- Intelligent rule selection and ranking

### Transformation Pipeline
- Elementary transformations (Translate, Rotate, Reflect, Recolor, Scale)
- Composite transformations (Pipeline, Conditional, Composite)
- Rule-based transformation application
- Extensible transformation framework

## Installation

```bash
cd arc_agi2_solver
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from arc_solver import ARCSolver, Grid, Task

# Create solver
solver = ARCSolver()

# Create a simple task
input_grid = Grid([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
output_grid = Grid([[0, 2, 0], [2, 2, 2], [0, 2, 0]])
train_pairs = [(input_grid, output_grid)]

# Create task
task = Task("simple_color_swap", train_pairs, [])

# Solve the task
results = solver.solve_task(task)
```

### Loading from JSON

```python
from arc_solver import solve_task_from_json

# Solve a task from JSON file
results = solve_task_from_json("path/to/task.json")
```

### Using Individual Components

```python
from arc_solver.detectors.color_detector import ColorDetector
from arc_solver.detectors.shape_detector import ShapeDetector
from arc_solver.transformations.transform import Recolor, Translate

# Color analysis
color_detector = ColorDetector()
color_analysis = color_detector.analyze_color_patterns(train_pairs)

# Shape analysis
shape_detector = ShapeDetector()
objects = shape_detector.find_all_objects(grid)

# Apply transformations
recolor = Recolor({1: 2})
translated = Translate(dx=1, dy=1)
```

## Module Structure

### Detectors

#### ColorDetector
- `analyze_color_frequency()` - Analyze color distribution
- `detect_color_swaps()` - Find consistent color mappings
- `detect_color_preservation()` - Find preserved colors
- `suggest_color_transformations()` - Generate color-based rules

#### ShapeDetector
- `find_all_objects()` - Identify all objects in grid
- `analyze_object_properties()` - Detailed shape analysis
- `detect_shape_changes()` - Compare input/output objects
- `detect_object_relationships()` - Analyze spatial relationships

#### PatternDetector
- `detect_lines()` - Find horizontal, vertical, diagonal lines
- `detect_symmetry()` - Analyze symmetry patterns
- `detect_repeated_motifs()` - Find repeating patterns
- `detect_alignment_patterns()` - Check object alignment

### Transformations

#### Elementary Transformations
- `Translate(dx, dy)` - Move grid by offset
- `Rotate(k)` - Rotate by k*90 degrees
- `Reflect(axis)` - Mirror along axis
- `Recolor(mapping)` - Change colors
- `Scale(factor)` - Resize grid
- `Crop(bounds)` - Extract region
- `Pad(size)` - Add padding

#### Composite Transformations
- `Pipeline(transformations)` - Apply sequence
- `ConditionalTransform(condition, transform)` - Conditional application
- `CompositeTransform(transforms, combine_func)` - Combine results

### Symbolic Reasoning

#### Rule System
- `TransformationRule` - Rule based on single transformation
- `ConditionalRule` - Rule with conditions
- `ObjectMappingRule` - Rule mapping objects
- `PatternRule` - Rule based on pattern matching
- `CompositeRule` - Rule combining sub-rules
- `LogicRule` - Rule with logical conditions

#### Inference Engine
- `InferenceEngine` - Generate rules from training data
- `RuleMatcher` - Match rules against examples
- Rule testing and validation
- Confidence scoring and ranking

## Testing

Run the test suite:

```bash
cd arc_agi2_solver
python -m pytest tests/
```

## Evaluation

The solver can be evaluated on ARC-AGI-2 tasks:

```python
from arc_solver import solve_multiple_tasks

# Solve multiple tasks
task_paths = ["task1.json", "task2.json", "task3.json"]
results = solve_multiple_tasks(task_paths)

# Check accuracy
for task_id, outputs in results.items():
    print(f"Task {task_id}: {len(outputs)} outputs")
```

## Extensibility

The modular design allows easy extension:

### Adding New Detectors
```python
class NewDetector:
    def analyze(self, grid):
        # Your analysis logic
        pass
    
    def suggest_transformations(self, analysis):
        # Your transformation suggestions
        pass
```

### Adding New Transformations
```python
class CustomTransform(Transformation):
    def apply(self, grid):
        # Your transformation logic
        return transformed_grid
```

### Adding New Rule Types
```python
class CustomRule(Rule):
    def apply(self, grid):
        # Your rule application
        pass
    
    def matches(self, input_grid, output_grid):
        # Your matching logic
        pass
```

## Performance

The solver is designed for:
- **Modularity**: Each component can be tested and optimized independently
- **Extensibility**: New detectors, transformations, and rules can be added easily
- **Composability**: Complex rules can be built from simple components
- **Symbolic reasoning**: Rules are represented as symbolic objects for better interpretability

## Kaggle Competition Submission

This solver is designed for the [ARC Prize 2025 Kaggle competition](https://www.kaggle.com/competitions/arc-prize-2025/overview).

### Submission Format

The solver outputs results in the required Kaggle format:

```python
from submission import create_submission_from_directory

# Create submission from task directory
results = create_submission_from_directory(
    task_dir="path/to/tasks/",
    output_dir="submission"
)
```

This generates:
- `submission/submission.csv` - Primary submission file
- `submission/submission.json` - Alternative format

### Submission Requirements

- **CSV Format**: `task_id,test_id,output`
- **Output Format**: 2D arrays of integers (0-9)
- **Color Mapping**: 0=Black, 1=Blue, 2=Red, 3=Green, 4=Yellow, 5=Gray, 6=Orange, 7=Purple, 8=Brown, 9=Pink

### Testing Your Submission

```bash
# Test submission format
python test_submission.py

# Create submission from sample task
python submission.py --task_file sample_task.json
```

See [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) for detailed submission instructions.

## Contributing

1. **Add new detectors** for additional pattern types
2. **Improve rule inference** with more sophisticated algorithms
3. **Add new transformations** for complex operations
4. **Enhance evaluation** with better metrics and analysis
5. **Optimize performance** for large grids and complex tasks
6. **Improve submission format** for better Kaggle integration

## License

This project is part of the ARC-AGI-2 benchmark challenge.

## References

- ARC-AGI-2: [arXiv paper](https://ar5iv.labs.arxiv.org)
- ARC-AGI-2 tasks: [GitHub repository](https://github.com)
- ARC Prize 2025: [Kaggle Competition](https://www.kaggle.com/competitions/arc-prize-2025/overview)
- Current AI performance on ARC-AGI-2 is very low, highlighting the need for novel solver architectures 