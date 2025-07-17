"""
Parser module for ARC-AGI-2 tasks.

Defines Grid, Task, and Puzzle classes for loading and representing
ARC-AGI-2 tasks from JSON format.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class Grid:
    """Represents a 2D grid of integers (0-9 for colors)."""
    
    def __init__(self, data: List[List[int]]):
        """
        Initialize a Grid from a 2D list of integers.
        
        Args:
            data: 2D list of integers (0-9) representing colors
        """
        self.data = np.array(data, dtype=np.int8)
        self.height, self.width = self.data.shape
        
    def __getitem__(self, key):
        """Allow grid[y, x] indexing."""
        return self.data[key]
    
    def __setitem__(self, key, value):
        """Allow grid[y, x] = value assignment."""
        self.data[key] = value
    
    def __eq__(self, other):
        """Check if two grids are equal."""
        if not isinstance(other, Grid):
            return False
        return bool(np.array_equal(self.data, other.data))
    
    def __str__(self):
        """String representation of the grid."""
        return str(self.data)
    
    def __repr__(self):
        """Detailed representation of the grid."""
        return f"Grid(shape=({self.height}, {self.width}))"
    
    def copy(self) -> 'Grid':
        """Create a copy of this grid."""
        return Grid(self.data.copy().tolist())
    
    def get_unique_colors(self) -> List[int]:
        """Get list of unique colors in the grid."""
        return sorted(np.unique(self.data).tolist())
    
    def get_color_count(self, color: int) -> int:
        """Count occurrences of a specific color."""
        return np.sum(self.data == color)
    
    def is_empty(self) -> bool:
        """Check if grid is empty (all zeros)."""
        return np.all(self.data == 0)
    
    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get bounding box of non-zero elements (min_y, max_y, min_x, max_x)."""
        if self.is_empty():
            return (0, 0, 0, 0)
        
        non_zero = np.where(self.data != 0)
        if len(non_zero[0]) == 0:
            return (0, 0, 0, 0)
        
        min_y, max_y = int(np.min(non_zero[0])), int(np.max(non_zero[0]))
        min_x, max_x = int(np.min(non_zero[1])), int(np.max(non_zero[1]))
        return (min_y, max_y, min_x, max_x)
    
    def crop(self, min_y: int, max_y: int, min_x: int, max_x: int) -> 'Grid':
        """Crop grid to specified bounds."""
        cropped_data = self.data[min_y:max_y+1, min_x:max_x+1]
        return Grid(cropped_data.tolist())
    
    def pad(self, target_height: int, target_width: int, fill_value: int = 0) -> 'Grid':
        """Pad grid to target size with fill_value."""
        if target_height < self.height or target_width < self.width:
            raise ValueError("Target size must be larger than current size")
        
        padded = np.full((target_height, target_width), fill_value, dtype=np.int8)
        padded[:self.height, :self.width] = self.data
        return Grid(padded.tolist())
    
    def rotate_90(self, k: int = 1) -> 'Grid':
        """Rotate grid by k*90 degrees clockwise."""
        rotated = np.rot90(self.data, k=-k)  # numpy uses counter-clockwise
        return Grid(rotated.tolist())
    
    def flip_horizontal(self) -> 'Grid':
        """Flip grid horizontally."""
        flipped = np.flip(self.data, axis=1)
        return Grid(flipped.tolist())
    
    def flip_vertical(self) -> 'Grid':
        """Flip grid vertically."""
        flipped = np.flip(self.data, axis=0)
        return Grid(flipped.tolist())
    
    def to_list(self) -> List[List[int]]:
        """Convert to 2D list."""
        return self.data.tolist()


class Task:
    """Represents a single ARC-AGI-2 task with train and test pairs."""
    
    def __init__(self, task_id: str, train_pairs: List[Tuple[Grid, Grid]], 
                 test_pairs: List[Tuple[Grid, Grid]]):
        """
        Initialize a Task.
        
        Args:
            task_id: Unique identifier for the task
            train_pairs: List of (input, output) grid pairs for training
            test_pairs: List of (input, output) grid pairs for testing
        """
        self.task_id = task_id
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
    
    def __str__(self):
        """String representation of the task."""
        return f"Task({self.task_id}, {len(self.train_pairs)} train, {len(self.test_pairs)} test)"
    
    def get_train_inputs(self) -> List[Grid]:
        """Get all training input grids."""
        return [pair[0] for pair in self.train_pairs]
    
    def get_train_outputs(self) -> List[Grid]:
        """Get all training output grids."""
        return [pair[1] for pair in self.train_pairs]
    
    def get_test_inputs(self) -> List[Grid]:
        """Get all test input grids."""
        return [pair[0] for pair in self.test_pairs]
    
    def get_test_outputs(self) -> List[Grid]:
        """Get all test output grids."""
        return [pair[1] for pair in self.test_pairs]


class Puzzle:
    """Collection of ARC-AGI-2 tasks."""
    
    def __init__(self, tasks: List[Task]):
        """
        Initialize a Puzzle with a list of tasks.
        
        Args:
            tasks: List of Task objects
        """
        self.tasks = tasks
        self.task_dict = {task.task_id: task for task in tasks}
    
    def __len__(self):
        """Number of tasks in the puzzle."""
        return len(self.tasks)
    
    def __getitem__(self, task_id: str) -> Task:
        """Get task by ID."""
        return self.task_dict[task_id]
    
    def get_task_ids(self) -> List[str]:
        """Get list of all task IDs."""
        return list(self.task_dict.keys())


def load_task_from_json(json_path: str) -> Task:
    """
    Load a single task from JSON file.
    
    Args:
        json_path: Path to JSON file containing task data
        
    Returns:
        Task object
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract task ID from filename
    task_id = Path(json_path).stem
    
    # Parse train pairs
    train_pairs = []
    for pair in data.get('train', []):
        input_grid = Grid(pair['input'])
        output_grid = Grid(pair['output'])
        train_pairs.append((input_grid, output_grid))
    
    # Parse test pairs
    test_pairs = []
    for pair in data.get('test', []):
        input_grid = Grid(pair['input'])
        output_grid = Grid(pair['output'])
        test_pairs.append((input_grid, output_grid))
    
    return Task(task_id, train_pairs, test_pairs)


def load_puzzle_from_directory(directory_path: str) -> Puzzle:
    """
    Load all tasks from a directory containing JSON files.
    
    Args:
        directory_path: Path to directory containing task JSON files
        
    Returns:
        Puzzle object containing all tasks
    """
    directory = Path(directory_path)
    tasks = []
    
    for json_file in directory.glob("*.json"):
        try:
            task = load_task_from_json(str(json_file))
            tasks.append(task)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    return Puzzle(tasks)


def save_task_to_json(task: Task, json_path: str):
    """
    Save a task to JSON file.
    
    Args:
        task: Task object to save
        json_path: Path to save the JSON file
    """
    data = {
        'train': [
            {'input': pair[0].to_list(), 'output': pair[1].to_list()}
            for pair in task.train_pairs
        ],
        'test': [
            {'input': pair[0].to_list(), 'output': pair[1].to_list()}
            for pair in task.test_pairs
        ]
    }
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2) 