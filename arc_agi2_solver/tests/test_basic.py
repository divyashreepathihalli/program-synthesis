#!/usr/bin/env python3
"""
Basic tests for ARC-AGI-2 solver.

Tests core functionality to ensure the solver works correctly.
"""

import unittest
import numpy as np
from arc_solver import ARCSolver, Grid, Task


class TestARCSolver(unittest.TestCase):
    """Test cases for the ARC solver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.solver = ARCSolver()
    
    def test_grid_creation(self):
        """Test Grid creation and basic operations."""
        data = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        grid = Grid(data)
        
        self.assertEqual(grid.height, 3)
        self.assertEqual(grid.width, 3)
        self.assertEqual(grid.data.tolist(), data)
    
    def test_simple_color_swap(self):
        """Test solving a simple color swap task."""
        # Create a simple task: swap color 1 to color 2
        input_grid = Grid([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        output_grid = Grid([[0, 2, 0], [2, 2, 2], [0, 2, 0]])
        
        train_pairs = [(input_grid, output_grid)]
        task = Task("simple_color_swap", train_pairs, [])
        
        # Solve the task
        results = self.solver.solve_task(task)
        
        # Should return at least one result
        self.assertGreater(len(results), 0)
        
        # Check that results are Grid objects
        for result in results:
            self.assertIsInstance(result, Grid)
    
    def test_grid_operations(self):
        """Test grid utility operations."""
        data = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        grid = Grid(data)
        
        # Test rotation
        rotated = grid.rotate(1)  # 90 degrees
        self.assertEqual(rotated.height, 3)
        self.assertEqual(rotated.width, 3)
        
        # Test reflection
        reflected = grid.reflect('horizontal')
        self.assertEqual(reflected.height, 3)
        self.assertEqual(reflected.width, 3)
    
    def test_task_creation(self):
        """Test Task creation and validation."""
        input_grid = Grid([[1, 1], [1, 1]])
        output_grid = Grid([[2, 2], [2, 2]])
        
        train_pairs = [(input_grid, output_grid)]
        test_pairs = [(Grid([[1, 0], [0, 1]]), Grid([[2, 0], [0, 2]]))]
        
        task = Task("test_task", train_pairs, test_pairs)
        
        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(len(task.train_pairs), 1)
        self.assertEqual(len(task.test_pairs), 1)


class TestDetectors(unittest.TestCase):
    """Test cases for detectors."""
    
    def test_color_detector(self):
        """Test color detector functionality."""
        from arc_solver.detectors.color_detector import ColorDetector
        
        detector = ColorDetector()
        grid = Grid([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        
        analysis = detector.analyze_color_patterns([(grid, grid)])
        self.assertIsInstance(analysis, dict)
    
    def test_shape_detector(self):
        """Test shape detector functionality."""
        from arc_solver.detectors.shape_detector import ShapeDetector
        
        detector = ShapeDetector()
        grid = Grid([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        
        objects = detector.find_all_objects(grid)
        self.assertIsInstance(objects, list)


class TestTransformations(unittest.TestCase):
    """Test cases for transformations."""
    
    def test_recolor_transformation(self):
        """Test recolor transformation."""
        from arc_solver.transformations.transform import Recolor
        
        grid = Grid([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        recolor = Recolor({1: 2})
        
        result = recolor.apply(grid)
        self.assertIsInstance(result, Grid)
        
        # Check that 1s became 2s
        self.assertTrue(np.any(result.data == 2))
    
    def test_translate_transformation(self):
        """Test translate transformation."""
        from arc_solver.transformations.transform import Translate
        
        grid = Grid([[1, 0], [0, 0]])
        translate = Translate(dx=1, dy=1)
        
        result = translate.apply(grid)
        self.assertIsInstance(result, Grid)


if __name__ == "__main__":
    unittest.main() 