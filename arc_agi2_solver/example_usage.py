#!/usr/bin/env python3
"""
Example usage of the ARC-AGI-2 solver.

Demonstrates the modular architecture and various components
of the solver system.
"""

import numpy as np
from arc_solver import ARCSolver, Grid, Task
from arc_solver.detectors.color_detector import ColorDetector
from arc_solver.detectors.shape_detector import ShapeDetector
from arc_solver.detectors.pattern_detector import PatternDetector
from arc_solver.transformations.transform import Recolor, Translate, Rotate, Reflect
from arc_solver.symbolic.rule import RuleFactory


def demonstrate_basic_usage():
    """Demonstrate basic solver usage."""
    print("=== Basic Solver Usage ===")
    
    # Create solver
    solver = ARCSolver()
    
    # Create a simple color swap task
    input_grid = Grid([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    output_grid = Grid([[0, 2, 0], [2, 2, 2], [0, 2, 0]])
    train_pairs = [(input_grid, output_grid)]
    
    # Create task
    task = Task("simple_color_swap", train_pairs, [])
    
    print(f"Input grid:\n{input_grid}")
    print(f"Expected output:\n{output_grid}")
    
    # Solve the task
    results = solver.solve_task(task)
    print(f"Number of results: {len(results)}")
    
    if results:
        print(f"First result:\n{results[0]}")


def demonstrate_detectors():
    """Demonstrate individual detector components."""
    print("\n=== Detector Components ===")
    
    # Create a test grid
    grid = Grid([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ])
    
    print(f"Test grid:\n{grid}")
    
    # Color detector
    print("\n--- Color Detector ---")
    color_detector = ColorDetector()
    unique_colors = color_detector.get_unique_colors(grid)
    color_freq = color_detector.analyze_color_frequency(grid)
    print(f"Unique colors: {unique_colors}")
    print(f"Color frequencies: {color_freq}")
    
    # Shape detector
    print("\n--- Shape Detector ---")
    shape_detector = ShapeDetector()
    objects = shape_detector.find_all_objects(grid)
    print(f"Found {len(objects)} objects:")
    
    for i, obj in enumerate(objects):
        print(f"  Object {i+1}: Color={obj['color']}, Area={obj['area']}, Centroid={obj['centroid']}")
    
    # Pattern detector
    print("\n--- Pattern Detector ---")
    pattern_detector = PatternDetector()
    symmetry = pattern_detector.detect_symmetry(grid)
    alignment = pattern_detector.detect_alignment_patterns(grid)
    print(f"Symmetry: {symmetry}")
    print(f"Alignment: {alignment}")


def demonstrate_transformations():
    """Demonstrate transformation components."""
    print("\n=== Transformation Components ===")
    
    # Create a test grid
    grid = Grid([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])
    
    print(f"Original grid:\n{grid}")
    
    # Color transformation
    print("\n--- Color Transformation ---")
    recolor = Recolor({1: 2})
    colored_grid = recolor.apply(grid)
    print(f"After color swap (1->2):\n{colored_grid}")
    
    # Translation
    print("\n--- Translation ---")
    translate = Translate(dx=1, dy=1)
    translated_grid = translate.apply(grid)
    print(f"After translation (dx=1, dy=1):\n{translated_grid}")
    
    # Rotation
    print("\n--- Rotation ---")
    rotate = Rotate(k=1)
    rotated_grid = rotate.apply(grid)
    print(f"After 90-degree rotation:\n{rotated_grid}")
    
    # Reflection
    print("\n--- Reflection ---")
    reflect = Reflect(axis='horizontal')
    reflected_grid = reflect.apply(grid)
    print(f"After horizontal reflection:\n{reflected_grid}")


def demonstrate_rule_system():
    """Demonstrate the symbolic rule system."""
    print("\n=== Symbolic Rule System ===")
    
    # Create training pairs
    input_grid = Grid([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    output_grid = Grid([[0, 2, 0], [2, 2, 2], [0, 2, 0]])
    
    print(f"Input:\n{input_grid}")
    print(f"Expected output:\n{output_grid}")
    
    # Create a color swap rule
    color_rule = RuleFactory.from_color_mapping({1: 2}, confidence=0.9)
    print(f"\nCreated rule: {color_rule}")
    
    # Test the rule
    result = color_rule.apply(input_grid)
    print(f"Rule result:\n{result}")
    
    # Check if rule matches
    matches = color_rule.matches(input_grid, output_grid)
    print(f"Rule matches expected output: {matches}")


def demonstrate_pipeline():
    """Demonstrate the complete solving pipeline."""
    print("\n=== Complete Pipeline ===")
    
    # Create a more complex task
    input_grid1 = Grid([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    output_grid1 = Grid([[0, 2, 0], [2, 2, 2], [0, 2, 0]])
    
    input_grid2 = Grid([[0, 0, 1], [0, 1, 1], [1, 1, 0]])
    output_grid2 = Grid([[0, 0, 2], [0, 2, 2], [2, 2, 0]])
    
    train_pairs = [(input_grid1, output_grid1), (input_grid2, output_grid2)]
    
    print("Training pairs:")
    for i, (input_grid, output_grid) in enumerate(train_pairs):
        print(f"  Pair {i+1}:")
        print(f"    Input:\n{input_grid}")
        print(f"    Output:\n{output_grid}")
    
    # Create solver and solve
    solver = ARCSolver()
    task = Task("complex_task", train_pairs, [])
    
    print("\nSolving task...")
    results = solver.solve_task(task)
    
    print(f"Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"Result {i+1}:\n{result}")


def main():
    """Run all demonstrations."""
    print("ARC-AGI-2 Solver Demonstration")
    print("=" * 50)
    
    try:
        demonstrate_basic_usage()
        demonstrate_detectors()
        demonstrate_transformations()
        demonstrate_rule_system()
        demonstrate_pipeline()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 