"""
ARC-AGI-2 Solver Package

A modular solver for the ARC-AGI-2 benchmark that uses symbolic reasoning
and compositional transformations to solve grid-based puzzles.
"""

from .parser import Grid, Task, Puzzle
from .pipeline import ARCSolver
from .transformations.transform import Transformation, Pipeline

__version__ = "1.0.0"
__all__ = ["Grid", "Task", "Puzzle", "ARCSolver", "Transformation", "Pipeline"] 