"""
Transformations for ARC-AGI-2 solver.

Defines base Transformation class and elementary transformations like
Translate, Rotate, Recolor, Reflect, etc.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from ..parser import Grid


class Transformation(ABC):
    """Base class for all transformations."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    @abstractmethod
    def apply(self, grid: Grid) -> Grid:
        """
        Apply transformation to a grid.
        
        Args:
            grid: Input grid
            
        Returns:
            Transformed grid
        """
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.params})"
    
    def __repr__(self):
        return self.__str__()


class Translate(Transformation):
    """Translate grid by specified offset."""
    
    def __init__(self, dx: int = 0, dy: int = 0):
        super().__init__(dx=dx, dy=dy)
        self.dx = dx
        self.dy = dy
    
    def apply(self, grid: Grid) -> Grid:
        """Apply translation to grid."""
        result = Grid([[0 for _ in range(grid.width)] for _ in range(grid.height)])
        
        for y in range(grid.height):
            for x in range(grid.width):
                new_y = y + self.dy
                new_x = x + self.dx
                
                if 0 <= new_y < grid.height and 0 <= new_x < grid.width:
                    result.data[new_y, new_x] = grid.data[y, x]
        
        return result


class Rotate(Transformation):
    """Rotate grid by specified number of 90-degree increments."""
    
    def __init__(self, k: int = 1):
        super().__init__(k=k)
        self.k = k
    
    def apply(self, grid: Grid) -> Grid:
        """Apply rotation to grid."""
        return grid.rotate_90(self.k)


class Reflect(Transformation):
    """Reflect grid along specified axis."""
    
    def __init__(self, axis: str = 'horizontal'):
        super().__init__(axis=axis)
        self.axis = axis
    
    def apply(self, grid: Grid) -> Grid:
        """Apply reflection to grid."""
        if self.axis == 'horizontal':
            return grid.flip_horizontal()
        elif self.axis == 'vertical':
            return grid.flip_vertical()
        else:
            raise ValueError(f"Unknown axis: {self.axis}")


class Recolor(Transformation):
    """Recolor grid using color mapping."""
    
    def __init__(self, color_mapping: Dict[int, int]):
        super().__init__(color_mapping=color_mapping)
        self.color_mapping = color_mapping
    
    def apply(self, grid: Grid) -> Grid:
        """Apply color mapping to grid."""
        result = grid.copy()
        
        for from_color, to_color in self.color_mapping.items():
            result.data[grid.data == from_color] = to_color
        
        return result


class Scale(Transformation):
    """Scale grid by specified factor."""
    
    def __init__(self, scale_factor: float):
        super().__init__(scale_factor=scale_factor)
        self.scale_factor = scale_factor
    
    def apply(self, grid: Grid) -> Grid:
        """Apply scaling to grid."""
        if self.scale_factor <= 0:
            raise ValueError("Scale factor must be positive")
        
        if self.scale_factor == 1:
            return grid.copy()
        
        new_height = int(grid.height * self.scale_factor)
        new_width = int(grid.width * self.scale_factor)
        
        result = Grid([[0 for _ in range(new_width)] for _ in range(new_height)])
        
        for y in range(new_height):
            for x in range(new_width):
                src_y = int(y / self.scale_factor)
                src_x = int(x / self.scale_factor)
                
                if 0 <= src_y < grid.height and 0 <= src_x < grid.width:
                    result.data[y, x] = grid.data[src_y, src_x]
        
        return result


class Crop(Transformation):
    """Crop grid to specified bounds."""
    
    def __init__(self, min_y: int, max_y: int, min_x: int, max_x: int):
        super().__init__(min_y=min_y, max_y=max_y, min_x=min_x, max_x=max_x)
        self.min_y = min_y
        self.max_y = max_y
        self.min_x = min_x
        self.max_x = max_x
    
    def apply(self, grid: Grid) -> Grid:
        """Apply cropping to grid."""
        return grid.crop(self.min_y, self.max_y, self.min_x, self.max_x)


class Pad(Transformation):
    """Pad grid to specified size."""
    
    def __init__(self, target_height: int, target_width: int, fill_value: int = 0):
        super().__init__(target_height=target_height, target_width=target_width, fill_value=fill_value)
        self.target_height = target_height
        self.target_width = target_width
        self.fill_value = fill_value
    
    def apply(self, grid: Grid) -> Grid:
        """Apply padding to grid."""
        return grid.pad(self.target_height, self.target_width, self.fill_value)


class FloodFill(Transformation):
    """Flood fill from a starting point."""
    
    def __init__(self, start_y: int, start_x: int, target_color: int, replacement_color: int):
        super().__init__(start_y=start_y, start_x=start_x, target_color=target_color, replacement_color=replacement_color)
        self.start_y = start_y
        self.start_x = start_x
        self.target_color = target_color
        self.replacement_color = replacement_color
    
    def apply(self, grid: Grid) -> Grid:
        """Apply flood fill to grid."""
        from ..grid import flood_fill
        return flood_fill(grid, self.start_y, self.start_x, self.target_color, self.replacement_color)


class ExtractSegment(Transformation):
    """Extract a segment from the grid."""
    
    def __init__(self, segment_mask: np.ndarray):
        super().__init__(segment_mask=segment_mask)
        self.segment_mask = segment_mask
    
    def apply(self, grid: Grid) -> Grid:
        """Extract segment from grid."""
        if grid.data.shape != self.segment_mask.shape:
            raise ValueError("Segment mask must match grid shape")
        
        result = Grid([[0 for _ in range(grid.width)] for _ in range(grid.height)])
        result.data[self.segment_mask] = grid.data[self.segment_mask]
        
        return result


class Pipeline(Transformation):
    """Apply multiple transformations in sequence."""
    
    def __init__(self, transformations: List[Transformation]):
        super().__init__(transformations=transformations)
        self.transformations = transformations
    
    def apply(self, grid: Grid) -> Grid:
        """Apply all transformations in sequence."""
        result = grid
        for transformation in self.transformations:
            result = transformation.apply(result)
        return result
    
    def __str__(self):
        return f"Pipeline({[str(t) for t in self.transformations]})"


class ConditionalTransform(Transformation):
    """Apply transformation only if condition is met."""
    
    def __init__(self, condition: callable, transformation: Transformation):
        super().__init__(condition=condition, transformation=transformation)
        self.condition = condition
        self.transformation = transformation
    
    def apply(self, grid: Grid) -> Grid:
        """Apply transformation if condition is met."""
        if self.condition(grid):
            return self.transformation.apply(grid)
        else:
            return grid.copy()


class CompositeTransform(Transformation):
    """Apply multiple transformations and combine results."""
    
    def __init__(self, transformations: List[Transformation], combine_func: callable = None):
        super().__init__(transformations=transformations, combine_func=combine_func)
        self.transformations = transformations
        self.combine_func = combine_func or self._default_combine
    
    def apply(self, grid: Grid) -> Grid:
        """Apply all transformations and combine results."""
        results = [t.apply(grid) for t in self.transformations]
        return self.combine_func(results)
    
    def _default_combine(self, grids: List[Grid]) -> Grid:
        """Default combination: take the first non-empty grid."""
        for grid in grids:
            if not grid.is_empty():
                return grid
        return grids[0] if grids else Grid([[0]])


# Factory functions for common transformations
def create_color_swap(from_color: int, to_color: int) -> Transformation:
    """Create a color swap transformation."""
    return Recolor({from_color: to_color})


def create_translation(dx: int, dy: int) -> Transformation:
    """Create a translation transformation."""
    return Translate(dx=dx, dy=dy)


def create_rotation(k: int) -> Transformation:
    """Create a rotation transformation."""
    return Rotate(k=k)


def create_reflection(axis: str) -> Transformation:
    """Create a reflection transformation."""
    return Reflect(axis=axis)


def create_pipeline(*transformations: Transformation) -> Transformation:
    """Create a pipeline of transformations."""
    return Pipeline(list(transformations))


def create_conditional(condition: callable, transformation: Transformation) -> Transformation:
    """Create a conditional transformation."""
    return ConditionalTransform(condition, transformation) 