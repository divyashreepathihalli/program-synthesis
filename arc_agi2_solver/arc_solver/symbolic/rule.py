"""
Symbolic rule representation for ARC-AGI-2 solver.

Represents candidate rules as symbolic objects, including mappings
from input objects to output objects and logic formulas.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
from ..parser import Grid
from ..transformations.transform import Transformation


class Rule(ABC):
    """Base class for symbolic rules."""
    
    def __init__(self, confidence: float = 0.0):
        self.confidence = confidence
    
    @abstractmethod
    def apply(self, grid: Grid) -> Grid:
        """
        Apply the rule to a grid.
        
        Args:
            grid: Input grid
            
        Returns:
            Transformed grid
        """
        pass
    
    @abstractmethod
    def matches(self, input_grid: Grid, output_grid: Grid) -> bool:
        """
        Check if this rule explains the input-output pair.
        
        Args:
            input_grid: Input grid
            output_grid: Expected output grid
            
        Returns:
            True if rule matches the pair
        """
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}(confidence={self.confidence})"


class TransformationRule(Rule):
    """Rule based on a single transformation."""
    
    def __init__(self, transformation: Transformation, confidence: float = 0.0):
        super().__init__(confidence)
        self.transformation = transformation
    
    def apply(self, grid: Grid) -> Grid:
        """Apply the transformation to the grid."""
        return self.transformation.apply(grid)
    
    def matches(self, input_grid: Grid, output_grid: Grid) -> bool:
        """Check if transformation produces expected output."""
        result = self.apply(input_grid)
        return result == output_grid
    
    def __str__(self):
        return f"TransformationRule({self.transformation}, confidence={self.confidence})"


class ConditionalRule(Rule):
    """Rule that applies different transformations based on conditions."""
    
    def __init__(self, conditions: List[Callable], transformations: List[Transformation], 
                 confidence: float = 0.0):
        super().__init__(confidence)
        self.conditions = conditions
        self.transformations = transformations
        
        if len(conditions) != len(transformations):
            raise ValueError("Number of conditions must match number of transformations")
    
    def apply(self, grid: Grid) -> Grid:
        """Apply the first transformation whose condition is met."""
        for condition, transformation in zip(self.conditions, self.transformations):
            if condition(grid):
                return transformation.apply(grid)
        
        # If no condition is met, return original grid
        return grid.copy()
    
    def matches(self, input_grid: Grid, output_grid: Grid) -> bool:
        """Check if rule produces expected output."""
        result = self.apply(input_grid)
        return result == output_grid
    
    def __str__(self):
        return f"ConditionalRule({len(self.conditions)} conditions, confidence={self.confidence})"


class ObjectMappingRule(Rule):
    """Rule that maps input objects to output objects."""
    
    def __init__(self, object_mappings: Dict[str, Dict[str, Any]], confidence: float = 0.0):
        super().__init__(confidence)
        self.object_mappings = object_mappings
    
    def apply(self, grid: Grid) -> Grid:
        """Apply object mappings to grid."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated object detection and mapping
        result = grid.copy()
        
        # Apply color mappings
        for mapping in self.object_mappings.values():
            if 'color_change' in mapping:
                from_color = mapping['color_change']['from']
                to_color = mapping['color_change']['to']
                result.data[result.data == from_color] = to_color
        
        return result
    
    def matches(self, input_grid: Grid, output_grid: Grid) -> bool:
        """Check if object mappings explain the transformation."""
        result = self.apply(input_grid)
        return result == output_grid
    
    def __str__(self):
        return f"ObjectMappingRule({len(self.object_mappings)} mappings, confidence={self.confidence})"


class PatternRule(Rule):
    """Rule based on pattern matching and replacement."""
    
    def __init__(self, pattern: Grid, replacement: Grid, confidence: float = 0.0):
        super().__init__(confidence)
        self.pattern = pattern
        self.replacement = replacement
    
    def apply(self, grid: Grid) -> Grid:
        """Replace pattern with replacement in grid."""
        from ..grid import find_subgrid_positions
        
        result = grid.copy()
        positions = find_subgrid_positions(grid, self.pattern)
        
        for y, x in positions:
            # Replace pattern with replacement
            pattern_height, pattern_width = self.pattern.height, self.pattern.width
            replacement_height, replacement_width = self.replacement.height, self.replacement.width
            
            # Ensure replacement fits
            if (y + replacement_height <= result.height and 
                x + replacement_width <= result.width):
                
                for ry in range(replacement_height):
                    for rx in range(replacement_width):
                        result.data[y + ry, x + rx] = self.replacement.data[ry, rx]
        
        return result
    
    def matches(self, input_grid: Grid, output_grid: Grid) -> bool:
        """Check if pattern replacement produces expected output."""
        result = self.apply(input_grid)
        return result == output_grid
    
    def __str__(self):
        return f"PatternRule(pattern=({self.pattern.height},{self.pattern.width}), replacement=({self.replacement.height},{self.replacement.width}), confidence={self.confidence})"


class CompositeRule(Rule):
    """Rule that combines multiple sub-rules."""
    
    def __init__(self, sub_rules: List[Rule], confidence: float = 0.0):
        super().__init__(confidence)
        self.sub_rules = sub_rules
    
    def apply(self, grid: Grid) -> Grid:
        """Apply all sub-rules in sequence."""
        result = grid
        for rule in self.sub_rules:
            result = rule.apply(result)
        return result
    
    def matches(self, input_grid: Grid, output_grid: Grid) -> bool:
        """Check if composite rule produces expected output."""
        result = self.apply(input_grid)
        return result == output_grid
    
    def __str__(self):
        return f"CompositeRule({len(self.sub_rules)} sub-rules, confidence={self.confidence})"


class LogicRule(Rule):
    """Rule based on logical conditions."""
    
    def __init__(self, condition: Callable[[Grid], bool], 
                 transformation: Transformation, confidence: float = 0.0):
        super().__init__(confidence)
        self.condition = condition
        self.transformation = transformation
    
    def apply(self, grid: Grid) -> Grid:
        """Apply transformation if condition is met."""
        if self.condition(grid):
            return self.transformation.apply(grid)
        else:
            return grid.copy()
    
    def matches(self, input_grid: Grid, output_grid: Grid) -> bool:
        """Check if logical rule produces expected output."""
        result = self.apply(input_grid)
        return result == output_grid
    
    def __str__(self):
        return f"LogicRule(condition={self.condition.__name__}, confidence={self.confidence})"


class RuleFactory:
    """Factory for creating rules from various sources."""
    
    @staticmethod
    def from_transformation(transformation: Transformation, confidence: float = 0.0) -> Rule:
        """Create a rule from a transformation."""
        return TransformationRule(transformation, confidence)
    
    @staticmethod
    def from_color_mapping(color_mapping: Dict[int, int], confidence: float = 0.0) -> Rule:
        """Create a rule from a color mapping."""
        from ..transformations.transform import Recolor
        transformation = Recolor(color_mapping)
        return TransformationRule(transformation, confidence)
    
    @staticmethod
    def from_object_mappings(mappings: Dict[str, Dict[str, Any]], confidence: float = 0.0) -> Rule:
        """Create a rule from object mappings."""
        return ObjectMappingRule(mappings, confidence)
    
    @staticmethod
    def from_pattern(pattern: Grid, replacement: Grid, confidence: float = 0.0) -> Rule:
        """Create a rule from a pattern and replacement."""
        return PatternRule(pattern, replacement, confidence)
    
    @staticmethod
    def composite(rules: List[Rule], confidence: float = 0.0) -> Rule:
        """Create a composite rule from multiple sub-rules."""
        return CompositeRule(rules, confidence)
    
    @staticmethod
    def conditional(condition: Callable[[Grid], bool], 
                   transformation: Transformation, confidence: float = 0.0) -> Rule:
        """Create a conditional rule."""
        return LogicRule(condition, transformation, confidence)


def create_simple_color_rule(from_color: int, to_color: int, confidence: float = 0.8) -> Rule:
    """Create a simple color swap rule."""
    return RuleFactory.from_color_mapping({from_color: to_color}, confidence)


def create_translation_rule(dx: int, dy: int, confidence: float = 0.7) -> Rule:
    """Create a translation rule."""
    from ..transformations.transform import Translate
    transformation = Translate(dx=dx, dy=dy)
    return RuleFactory.from_transformation(transformation, confidence)


def create_rotation_rule(k: int, confidence: float = 0.6) -> Rule:
    """Create a rotation rule."""
    from ..transformations.transform import Rotate
    transformation = Rotate(k=k)
    return RuleFactory.from_transformation(transformation, confidence) 