"""
Color detector for ARC-AGI-2 solver.

Analyzes color frequency, unique colors, and color-matching patterns
across input/output pairs.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from ..parser import Grid


class ColorDetector:
    """Detects color patterns and transformations in ARC tasks."""
    
    def __init__(self):
        self.color_mappings = {}
        self.color_frequencies = {}
    
    def analyze_color_frequency(self, grid: Grid) -> Dict[int, int]:
        """
        Analyze color frequency in a grid.
        
        Args:
            grid: Input grid
            
        Returns:
            Dictionary mapping colors to their counts
        """
        unique, counts = np.unique(grid.data, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_unique_colors(self, grid: Grid) -> List[int]:
        """
        Get list of unique colors in a grid.
        
        Args:
            grid: Input grid
            
        Returns:
            List of unique colors (sorted)
        """
        return sorted(np.unique(grid.data).tolist())
    
    def detect_color_mapping(self, input_grid: Grid, output_grid: Grid) -> Dict[int, int]:
        """
        Detect color mapping between input and output grids.
        
        Args:
            input_grid: Input grid
            output_grid: Output grid
            
        Returns:
            Dictionary mapping input colors to output colors
        """
        if input_grid.height != output_grid.height or input_grid.width != output_grid.width:
            return {}
        
        # Find corresponding positions with different colors
        color_mapping = {}
        input_colors = input_grid.get_unique_colors()
        output_colors = output_grid.get_unique_colors()
        
        # Simple mapping: assume colors map in order of frequency
        input_freq = self.analyze_color_frequency(input_grid)
        output_freq = self.analyze_color_frequency(output_grid)
        
        # Sort by frequency
        input_sorted = sorted(input_freq.items(), key=lambda x: x[1], reverse=True)
        output_sorted = sorted(output_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Map colors by frequency order
        for i, (in_color, _) in enumerate(input_sorted):
            if i < len(output_sorted):
                out_color = output_sorted[i][0]
                if in_color != out_color:
                    color_mapping[in_color] = out_color
        
        return color_mapping
    
    def detect_color_swaps(self, train_pairs: List[Tuple[Grid, Grid]]) -> Dict[int, int]:
        """
        Detect consistent color swaps across training pairs.
        
        Args:
            train_pairs: List of (input, output) grid pairs
            
        Returns:
            Dictionary mapping input colors to output colors
        """
        all_mappings = []
        
        for input_grid, output_grid in train_pairs:
            mapping = self.detect_color_mapping(input_grid, output_grid)
            all_mappings.append(mapping)
        
        # Find consistent mappings across all pairs
        consistent_mapping = {}
        if all_mappings:
            # Start with the first mapping
            consistent_mapping = all_mappings[0].copy()
            
            # Check consistency with other mappings
            for mapping in all_mappings[1:]:
                for in_color, out_color in mapping.items():
                    if in_color in consistent_mapping:
                        if consistent_mapping[in_color] != out_color:
                            # Inconsistent mapping, remove it
                            del consistent_mapping[in_color]
                    else:
                        consistent_mapping[in_color] = out_color
        
        return consistent_mapping
    
    def detect_color_preservation(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[int]:
        """
        Detect colors that are preserved across all training pairs.
        
        Args:
            train_pairs: List of (input, output) grid pairs
            
        Returns:
            List of colors that are preserved
        """
        preserved_colors = set()
        
        for input_grid, output_grid in train_pairs:
            input_colors = set(input_grid.get_unique_colors())
            output_colors = set(output_grid.get_unique_colors())
            
            # Colors present in both input and output
            common_colors = input_colors & output_colors
            
            if not preserved_colors:
                preserved_colors = common_colors
            else:
                preserved_colors &= common_colors
        
        return list(preserved_colors)
    
    def detect_color_addition(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[int]:
        """
        Detect colors that are added in outputs but not in inputs.
        
        Args:
            train_pairs: List of (input, output) grid pairs
            
        Returns:
            List of colors that are consistently added
        """
        added_colors = set()
        
        for input_grid, output_grid in train_pairs:
            input_colors = set(input_grid.get_unique_colors())
            output_colors = set(output_grid.get_unique_colors())
            
            # Colors in output but not in input
            new_colors = output_colors - input_colors
            
            if not added_colors:
                added_colors = new_colors
            else:
                added_colors &= new_colors
        
        return list(added_colors)
    
    def detect_color_removal(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[int]:
        """
        Detect colors that are removed from inputs in outputs.
        
        Args:
            train_pairs: List of (input, output) grid pairs
            
        Returns:
            List of colors that are consistently removed
        """
        removed_colors = set()
        
        for input_grid, output_grid in train_pairs:
            input_colors = set(input_grid.get_unique_colors())
            output_colors = set(output_grid.get_unique_colors())
            
            # Colors in input but not in output
            missing_colors = input_colors - output_colors
            
            if not removed_colors:
                removed_colors = missing_colors
            else:
                removed_colors &= missing_colors
        
        return list(removed_colors)
    
    def analyze_color_patterns(self, train_pairs: List[Tuple[Grid, Grid]]) -> Dict[str, Any]:
        """
        Comprehensive color pattern analysis.
        
        Args:
            train_pairs: List of (input, output) grid pairs
            
        Returns:
            Dictionary with color analysis results
        """
        # Detect various color patterns
        color_swaps = self.detect_color_swaps(train_pairs)
        preserved_colors = self.detect_color_preservation(train_pairs)
        added_colors = self.detect_color_addition(train_pairs)
        removed_colors = self.detect_color_removal(train_pairs)
        
        # Analyze color frequency patterns
        input_freqs = []
        output_freqs = []
        
        for input_grid, output_grid in train_pairs:
            input_freqs.append(self.analyze_color_frequency(input_grid))
            output_freqs.append(self.analyze_color_frequency(output_grid))
        
        # Check if color counts change consistently
        color_count_changes = {}
        for color in set().union(*[freq.keys() for freq in input_freqs + output_freqs]):
            input_counts = [freq.get(color, 0) for freq in input_freqs]
            output_counts = [freq.get(color, 0) for freq in output_freqs]
            
            if len(input_counts) == len(output_counts):
                # Check if there's a consistent pattern
                changes = [out_count - in_count for in_count, out_count in zip(input_counts, output_counts)]
                if len(set(changes)) == 1:  # All changes are the same
                    color_count_changes[color] = changes[0]
        
        return {
            'color_swaps': color_swaps,
            'preserved_colors': preserved_colors,
            'added_colors': added_colors,
            'removed_colors': removed_colors,
            'color_count_changes': color_count_changes,
            'has_color_transformation': bool(color_swaps or added_colors or removed_colors)
        }
    
    def suggest_color_transformations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest color transformations based on analysis.
        
        Args:
            analysis: Color analysis results
            
        Returns:
            List of suggested color transformations
        """
        suggestions = []
        
        # Color swap suggestions
        for from_color, to_color in analysis['color_swaps'].items():
            suggestions.append({
                'type': 'color_swap',
                'from_color': from_color,
                'to_color': to_color,
                'confidence': 0.9
            })
        
        # Color addition suggestions
        for color in analysis['added_colors']:
            suggestions.append({
                'type': 'color_addition',
                'color': color,
                'confidence': 0.8
            })
        
        # Color removal suggestions
        for color in analysis['removed_colors']:
            suggestions.append({
                'type': 'color_removal',
                'color': color,
                'confidence': 0.8
            })
        
        # Color count change suggestions
        for color, change in analysis['color_count_changes'].items():
            if change > 0:
                suggestions.append({
                    'type': 'color_increase',
                    'color': color,
                    'amount': change,
                    'confidence': 0.7
                })
            elif change < 0:
                suggestions.append({
                    'type': 'color_decrease',
                    'color': color,
                    'amount': abs(change),
                    'confidence': 0.7
                })
        
        return suggestions 