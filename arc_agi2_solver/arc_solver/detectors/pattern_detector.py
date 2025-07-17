"""
Pattern detector for ARC-AGI-2 solver.

Detects higher-level spatial patterns: lines, symmetry axes, repeated motifs,
relative positioning, and alignment patterns.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ..parser import Grid
from ..grid import find_symmetry_axes, extract_pattern


class PatternDetector:
    """Detects spatial patterns and motifs in ARC grids."""
    
    def __init__(self):
        self.pattern_cache = {}
    
    def detect_lines(self, grid: Grid) -> List[Dict[str, Any]]:
        """
        Detect lines of the same color in the grid.
        
        Args:
            grid: Input grid
            
        Returns:
            List of line dictionaries with properties
        """
        lines = []
        unique_colors = grid.get_unique_colors()
        
        for color in unique_colors:
            if color == 0:  # Skip background
                continue
            
            # Find all positions of this color
            color_positions = np.where(grid.data == color)
            y_coords, x_coords = color_positions
            
            if len(y_coords) < 2:
                continue
            
            # Check for horizontal lines
            horizontal_lines = self._find_horizontal_lines(y_coords, x_coords, color)
            lines.extend(horizontal_lines)
            
            # Check for vertical lines
            vertical_lines = self._find_vertical_lines(y_coords, x_coords, color)
            lines.extend(vertical_lines)
            
            # Check for diagonal lines
            diagonal_lines = self._find_diagonal_lines(y_coords, x_coords, color)
            lines.extend(diagonal_lines)
        
        return lines
    
    def _find_horizontal_lines(self, y_coords: np.ndarray, x_coords: np.ndarray, 
                             color: int) -> List[Dict[str, Any]]:
        """Find horizontal lines of the same color."""
        lines = []
        
        # Group by y-coordinate
        unique_y = np.unique(y_coords)
        for y in unique_y:
            x_at_y = x_coords[y_coords == y]
            if len(x_at_y) >= 2:
                x_at_y = np.sort(x_at_y)
                
                # Find consecutive sequences
                sequences = self._find_consecutive_sequences(x_at_y)
                for seq in sequences:
                    if len(seq) >= 2:
                        lines.append({
                            'type': 'horizontal',
                            'color': color,
                            'y': int(y),
                            'x_start': int(seq[0]),
                            'x_end': int(seq[-1]),
                            'length': len(seq),
                            'positions': [(int(y), int(x)) for x in seq]
                        })
        
        return lines
    
    def _find_vertical_lines(self, y_coords: np.ndarray, x_coords: np.ndarray, 
                           color: int) -> List[Dict[str, Any]]:
        """Find vertical lines of the same color."""
        lines = []
        
        # Group by x-coordinate
        unique_x = np.unique(x_coords)
        for x in unique_x:
            y_at_x = y_coords[x_coords == x]
            if len(y_at_x) >= 2:
                y_at_x = np.sort(y_at_x)
                
                # Find consecutive sequences
                sequences = self._find_consecutive_sequences(y_at_x)
                for seq in sequences:
                    if len(seq) >= 2:
                        lines.append({
                            'type': 'vertical',
                            'color': color,
                            'x': int(x),
                            'y_start': int(seq[0]),
                            'y_end': int(seq[-1]),
                            'length': len(seq),
                            'positions': [(int(y), int(x)) for y in seq]
                        })
        
        return lines
    
    def _find_diagonal_lines(self, y_coords: np.ndarray, x_coords: np.ndarray, 
                           color: int) -> List[Dict[str, Any]]:
        """Find diagonal lines of the same color."""
        lines = []
        
        # Check for diagonal patterns
        positions = list(zip(y_coords, x_coords))
        
        # Check main diagonal (y = x + c)
        for start_pos in positions:
            diagonal_positions = self._find_diagonal_sequence(positions, start_pos, 1, 1)
            if len(diagonal_positions) >= 2:
                lines.append({
                    'type': 'diagonal',
                    'color': color,
                    'direction': 'main',
                    'length': len(diagonal_positions),
                    'positions': diagonal_positions
                })
            
            # Check anti-diagonal (y = -x + c)
            anti_diagonal_positions = self._find_diagonal_sequence(positions, start_pos, 1, -1)
            if len(anti_diagonal_positions) >= 2:
                lines.append({
                    'type': 'diagonal',
                    'color': color,
                    'direction': 'anti',
                    'length': len(anti_diagonal_positions),
                    'positions': anti_diagonal_positions
                })
        
        return lines
    
    def _find_consecutive_sequences(self, coords: np.ndarray) -> List[List[int]]:
        """Find consecutive sequences in sorted coordinates."""
        sequences = []
        current_seq = [coords[0]]
        
        for i in range(1, len(coords)):
            if coords[i] == coords[i-1] + 1:
                current_seq.append(coords[i])
            else:
                if len(current_seq) >= 2:
                    sequences.append(current_seq)
                current_seq = [coords[i]]
        
        if len(current_seq) >= 2:
            sequences.append(current_seq)
        
        return sequences
    
    def _find_diagonal_sequence(self, positions: List[Tuple[int, int]], 
                               start_pos: Tuple[int, int], dy: int, dx: int) -> List[Tuple[int, int]]:
        """Find diagonal sequence starting from a position."""
        sequence = [start_pos]
        y, x = start_pos
        
        while True:
            next_y, next_x = y + dy, x + dx
            if (next_y, next_x) in positions:
                sequence.append((next_y, next_x))
                y, x = next_y, next_x
            else:
                break
        
        return sequence
    
    def detect_symmetry(self, grid: Grid) -> Dict[str, Any]:
        """
        Detect symmetry patterns in the grid.
        
        Args:
            grid: Input grid
            
        Returns:
            Dictionary with symmetry information
        """
        # Get basic symmetry
        basic_symmetry = find_symmetry_axes(grid)
        
        # Check for rotational symmetry
        rotational_symmetry = self._check_rotational_symmetry(grid)
        
        # Check for translational symmetry (repeating patterns)
        translational_symmetry = self._check_translational_symmetry(grid)
        
        return {
            **basic_symmetry,
            'rotational_symmetry': rotational_symmetry,
            'translational_symmetry': translational_symmetry,
            'any_symmetry': (basic_symmetry['any_symmetry'] or 
                           rotational_symmetry['has_rotational_symmetry'] or
                           translational_symmetry['has_translational_symmetry'])
        }
    
    def _check_rotational_symmetry(self, grid: Grid) -> Dict[str, Any]:
        """Check for rotational symmetry."""
        # Check 90-degree rotations
        rotated_90 = grid.rotate_90(1)
        symmetric_90 = np.array_equal(grid.data, rotated_90.data)
        
        # Check 180-degree rotations
        rotated_180 = grid.rotate_90(2)
        symmetric_180 = np.array_equal(grid.data, rotated_180.data)
        
        # Check 270-degree rotations
        rotated_270 = grid.rotate_90(3)
        symmetric_270 = np.array_equal(grid.data, rotated_270.data)
        
        return {
            'symmetric_90': bool(symmetric_90),
            'symmetric_180': bool(symmetric_180),
            'symmetric_270': bool(symmetric_270),
            'has_rotational_symmetry': bool(symmetric_90 or symmetric_180 or symmetric_270)
        }
    
    def _check_translational_symmetry(self, grid: Grid) -> Dict[str, Any]:
        """Check for translational symmetry (repeating patterns)."""
        # Check for repeating patterns
        for pattern_height in range(1, grid.height // 2 + 1):
            for pattern_width in range(1, grid.width // 2 + 1):
                if self._has_repeating_pattern(grid, pattern_height, pattern_width):
                    return {
                        'has_translational_symmetry': True,
                        'pattern_height': pattern_height,
                        'pattern_width': pattern_width
                    }
        
        return {
            'has_translational_symmetry': False,
            'pattern_height': None,
            'pattern_width': None
        }
    
    def _has_repeating_pattern(self, grid: Grid, pattern_height: int, pattern_width: int) -> bool:
        """Check if grid has a repeating pattern of given size."""
        if pattern_height > grid.height or pattern_width > grid.width:
            return False
        
        # Extract the base pattern
        base_pattern = grid.data[:pattern_height, :pattern_width]
        
        # Check if the pattern repeats
        for y in range(0, grid.height, pattern_height):
            for x in range(0, grid.width, pattern_width):
                if (y + pattern_height <= grid.height and 
                    x + pattern_width <= grid.width):
                    current_pattern = grid.data[y:y+pattern_height, x:x+pattern_width]
                    if not np.array_equal(current_pattern, base_pattern):
                        return False
        
        return True
    
    def detect_repeated_motifs(self, grid: Grid, min_size: int = 2, max_size: int = 5) -> List[Dict[str, Any]]:
        """
        Detect repeated motifs in the grid.
        
        Args:
            grid: Input grid
            min_size: Minimum motif size
            max_size: Maximum motif size
            
        Returns:
            List of motif dictionaries
        """
        motifs = []
        
        for height in range(min_size, min(max_size + 1, grid.height + 1)):
            for width in range(min_size, min(max_size + 1, grid.width + 1)):
                patterns = extract_pattern(grid, (height, width))
                
                # Count occurrences of each pattern
                pattern_counts = {}
                for pattern in patterns:
                    pattern_key = str(pattern.data.tolist())
                    pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
                
                # Find patterns that repeat
                for pattern_key, count in pattern_counts.items():
                    if count >= 2:
                        pattern_data = eval(pattern_key)  # Convert back to list
                        motifs.append({
                            'height': height,
                            'width': width,
                            'pattern': pattern_data,
                            'count': count,
                            'positions': self._find_motif_positions(grid, pattern_data)
                        })
        
        return motifs
    
    def _find_motif_positions(self, grid: Grid, motif_data: List[List[int]]) -> List[Tuple[int, int]]:
        """Find all positions where a motif appears."""
        positions = []
        motif_height, motif_width = len(motif_data), len(motif_data[0])
        
        for y in range(grid.height - motif_height + 1):
            for x in range(grid.width - motif_width + 1):
                if self._matches_motif(grid, motif_data, y, x):
                    positions.append((y, x))
        
        return positions
    
    def _matches_motif(self, grid: Grid, motif_data: List[List[int]], 
                      start_y: int, start_x: int) -> bool:
        """Check if motif matches at given position."""
        motif_height, motif_width = len(motif_data), len(motif_data[0])
        
        for y in range(motif_height):
            for x in range(motif_width):
                if grid.data[start_y + y, start_x + x] != motif_data[y][x]:
                    return False
        
        return True
    
    def detect_alignment_patterns(self, grid: Grid) -> Dict[str, Any]:
        """
        Detect alignment patterns in the grid.
        
        Args:
            grid: Input grid
            
        Returns:
            Dictionary with alignment information
        """
        # Find all non-zero positions
        non_zero_positions = np.where(grid.data != 0)
        y_coords, x_coords = non_zero_positions
        
        if len(y_coords) < 2:
            return {'aligned': False, 'alignment_type': None}
        
        # Check horizontal alignment
        unique_y = np.unique(y_coords)
        horizontal_aligned = len(unique_y) == 1
        
        # Check vertical alignment
        unique_x = np.unique(x_coords)
        vertical_aligned = len(unique_x) == 1
        
        # Check diagonal alignment
        diagonal_aligned = self._check_diagonal_alignment(y_coords, x_coords)
        
        # Determine alignment type
        if horizontal_aligned:
            alignment_type = 'horizontal'
        elif vertical_aligned:
            alignment_type = 'vertical'
        elif diagonal_aligned:
            alignment_type = 'diagonal'
        else:
            alignment_type = 'none'
        
        return {
            'aligned': bool(horizontal_aligned or vertical_aligned or diagonal_aligned),
            'alignment_type': alignment_type,
            'horizontal_aligned': bool(horizontal_aligned),
            'vertical_aligned': bool(vertical_aligned),
            'diagonal_aligned': bool(diagonal_aligned)
        }
    
    def _check_diagonal_alignment(self, y_coords: np.ndarray, x_coords: np.ndarray) -> bool:
        """Check if points are aligned diagonally."""
        if len(y_coords) < 2:
            return False
        
        # Check if all points lie on a diagonal line
        # Calculate differences
        y_diffs = y_coords[1:] - y_coords[:-1]
        x_diffs = x_coords[1:] - x_coords[:-1]
        
        # Check if all differences have the same ratio
        ratios = x_diffs / y_diffs
        return len(np.unique(np.round(ratios, 3))) == 1
    
    def suggest_pattern_transformations(self, input_patterns: Dict[str, Any], 
                                     output_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest pattern transformations based on pattern analysis.
        
        Args:
            input_patterns: Pattern analysis of input grid
            output_patterns: Pattern analysis of output grid
            
        Returns:
            List of suggested pattern transformations
        """
        suggestions = []
        
        # Check for symmetry changes
        if input_patterns.get('symmetry', {}).get('any_symmetry') != output_patterns.get('symmetry', {}).get('any_symmetry'):
            suggestions.append({
                'type': 'symmetry_change',
                'confidence': 0.7
            })
        
        # Check for alignment changes
        if input_patterns.get('alignment', {}).get('aligned') != output_patterns.get('alignment', {}).get('aligned'):
            suggestions.append({
                'type': 'alignment_change',
                'confidence': 0.6
            })
        
        # Check for motif changes
        input_motifs = len(input_patterns.get('motifs', []))
        output_motifs = len(output_patterns.get('motifs', []))
        if input_motifs != output_motifs:
            suggestions.append({
                'type': 'motif_count_change',
                'from_count': input_motifs,
                'to_count': output_motifs,
                'confidence': 0.5
            })
        
        return suggestions 