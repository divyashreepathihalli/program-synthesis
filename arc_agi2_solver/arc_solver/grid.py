"""
Grid utilities for ARC-AGI-2 solver.

Provides utility functions for grid operations like rotation, reflection,
flood-fill, connected components, and pattern matching.
"""

import numpy as np
from typing import List, Tuple, Set, Optional, Dict, Any
from scipy import ndimage
from .parser import Grid


def find_connected_components(grid: Grid, color: int) -> List[Dict[str, Any]]:
    """
    Find connected components of a specific color in the grid.
    
    Args:
        grid: Input grid
        color: Color to find components for
        
    Returns:
        List of component dictionaries with properties
    """
    # Create binary mask for the color
    mask = (grid.data == color)
    
    # Find connected components
    labeled, num_features = ndimage.label(mask)
    
    components = []
    for i in range(1, num_features + 1):
        component_mask = (labeled == i)
        
        # Get bounding box
        rows = np.any(component_mask, axis=1)
        cols = np.any(component_mask, axis=0)
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        if len(row_indices) == 0 or len(col_indices) == 0:
            continue
            
        rmin, rmax = int(row_indices[0]), int(row_indices[-1])
        cmin, cmax = int(col_indices[0]), int(col_indices[-1])
        
        # Calculate properties
        area = int(np.sum(component_mask))
        centroid = ndimage.center_of_mass(component_mask)
        centroid = (float(centroid[0]), float(centroid[1]))
        
        # Get pixel coordinates
        pixels = np.column_stack(np.where(component_mask))
        pixel_list = [(int(p[0]), int(p[1])) for p in pixels]
        
        component = {
            'color': color,
            'mask': component_mask,
            'bbox': (rmin, rmax, cmin, cmax),
            'area': area,
            'centroid': centroid,
            'pixels': pixel_list,
            'width': cmax - cmin + 1,
            'height': rmax - rmin + 1
        }
        
        components.append(component)
    
    return components


def flood_fill(grid: Grid, start_y: int, start_x: int, target_color: int, 
               replacement_color: int) -> Grid:
    """
    Flood fill from a starting point.
    
    Args:
        grid: Input grid
        start_y, start_x: Starting coordinates
        target_color: Color to replace
        replacement_color: New color
        
    Returns:
        New grid with flood fill applied
    """
    if (start_y < 0 or start_y >= grid.height or 
        start_x < 0 or start_x >= grid.width):
        return grid.copy()
    
    if grid.data[start_y, start_x] != target_color:
        return grid.copy()
    
    # Use scipy's flood fill
    filled_data = grid.data.copy()
    mask = (filled_data == target_color)
    
    # Create a mask for the connected region
    structure = np.ones((3, 3), dtype=bool)
    filled_mask = ndimage.binary_fill_holes(mask)
    
    # Apply the fill
    filled_data[filled_mask] = replacement_color
    
    return Grid(filled_data.tolist())


def find_neighbors(grid: Grid, y: int, x: int, include_diagonals: bool = False) -> List[Tuple[int, int]]:
    """
    Find valid neighbors of a cell.
    
    Args:
        grid: Input grid
        y, x: Cell coordinates
        include_diagonals: Whether to include diagonal neighbors
        
    Returns:
        List of (y, x) coordinates of valid neighbors
    """
    neighbors = []
    
    if include_diagonals:
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if (0 <= ny < grid.height and 0 <= nx < grid.width):
            neighbors.append((ny, nx))
    
    return neighbors


def match_subgrid(grid: Grid, subgrid: Grid, start_y: int, start_x: int) -> bool:
    """
    Check if a subgrid matches at a specific position.
    
    Args:
        grid: Main grid
        subgrid: Subgrid to match
        start_y, start_x: Starting position in main grid
        
    Returns:
        True if subgrid matches at the position
    """
    if (start_y + subgrid.height > grid.height or 
        start_x + subgrid.width > grid.width):
        return False
    
    return np.array_equal(
        grid.data[start_y:start_y + subgrid.height, start_x:start_x + subgrid.width],
        subgrid.data
    )


def find_subgrid_positions(grid: Grid, subgrid: Grid) -> List[Tuple[int, int]]:
    """
    Find all positions where a subgrid appears in the main grid.
    
    Args:
        grid: Main grid
        subgrid: Subgrid to find
        
    Returns:
        List of (y, x) positions where subgrid appears
    """
    positions = []
    
    for y in range(grid.height - subgrid.height + 1):
        for x in range(grid.width - subgrid.width + 1):
            if match_subgrid(grid, subgrid, y, x):
                positions.append((y, x))
    
    return positions


def compare_grids(grid1: Grid, grid2: Grid) -> Dict[str, Any]:
    """
    Compare two grids and return differences.
    
    Args:
        grid1, grid2: Grids to compare
        
    Returns:
        Dictionary with comparison results
    """
    if grid1.height != grid2.height or grid1.width != grid2.width:
        return {
            'equal': False,
            'size_match': False,
            'different_pixels': None,
            'color_differences': None
        }
    
    # Find different pixels
    diff_mask = (grid1.data != grid2.data)
    different_pixels = np.column_stack(np.where(diff_mask))
    
    # Count color differences
    color_differences = {}
    for y, x in different_pixels:
        color1 = int(grid1.data[y, x])
        color2 = int(grid2.data[y, x])
        key = (color1, color2)
        color_differences[key] = color_differences.get(key, 0) + 1
    
    return {
        'equal': bool(np.array_equal(grid1.data, grid2.data)),
        'size_match': True,
        'different_pixels': [(int(p[0]), int(p[1])) for p in different_pixels],
        'color_differences': color_differences,
        'num_differences': len(different_pixels)
    }


def get_grid_statistics(grid: Grid) -> Dict[str, Any]:
    """
    Get comprehensive statistics about a grid.
    
    Args:
        grid: Input grid
        
    Returns:
        Dictionary with grid statistics
    """
    unique_colors = grid.get_unique_colors()
    color_counts = {color: grid.get_color_count(color) for color in unique_colors}
    
    # Find non-zero bounds
    bounds = grid.get_bounds()
    min_y, max_y, min_x, max_x = bounds
    
    # Calculate density
    total_pixels = grid.height * grid.width
    non_zero_pixels = total_pixels - color_counts.get(0, 0)
    density = non_zero_pixels / total_pixels if total_pixels > 0 else 0
    
    return {
        'shape': (grid.height, grid.width),
        'unique_colors': unique_colors,
        'color_counts': color_counts,
        'bounds': bounds,
        'density': density,
        'non_zero_pixels': non_zero_pixels,
        'total_pixels': total_pixels
    }


def extract_pattern(grid: Grid, pattern_size: Tuple[int, int], 
                   stride: Tuple[int, int] = (1, 1)) -> List[Grid]:
    """
    Extract all patterns of a given size from the grid.
    
    Args:
        grid: Input grid
        pattern_size: (height, width) of patterns to extract
        stride: (y_stride, x_stride) for extraction
        
    Returns:
        List of extracted pattern grids
    """
    pattern_height, pattern_width = pattern_size
    y_stride, x_stride = stride
    
    patterns = []
    
    for y in range(0, grid.height - pattern_height + 1, y_stride):
        for x in range(0, grid.width - pattern_width + 1, x_stride):
            pattern_data = grid.data[y:y + pattern_height, x:x + pattern_width]
            patterns.append(Grid(pattern_data.tolist()))
    
    return patterns


def find_symmetry_axes(grid: Grid) -> Dict[str, bool]:
    """
    Find symmetry axes in the grid.
    
    Args:
        grid: Input grid
        
    Returns:
        Dictionary with symmetry information
    """
    # Check horizontal symmetry
    horizontal_symmetric = np.array_equal(grid.data, grid.flip_horizontal().data)
    
    # Check vertical symmetry
    vertical_symmetric = np.array_equal(grid.data, grid.flip_vertical().data)
    
    # Check diagonal symmetries
    diagonal_symmetric = np.array_equal(grid.data, grid.rotate_90(2).flip_horizontal().data)
    
    return {
        'horizontal': bool(horizontal_symmetric),
        'vertical': bool(vertical_symmetric),
        'diagonal': bool(diagonal_symmetric),
        'any_symmetry': bool(horizontal_symmetric or vertical_symmetric or diagonal_symmetric)
    } 