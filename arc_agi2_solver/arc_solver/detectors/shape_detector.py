"""
Shape detector for ARC-AGI-2 solver.

Identifies distinct objects via connected-component labeling for each color.
Computes bounding box, shape descriptors, and attributes for each object.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy import ndimage
from ..parser import Grid
from ..grid import find_connected_components


class ShapeDetector:
    """Detects and analyzes shapes/objects in ARC grids."""
    
    def __init__(self):
        self.object_cache = {}
    
    def find_all_objects(self, grid: Grid) -> List[Dict[str, Any]]:
        """
        Find all objects in the grid using connected component analysis.
        
        Args:
            grid: Input grid
            
        Returns:
            List of object dictionaries with properties
        """
        all_objects = []
        unique_colors = grid.get_unique_colors()
        
        # Skip background color (usually 0)
        background_color = 0
        if background_color in unique_colors:
            unique_colors.remove(background_color)
        
        for color in unique_colors:
            components = find_connected_components(grid, color)
            all_objects.extend(components)
        
        return all_objects
    
    def analyze_object_properties(self, object_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze detailed properties of an object.
        
        Args:
            object_data: Object data from find_connected_components
            
        Returns:
            Dictionary with detailed object properties
        """
        mask = object_data['mask']
        bbox = object_data['bbox']
        area = object_data['area']
        
        # Calculate shape properties
        height, width = object_data['height'], object_data['width']
        aspect_ratio = width / height if height > 0 else 0
        
        # Calculate perimeter
        perimeter = self._calculate_perimeter(mask)
        
        # Calculate circularity
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Check if rectangular
        expected_area = height * width
        is_rectangular = abs(area - expected_area) <= 2
        
        # Check for holes
        has_holes = self._detect_holes(mask)
        
        # Calculate compactness
        compactness = area / (height * width) if height * width > 0 else 0
        
        # Detect symmetry
        symmetry = self._analyze_symmetry(mask)
        
        return {
            'color': object_data['color'],
            'area': area,
            'perimeter': perimeter,
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'is_rectangular': is_rectangular,
            'has_holes': has_holes,
            'compactness': compactness,
            'symmetry': symmetry,
            'centroid': object_data['centroid'],
            'bbox': bbox,
            'pixels': object_data['pixels']
        }
    
    def _calculate_perimeter(self, mask: np.ndarray) -> int:
        """Calculate perimeter of a binary mask."""
        # Use morphological operations to find perimeter
        from scipy import ndimage
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        eroded = ndimage.binary_erosion(mask, structure=kernel)
        perimeter = np.sum(mask) - np.sum(eroded)
        return int(perimeter)
    
    def _detect_holes(self, mask: np.ndarray) -> bool:
        """Detect if the object has holes."""
        # Fill holes and compare area
        from scipy import ndimage
        filled = ndimage.binary_fill_holes(mask)
        original_area = np.sum(mask)
        filled_area = np.sum(filled)
        return filled_area > original_area
    
    def _analyze_symmetry(self, mask: np.ndarray) -> Dict[str, bool]:
        """Analyze symmetry properties of the object."""
        # Check horizontal symmetry
        horizontal_symmetric = np.array_equal(mask, np.flip(mask, axis=1))
        
        # Check vertical symmetry
        vertical_symmetric = np.array_equal(mask, np.flip(mask, axis=0))
        
        # Check diagonal symmetry
        diagonal_symmetric = np.array_equal(mask, np.rot90(mask, 2))
        
        return {
            'horizontal': bool(horizontal_symmetric),
            'vertical': bool(vertical_symmetric),
            'diagonal': bool(diagonal_symmetric),
            'any_symmetry': bool(horizontal_symmetric or vertical_symmetric or diagonal_symmetric)
        }
    
    def detect_shape_changes(self, input_objects: List[Dict[str, Any]], 
                           output_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect shape changes between input and output objects.
        
        Args:
            input_objects: Objects from input grid
            output_objects: Objects from output grid
            
        Returns:
            List of detected shape changes
        """
        changes = []
        
        # Match objects by color and position
        for in_obj in input_objects:
            in_color = in_obj['color']
            in_centroid = in_obj['centroid']
            
            # Find corresponding output object
            matching_out_objs = [obj for obj in output_objects if obj['color'] == in_color]
            
            if len(matching_out_objs) == 1:
                out_obj = matching_out_objs[0]
                
                # Check for size changes
                if in_obj['area'] != out_obj['area']:
                    scale_factor = out_obj['area'] / in_obj['area']
                    changes.append({
                        'type': 'size_change',
                        'color': in_color,
                        'scale_factor': scale_factor,
                        'confidence': 0.8
                    })
                
                # Check for position changes
                in_centroid = in_obj['centroid']
                out_centroid = out_obj['centroid']
                dx = out_centroid[1] - in_centroid[1]  # x difference
                dy = out_centroid[0] - in_centroid[0]  # y difference
                
                if abs(dx) > 0.1 or abs(dy) > 0.1:
                    changes.append({
                        'type': 'translation',
                        'color': in_color,
                        'dx': dx,
                        'dy': dy,
                        'confidence': 0.7
                    })
                
                # Check for shape changes
                if in_obj['is_rectangular'] != out_obj['is_rectangular']:
                    changes.append({
                        'type': 'shape_change',
                        'color': in_color,
                        'from_rectangular': in_obj['is_rectangular'],
                        'to_rectangular': out_obj['is_rectangular'],
                        'confidence': 0.6
                    })
        
        return changes
    
    def detect_object_relationships(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect relationships between objects.
        
        Args:
            objects: List of objects in the grid
            
        Returns:
            Dictionary with relationship information
        """
        if len(objects) < 2:
            return {'aligned': False, 'grid_pattern': False, 'spacing': None}
        
        # Check for alignment
        centroids = [obj['centroid'] for obj in objects]
        x_coords = [c[1] for c in centroids]
        y_coords = [c[0] for c in centroids]
        
        # Check horizontal alignment
        x_std = np.std(x_coords)
        horizontal_aligned = x_std < 0.5
        
        # Check vertical alignment
        y_std = np.std(y_coords)
        vertical_aligned = y_std < 0.5
        
        # Check for grid pattern
        grid_pattern = self._detect_grid_pattern(objects)
        
        # Calculate spacing
        spacing = self._calculate_spacing(objects)
        
        return {
            'horizontal_aligned': bool(horizontal_aligned),
            'vertical_aligned': bool(vertical_aligned),
            'grid_pattern': grid_pattern,
            'spacing': spacing,
            'num_objects': len(objects)
        }
    
    def _detect_grid_pattern(self, objects: List[Dict[str, Any]]) -> bool:
        """Detect if objects form a grid pattern."""
        if len(objects) < 4:
            return False
        
        # Check if objects are arranged in a grid
        centroids = [obj['centroid'] for obj in objects]
        x_coords = sorted([c[1] for c in centroids])
        y_coords = sorted([c[0] for c in centroids])
        
        # Check for regular spacing
        x_diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
        
        x_regular = len(set(round(d, 1) for d in x_diffs)) <= 2
        y_regular = len(set(round(d, 1) for d in y_diffs)) <= 2
        
        return x_regular and y_regular
    
    def _calculate_spacing(self, objects: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate average spacing between objects."""
        if len(objects) < 2:
            return None
        
        centroids = [obj['centroid'] for obj in objects]
        distances = []
        
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                c1, c2 = centroids[i], centroids[j]
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                distances.append(dist)
        
        return float(np.mean(distances)) if distances else None
    
    def suggest_shape_transformations(self, input_objects: List[Dict[str, Any]], 
                                   output_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Suggest shape transformations based on object analysis.
        
        Args:
            input_objects: Objects from input grid
            output_objects: Objects from output grid
            
        Returns:
            List of suggested shape transformations
        """
        suggestions = []
        
        # Detect shape changes
        changes = self.detect_shape_changes(input_objects, output_objects)
        suggestions.extend(changes)
        
        # Analyze relationships
        input_relationships = self.detect_object_relationships(input_objects)
        output_relationships = self.detect_object_relationships(output_objects)
        
        # Check for alignment changes
        if input_relationships['horizontal_aligned'] != output_relationships['horizontal_aligned']:
            suggestions.append({
                'type': 'alignment_change',
                'direction': 'horizontal',
                'confidence': 0.6
            })
        
        if input_relationships['vertical_aligned'] != output_relationships['vertical_aligned']:
            suggestions.append({
                'type': 'alignment_change',
                'direction': 'vertical',
                'confidence': 0.6
            })
        
        # Check for grid pattern changes
        if input_relationships['grid_pattern'] != output_relationships['grid_pattern']:
            suggestions.append({
                'type': 'grid_pattern_change',
                'confidence': 0.5
            })
        
        return suggestions 