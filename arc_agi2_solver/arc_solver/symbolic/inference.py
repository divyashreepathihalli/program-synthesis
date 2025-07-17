"""
Inference engine for ARC-AGI-2 solver.

Logic to match symbolic rules against example pairs and
generate candidate rules from training data.
"""

from typing import List, Dict, Any, Optional, Tuple
from ..parser import Grid, Task
from .rule import Rule, RuleFactory, TransformationRule, CompositeRule
from ..transformations.transform import Transformation, Translate, Rotate, Reflect, Recolor
from ..detectors.color_detector import ColorDetector
from ..detectors.shape_detector import ShapeDetector
from ..detectors.pattern_detector import PatternDetector


class InferenceEngine:
    """Engine for inferring rules from training examples."""
    
    def __init__(self):
        self.color_detector = ColorDetector()
        self.shape_detector = ShapeDetector()
        self.pattern_detector = PatternDetector()
        self.candidate_rules = []
    
    def infer_rules(self, task: Task) -> List[Rule]:
        """
        Infer rules from a task's training pairs.
        
        Args:
            task: Task with training pairs
            
        Returns:
            List of candidate rules
        """
        self.candidate_rules = []
        
        # Analyze training pairs
        train_pairs = task.train_pairs
        if not train_pairs:
            return []
        
        # Generate candidate rules
        self._generate_color_rules(train_pairs)
        self._generate_geometric_rules(train_pairs)
        self._generate_pattern_rules(train_pairs)
        self._generate_composite_rules(train_pairs)
        
        # Test and rank rules
        valid_rules = self._test_rules(train_pairs)
        
        return valid_rules
    
    def _generate_color_rules(self, train_pairs: List[Tuple[Grid, Grid]]):
        """Generate rules based on color analysis."""
        # Analyze color patterns
        color_analysis = self.color_detector.analyze_color_patterns(train_pairs)
        
        # Generate color swap rules
        for from_color, to_color in color_analysis['color_swaps'].items():
            rule = RuleFactory.from_color_mapping({from_color: to_color}, confidence=0.9)
            self.candidate_rules.append(rule)
        
        # Generate color addition rules
        for color in color_analysis['added_colors']:
            # This would need more sophisticated implementation
            pass
        
        # Generate color removal rules
        for color in color_analysis['removed_colors']:
            # This would need more sophisticated implementation
            pass
    
    def _generate_geometric_rules(self, train_pairs: List[Tuple[Grid, Grid]]):
        """Generate rules based on geometric analysis."""
        for input_grid, output_grid in train_pairs:
            # Analyze objects in both grids
            input_objects = self.shape_detector.find_all_objects(input_grid)
            output_objects = self.shape_detector.find_all_objects(output_grid)
            
            # Detect shape changes
            shape_changes = self.shape_detector.detect_shape_changes(input_objects, output_objects)
            
            for change in shape_changes:
                if change['type'] == 'translation':
                    rule = RuleFactory.from_transformation(
                        Translate(dx=int(change['dx']), dy=int(change['dy'])),
                        confidence=change['confidence']
                    )
                    self.candidate_rules.append(rule)
                
                elif change['type'] == 'size_change':
                    # This would need scaling transformation
                    pass
    
    def _generate_pattern_rules(self, train_pairs: List[Tuple[Grid, Grid]]):
        """Generate rules based on pattern analysis."""
        for input_grid, output_grid in train_pairs:
            # Analyze patterns in both grids
            input_patterns = {
                'symmetry': self.pattern_detector.detect_symmetry(input_grid),
                'alignment': self.pattern_detector.detect_alignment_patterns(input_grid),
                'motifs': self.pattern_detector.detect_repeated_motifs(input_grid)
            }
            
            output_patterns = {
                'symmetry': self.pattern_detector.detect_symmetry(output_grid),
                'alignment': self.pattern_detector.detect_alignment_patterns(output_grid),
                'motifs': self.pattern_detector.detect_repeated_motifs(output_grid)
            }
            
            # Generate symmetry rules
            if (input_patterns['symmetry']['any_symmetry'] != 
                output_patterns['symmetry']['any_symmetry']):
                
                # Check for specific symmetry changes
                if (input_patterns['symmetry']['horizontal'] != 
                    output_patterns['symmetry']['horizontal']):
                    rule = RuleFactory.from_transformation(
                        Reflect(axis='horizontal'), confidence=0.7
                    )
                    self.candidate_rules.append(rule)
                
                if (input_patterns['symmetry']['vertical'] != 
                    output_patterns['symmetry']['vertical']):
                    rule = RuleFactory.from_transformation(
                        Reflect(axis='vertical'), confidence=0.7
                    )
                    self.candidate_rules.append(rule)
    
    def _generate_composite_rules(self, train_pairs: List[Tuple[Grid, Grid]]):
        """Generate composite rules by combining simple rules."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated rule composition
        
        # Try combining color and geometric rules
        color_rules = [r for r in self.candidate_rules if isinstance(r, TransformationRule)]
        geometric_rules = [r for r in self.candidate_rules if isinstance(r, TransformationRule)]
        
        # Create some composite rules
        for i, color_rule in enumerate(color_rules[:3]):  # Limit to avoid explosion
            for j, geo_rule in enumerate(geometric_rules[:3]):
                if i != j:
                    composite = RuleFactory.composite([color_rule, geo_rule], confidence=0.6)
                    self.candidate_rules.append(composite)
    
    def _test_rules(self, train_pairs: List[Tuple[Grid, Grid]]) -> List[Rule]:
        """Test candidate rules against training pairs and return valid ones."""
        valid_rules = []
        
        for rule in self.candidate_rules:
            # Test rule on all training pairs
            matches_all = True
            for input_grid, output_grid in train_pairs:
                if not rule.matches(input_grid, output_grid):
                    matches_all = False
                    break
            
            if matches_all:
                valid_rules.append(rule)
        
        # Sort by confidence
        valid_rules.sort(key=lambda r: r.confidence, reverse=True)
        
        return valid_rules
    
    def select_best_rule(self, rules: List[Rule], test_input: Grid) -> Optional[Rule]:
        """
        Select the best rule for a given test input.
        
        Args:
            rules: List of candidate rules
            test_input: Test input grid
            
        Returns:
            Best rule or None if no suitable rule found
        """
        if not rules:
            return None
        
        # For now, return the highest confidence rule
        # In practice, you might want more sophisticated selection
        return rules[0]
    
    def apply_rule_to_test(self, rule: Rule, test_input: Grid) -> Grid:
        """
        Apply a rule to a test input.
        
        Args:
            rule: Rule to apply
            test_input: Test input grid
            
        Returns:
            Transformed grid
        """
        return rule.apply(test_input)


class RuleMatcher:
    """Matches rules against example pairs."""
    
    def __init__(self):
        self.engine = InferenceEngine()
    
    def find_matching_rules(self, train_pairs: List[Tuple[Grid, Grid]], 
                           max_rules: int = 10) -> List[Rule]:
        """
        Find rules that match all training pairs.
        
        Args:
            train_pairs: List of (input, output) pairs
            max_rules: Maximum number of rules to return
            
        Returns:
            List of matching rules
        """
        # Create a temporary task for inference
        temp_task = Task("temp", train_pairs, [])
        
        # Infer rules
        rules = self.engine.infer_rules(temp_task)
        
        # Return top rules
        return rules[:max_rules]
    
    def test_rule_consistency(self, rule: Rule, train_pairs: List[Tuple[Grid, Grid]]) -> float:
        """
        Test how well a rule explains the training pairs.
        
        Args:
            rule: Rule to test
            train_pairs: Training pairs
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if not train_pairs:
            return 0.0
        
        matches = 0
        for input_grid, output_grid in train_pairs:
            if rule.matches(input_grid, output_grid):
                matches += 1
        
        return matches / len(train_pairs)
    
    def rank_rules(self, rules: List[Rule], train_pairs: List[Tuple[Grid, Grid]]) -> List[Rule]:
        """
        Rank rules by their consistency with training pairs.
        
        Args:
            rules: List of rules to rank
            train_pairs: Training pairs
            
        Returns:
            Ranked list of rules
        """
        # Calculate consistency scores
        rule_scores = []
        for rule in rules:
            consistency = self.test_rule_consistency(rule, train_pairs)
            rule_scores.append((rule, consistency))
        
        # Sort by consistency score
        rule_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [rule for rule, score in rule_scores]


def create_simple_inference_engine() -> InferenceEngine:
    """Create a simple inference engine with basic rule generation."""
    return InferenceEngine()


def match_rules_to_task(task: Task) -> List[Rule]:
    """
    Match rules to a task using the inference engine.
    
    Args:
        task: Task to analyze
        
    Returns:
        List of matching rules
    """
    engine = InferenceEngine()
    return engine.infer_rules(task) 