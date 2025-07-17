"""
Main pipeline for ARC-AGI-2 solver.

Orchestrates the complete solving process:
1. Use detectors and analysis modules on example pairs
2. Generate candidate rules or pipelines of transformations
3. Test each candidate on the demo pairs and select those that fit
4. Apply the winning pipeline to the test input(s)
"""

from typing import List, Dict, Any, Optional, Tuple
from .parser import Grid, Task
from .detectors.color_detector import ColorDetector
from .detectors.shape_detector import ShapeDetector
from .detectors.pattern_detector import PatternDetector
from .symbolic.inference import InferenceEngine, RuleMatcher
from .symbolic.rule import Rule, RuleFactory
from .transformations.transform import Transformation, Pipeline


class ARCSolver:
    """Main ARC-AGI-2 solver that orchestrates the complete solving process."""
    
    def __init__(self):
        self.color_detector = ColorDetector()
        self.shape_detector = ShapeDetector()
        self.pattern_detector = PatternDetector()
        self.inference_engine = InferenceEngine()
        self.rule_matcher = RuleMatcher()
        
        # Cache for analysis results
        self.analysis_cache = {}
    
    def solve_task(self, task: Task) -> List[Grid]:
        """
        Solve a complete ARC task.
        
        Args:
            task: Task to solve
            
        Returns:
            List of predicted output grids for test inputs
        """
        # Step 1: Analyze training pairs
        analysis = self._analyze_training_pairs(task.train_pairs)
        
        # Step 2: Generate candidate rules
        candidate_rules = self._generate_candidate_rules(task.train_pairs, analysis)
        
        # Step 3: Test and select best rules
        valid_rules = self._test_and_select_rules(candidate_rules, task.train_pairs)
        
        # Step 4: Apply rules to test inputs
        results = []
        for test_input in task.get_test_inputs():
            predicted_output = self._apply_best_rule(valid_rules, test_input)
            results.append(predicted_output)
        
        return results
    
    def _analyze_training_pairs(self, train_pairs: List[Tuple[Grid, Grid]]) -> Dict[str, Any]:
        """
        Analyze training pairs using all detectors.
        
        Args:
            train_pairs: List of (input, output) pairs
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'color_analysis': [],
            'shape_analysis': [],
            'pattern_analysis': []
        }
        
        for input_grid, output_grid in train_pairs:
            # Color analysis
            color_analysis = self.color_detector.analyze_color_patterns([(input_grid, output_grid)])
            analysis['color_analysis'].append(color_analysis)
            
            # Shape analysis
            input_objects = self.shape_detector.find_all_objects(input_grid)
            output_objects = self.shape_detector.find_all_objects(output_grid)
            shape_changes = self.shape_detector.detect_shape_changes(input_objects, output_objects)
            analysis['shape_analysis'].append({
                'input_objects': input_objects,
                'output_objects': output_objects,
                'shape_changes': shape_changes
            })
            
            # Pattern analysis
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
            analysis['pattern_analysis'].append({
                'input_patterns': input_patterns,
                'output_patterns': output_patterns
            })
        
        return analysis
    
    def _generate_candidate_rules(self, train_pairs: List[Tuple[Grid, Grid]], 
                                analysis: Dict[str, Any]) -> List[Rule]:
        """
        Generate candidate rules from analysis.
        
        Args:
            train_pairs: Training pairs
            analysis: Analysis results
            
        Returns:
            List of candidate rules
        """
        candidate_rules = []
        
        # Generate rules using inference engine
        temp_task = Task("temp", train_pairs, [])
        engine_rules = self.inference_engine.infer_rules(temp_task)
        candidate_rules.extend(engine_rules)
        
        # Generate additional rules based on analysis
        for i, (input_grid, output_grid) in enumerate(train_pairs):
            # Color-based rules
            color_analysis = analysis['color_analysis'][i]
            for from_color, to_color in color_analysis['color_swaps'].items():
                rule = RuleFactory.from_color_mapping({from_color: to_color}, confidence=0.8)
                candidate_rules.append(rule)
            
            # Shape-based rules
            shape_analysis = analysis['shape_analysis'][i]
            for change in shape_analysis['shape_changes']:
                if change['type'] == 'translation':
                    from ..transformations.transform import Translate
                    transformation = Translate(dx=int(change['dx']), dy=int(change['dy']))
                    rule = RuleFactory.from_transformation(transformation, confidence=change['confidence'])
                    candidate_rules.append(rule)
            
            # Pattern-based rules
            pattern_analysis = analysis['pattern_analysis'][i]
            input_patterns = pattern_analysis['input_patterns']
            output_patterns = pattern_analysis['output_patterns']
            
            # Symmetry rules
            if (input_patterns['symmetry']['horizontal'] != output_patterns['symmetry']['horizontal']):
                from ..transformations.transform import Reflect
                transformation = Reflect(axis='horizontal')
                rule = RuleFactory.from_transformation(transformation, confidence=0.7)
                candidate_rules.append(rule)
            
            if (input_patterns['symmetry']['vertical'] != output_patterns['symmetry']['vertical']):
                from ..transformations.transform import Reflect
                transformation = Reflect(axis='vertical')
                rule = RuleFactory.from_transformation(transformation, confidence=0.7)
                candidate_rules.append(rule)
        
        return candidate_rules
    
    def _test_and_select_rules(self, candidate_rules: List[Rule], 
                              train_pairs: List[Tuple[Grid, Grid]]) -> List[Rule]:
        """
        Test candidate rules and select the best ones.
        
        Args:
            candidate_rules: List of candidate rules
            train_pairs: Training pairs for testing
            
        Returns:
            List of valid rules ranked by performance
        """
        # Test each rule against training pairs
        valid_rules = []
        
        for rule in candidate_rules:
            # Test rule on all training pairs
            matches_all = True
            for input_grid, output_grid in train_pairs:
                if not rule.matches(input_grid, output_grid):
                    matches_all = False
                    break
            
            if matches_all:
                valid_rules.append(rule)
        
        # Rank rules by confidence
        valid_rules.sort(key=lambda r: r.confidence, reverse=True)
        
        return valid_rules
    
    def _apply_best_rule(self, rules: List[Rule], test_input: Grid) -> Grid:
        """
        Apply the best rule to a test input.
        
        Args:
            rules: List of valid rules
            test_input: Test input grid
            
        Returns:
            Predicted output grid
        """
        if not rules:
            # If no valid rules found, return original input
            return test_input.copy()
        
        # Apply the highest confidence rule
        best_rule = rules[0]
        return best_rule.apply(test_input)
    
    def solve_single_pair(self, input_grid: Grid, output_grid: Grid) -> Optional[Rule]:
        """
        Solve a single input-output pair to find the rule.
        
        Args:
            input_grid: Input grid
            output_grid: Expected output grid
            
        Returns:
            Rule that explains the transformation, or None
        """
        train_pairs = [(input_grid, output_grid)]
        analysis = self._analyze_training_pairs(train_pairs)
        candidate_rules = self._generate_candidate_rules(train_pairs, analysis)
        valid_rules = self._test_and_select_rules(candidate_rules, train_pairs)
        
        return valid_rules[0] if valid_rules else None
    
    def predict_output(self, train_pairs: List[Tuple[Grid, Grid]], 
                      test_input: Grid) -> Grid:
        """
        Predict output for a test input given training pairs.
        
        Args:
            train_pairs: Training pairs
            test_input: Test input grid
            
        Returns:
            Predicted output grid
        """
        # Create temporary task
        temp_task = Task("temp", train_pairs, [(test_input, Grid([[0]]))])
        
        # Solve the task
        results = self.solve_task(temp_task)
        
        return results[0] if results else test_input.copy()
    
    def get_rule_explanation(self, rule: Rule) -> str:
        """
        Get a human-readable explanation of a rule.
        
        Args:
            rule: Rule to explain
            
        Returns:
            String explanation of the rule
        """
        if hasattr(rule, 'transformation'):
            return f"Apply {rule.transformation} with confidence {rule.confidence}"
        else:
            return f"Apply {rule.__class__.__name__} with confidence {rule.confidence}"


def create_solver() -> ARCSolver:
    """Create a new ARC solver instance."""
    return ARCSolver()


def solve_task_from_json(json_path: str) -> List[Grid]:
    """
    Solve a task from JSON file.
    
    Args:
        json_path: Path to task JSON file
        
    Returns:
        List of predicted output grids
    """
    from .parser import load_task_from_json
    
    # Load task
    task = load_task_from_json(json_path)
    
    # Create solver and solve
    solver = ARCSolver()
    return solver.solve_task(task)


def solve_multiple_tasks(task_paths: List[str]) -> Dict[str, List[Grid]]:
    """
    Solve multiple tasks from JSON files.
    
    Args:
        task_paths: List of paths to task JSON files
        
    Returns:
        Dictionary mapping task IDs to predicted outputs
    """
    results = {}
    
    for task_path in task_paths:
        try:
            outputs = solve_task_from_json(task_path)
            task_id = task_path.split('/')[-1].replace('.json', '')
            results[task_id] = outputs
        except Exception as e:
            print(f"Error solving {task_path}: {e}")
            results[task_path] = []
    
    return results 