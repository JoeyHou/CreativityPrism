"""
Base classes for CreativityPrism evaluators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation metrics."""
    
    def __init__(self, **kwargs):
        """
        Initialize evaluator.
        
        Args:
            **kwargs: Evaluator-specific configuration
        """
        self.config = kwargs
    
    @abstractmethod
    def evaluate(
        self,
        response: str,
        input_data: Dict[str, Any],
        task: Any = None
    ) -> float:
        """
        Evaluate a model response.
        
        Args:
            response: Model's generated response
            input_data: Input example that was used to generate the response
            task: Task object (optional, for task-specific processing)
            
        Returns:
            Numerical score
        """
        pass
    
    def get_dimension(self) -> str:
        """
        Get the creativity dimension this metric evaluates.
        
        Returns:
            One of: 'quality', 'novelty', 'diversity'
        """
        return getattr(self, 'dimension', 'unknown')
    
    def get_range(self) -> tuple:
        """
        Get the min and max possible values for this metric.
        
        Returns:
            Tuple of (min_value, max_value)
        """
        return getattr(self, 'score_range', (0.0, 1.0))


# Evaluator Registry - all evaluators will register themselves here
EVALUATOR_REGISTRY = {}

def register_evaluator(name: str):
    """Decorator to register an evaluator class."""
    def decorator(cls):
        EVALUATOR_REGISTRY[name] = cls
        return cls
    return decorator
