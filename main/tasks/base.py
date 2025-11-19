"""
Base classes for CreativityPrism tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path


class BaseTask(ABC):
    """Abstract base class for all CreativityPrism tasks."""
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize task.
        
        Args:
            dataset_path: Path to dataset file
            **kwargs: Additional task-specific configuration
        """
        self.dataset_path = dataset_path
        self.config = kwargs
    
    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load and return the task dataset.
        
        Returns:
            List of examples, where each example is a dictionary
        """
        pass
    
    @abstractmethod
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Format an example into a prompt for the model.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            Formatted prompt string
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, List[str]]:
        """
        Get the evaluation metrics for this task organized by dimension.
        
        Returns:
            Dictionary mapping dimensions to metric names:
            {
                'quality': ['metric1', 'metric2'],
                'novelty': ['metric3'],
                'diversity': ['metric4']
            }
        """
        pass
    
    def preprocess_response(self, response: str) -> str:
        """
        Preprocess model response before evaluation (optional).
        
        Args:
            response: Raw model response
            
        Returns:
            Processed response
        """
        return response.strip()
    
    def get_domain(self) -> str:
        """
        Get the domain this task belongs to.
        
        Returns:
            One of: 'divergent_thinking', 'creative_writing', 'logical_reasoning'
        """
        return getattr(self, 'domain', 'unknown')


# Task Registry - all tasks will register themselves here
TASK_REGISTRY = {}

def register_task(name: str):
    """Decorator to register a task class."""
    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls
    return decorator
