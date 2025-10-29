"""
Alternative Uses Task (AUT) implementation.
Part of the Divergent Thinking domain.
"""

import json
from typing import Dict, List, Any
from pathlib import Path

from .base import BaseTask, register_task


@register_task('aut')
class AlternativeUsesTask(BaseTask):
    """
    Alternative Uses Task: Generate creative alternative uses for common objects.
    
    Evaluation dimensions:
    - Novelty: AUT Score (originality and unconventionality of uses)
    """
    
    domain = 'divergent_thinking'
    
    def __init__(self, dataset_path: str = None, num_uses: int = 10, **kwargs):
        super().__init__(dataset_path, **kwargs)
        self.num_uses = num_uses
        
        # Default list of objects if no dataset provided
        self.default_objects = [
            "bottle", "brick", "shoe", "newspaper", "paper clip",
            "tin can", "cardboard box", "rubber band", "wire hanger", "plastic bag"
        ]
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load AUT dataset with objects to find alternative uses for."""
        if self.dataset_path and Path(self.dataset_path).exists():
            with open(self.dataset_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            # Use default objects
            return [{'id': i, 'object': obj} for i, obj in enumerate(self.default_objects)]
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format AUT prompt."""
        object_name = example['object']
        
        prompt = f"""Create a list of {self.num_uses} creative alternative uses for a {object_name}.

Think of unconventional, original, and creative ways to use this object that go beyond its typical purpose. Be specific and practical in your suggestions.

List your {self.num_uses} alternative uses:"""
        
        return prompt
    
    def get_metrics(self) -> Dict[str, List[str]]:
        """AUT evaluates novelty dimension."""
        return {
            'quality': [],
            'novelty': ['aut_score'],
            'diversity': []
        }
    
    def parse_response(self, response: str) -> List[str]:
        """Parse model response into list of alternative uses."""
        uses = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove numbering if present
            if line and any(line[0].isdigit() or line.startswith(('-', '*', '•'))):
                # Strip leading numbering/bullets
                while line and (line[0].isdigit() or line[0] in '.-*•):'):
                    line = line[1:].strip()
                if line:
                    uses.append(line)
        return uses
