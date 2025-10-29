"""
Torrance Test of Creative Writing (TTCW) implementation.
Part of the Creative Writing domain.
"""

import json
from typing import Dict, List, Any
from pathlib import Path

from .base import BaseTask, register_task


@register_task('ttcw')
class TTCWTask(BaseTask):
    """
    Torrance Test of Creative Writing: Write creative stories from plot summaries.
    
    Evaluation dimensions:
    - Quality: Narrative Ending, Coherence, Elaboration
    - Diversity: Emotional Flexibility
    """
    
    domain = 'creative_writing'
    
    def __init__(
        self,
        dataset_path: str = None,
        min_words: int = 2000,
        **kwargs
    ):
        super().__init__(dataset_path, **kwargs)
        self.min_words = min_words
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load TTCW dataset with story plots."""
        if not self.dataset_path or not Path(self.dataset_path).exists():
            raise ValueError(f"TTCW dataset not found at {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        return data
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format TTCW prompt with plot summary."""
        plot = example['plot']
        
        prompt = f"""Write a New Yorker-style story based on the plot below. Make sure it is at least {self.min_words} words.

Plot: {plot}

Story:"""
        
        return prompt
    
    def get_metrics(self) -> Dict[str, List[str]]:
        """TTCW evaluates quality and diversity dimensions."""
        return {
            'quality': [
                'narrative_ending',
                'coherence',
                'elaboration',
                'world_building'
            ],
            'novelty': [],
            'diversity': ['emotional_flexibility']
        }
    
    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
