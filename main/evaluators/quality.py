"""
Quality evaluators for CreativityPrism.
"""

from typing import Any, Dict
from .base import BaseEvaluator, register_evaluator


@register_evaluator('aut_score')
class AUTScoreEvaluator(BaseEvaluator):
    """
    Evaluates originality and unconventionality of alternative uses using LLM-as-judge.
    
    Uses a 1-5 Likert scale based on the methodology from Organisciak et al. (2023).
    """
    
    dimension = 'novelty'
    score_range = (1.0, 5.0)
    
    def __init__(self, judge_model: str = 'gpt-4', **kwargs):
        super().__init__(**kwargs)
        self.judge_model = judge_model
        
        # Import model interface here to avoid circular imports
        from models.model_interface import ModelInterface
        self.judge = ModelInterface(judge_model)
    
    def evaluate(
        self,
        response: str,
        input_data: Dict[str, Any],
        task: Any = None
    ) -> float:
        """Evaluate AUT response using LLM-as-judge."""
        object_name = input_data.get('object', 'object')
        
        # Parse uses from response
        if task:
            uses = task.parse_response(response)
        else:
            uses = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Evaluate each use and average
        scores = []
        for use in uses[:10]:  # Evaluate up to 10 uses
            score = self._judge_single_use(object_name, use)
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 1.0
    
    def _judge_single_use(self, object_name: str, use: str) -> float:
        """Judge a single alternative use."""
        prompt = f"""Rate the creativity and originality of the following alternative use for a {object_name} on a scale of 1 to 5:

Alternative use: {use}

Rating criteria:
1 - Common/obvious use, not creative
2 - Somewhat uncommon but still fairly obvious
3 - Moderately creative and original
4 - Very creative and original
5 - Exceptionally creative, highly original and unconventional

Respond with only the number (1-5):"""
        
        try:
            response = self.judge.generate(prompt, temperature=0.0, max_tokens=10)
            # Extract number from response
            score = float(response.strip().split()[0])
            return max(1.0, min(5.0, score))
        except:
            return 3.0  # Default to middle score if parsing fails


@register_evaluator('coherence')
class CoherenceEvaluator(BaseEvaluator):
    """
    Evaluates story coherence using LLM-as-judge.
    """
    
    dimension = 'quality'
    score_range = (0.0, 1.0)
    
    def __init__(self, judge_model: str = 'gpt-4', **kwargs):
        super().__init__(**kwargs)
        self.judge_model = judge_model
        
        from models.model_interface import ModelInterface
        self.judge = ModelInterface(judge_model)
    
    def evaluate(
        self,
        response: str,
        input_data: Dict[str, Any],
        task: Any = None
    ) -> float:
        """Evaluate story coherence."""
        prompt = f"""Evaluate if the following story is coherent, understandable, and flows logically.

Story:
{response}

Is this story coherent and understandable? Respond with only 'yes' or 'no':"""
        
        try:
            judgment = self.judge.generate(prompt, temperature=0.0, max_tokens=10)
            return 1.0 if 'yes' in judgment.lower() else 0.0
        except:
            return 0.0


@register_evaluator('narrative_ending')
class NarrativeEndingEvaluator(BaseEvaluator):
    """
    Evaluates if a story has a proper narrative ending.
    """
    
    dimension = 'quality'
    score_range = (0.0, 1.0)
    
    def __init__(self, judge_model: str = 'gpt-4', **kwargs):
        super().__init__(**kwargs)
        self.judge_model = judge_model
        
        from models.model_interface import ModelInterface
        self.judge = ModelInterface(judge_model)
    
    def evaluate(
        self,
        response: str,
        input_data: Dict[str, Any],
        task: Any = None
    ) -> float:
        """Evaluate narrative ending."""
        prompt = f"""Does the following story have a proper narrative ending (not abrupt or incomplete)?

Story:
{response}

Does this story have a complete narrative ending? Respond with only 'yes' or 'no':"""
        
        try:
            judgment = self.judge.generate(prompt, temperature=0.0, max_tokens=10)
            return 1.0 if 'yes' in judgment.lower() else 0.0
        except:
            return 0.0
