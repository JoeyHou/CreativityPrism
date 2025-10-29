"""
CreativityPrism Tasks Module
"""

from .base import BaseTask, TASK_REGISTRY, register_task
from .aut import AlternativeUsesTask
from .ttcw import TTCWTask

# Import all other task implementations here as you create them
# from .dat import DATTask
# from .ttct import TTCTTask
# from .creative_story import CreativeStoryTask
# from .creativity_index import CreativityIndexTask
# from .cs4 import CS4Task
# from .neocoder import NeoCoderTask
# from .creative_math import CreativeMathTask

__all__ = [
    'BaseTask',
    'TASK_REGISTRY',
    'register_task',
    'AlternativeUsesTask',
    'TTCWTask',
]
