"""The 34 programming techniques NeoGauge scores against, and how to read them back.

Lifted from ``neocoder_dat/src/utils/configs.py`` (original bundle, kept locally under tasks/archive/ and not shipped with the repo) and
``src/generator/generator.py``. The list is the vocabulary: a technique the
reviewer names that is not on it is dropped, because divergence is measured
against human solutions labelled with these same 34 names.
"""
import re

CONTROL_FLOWS = [
    'if statement', 'for loop', 'while loop', 'break statement',
    'continue statement', 'pass statement', 'match statement', 'recursion',
]
DATA_STRUCTURES = [
    'stack', 'queue', 'tuple', 'set', 'dictionary', 'linked list', 'tree', 'graph',
]
ALGORITHMS = [
    'two pointers', 'sliding window', 'matrix operation', 'hashmap',
    'depth first search', 'width first search', 'back tracking',
    'divide & conquer', 'Kadanes algorithm', 'binary search', 'heap',
    'dynamic programming', 'greedy algorithm', 'misc', 'minimax',
    'topological sort', 'sorting', 'graph traversal',
]
TECHNIQUES = CONTROL_FLOWS + DATA_STRUCTURES + ALGORITHMS

CODE_REVIEWER = (
    "You are a code reviewer. Detect all the programming techniques from the "
    "input and return a list of programming techniques. Only select the "
    "techniques from this list: " + str(TECHNIQUES) +
    "\nYour output should look like this:\n- technique 1\n- technique 2\n"
    "- technique 3\n- ..."
)

# A reviewer that names more than this has almost certainly echoed the prompt's
# own list rather than read the code, so the answer is retried.
MAX_PLAUSIBLE_TECHNIQUES = 10


def parse_techniques(technique_str, default=None):
    """Read the reviewer's bulleted list, keeping only known technique names.

    Two behaviours carried over deliberately, both worth knowing:

    * Anything not in TECHNIQUES is dropped silently. That is the point -- the
      human labels use this vocabulary -- but it means a reviewer that invents
      its own wording scores as having used nothing.
    * The original returned ``['for loop']`` when nothing matched, so a failed
      detection became "this solution used a for loop" and the next denial round
      forbade it. Here the fallback is a parameter and defaults to *empty*: an
      empty list says "we could not tell", which is the truth, and the caller
      decides what that means. Pass ``default=['for loop']`` to reproduce the
      published behaviour.
    """
    if not isinstance(technique_str, str):
        return list(default or [])
    names = [re.sub(r'\s*\(.*?\)', '', s.strip('- ').strip())
             for s in technique_str.split('\n')]
    names = [s.replace('"', '').replace("'", "") for s in names]
    found = [n for n in names if n in TECHNIQUES]
    return found if found else list(default or [])
