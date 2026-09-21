"""Reading a run's outputs back as one table.

`loader` is importable two ways on purpose -- `from result_analysis import loader`
with the repo root on the path, and `import loader` from inside this directory,
which is what a notebook sitting here has to do.
"""
from . import loader  # noqa: F401
