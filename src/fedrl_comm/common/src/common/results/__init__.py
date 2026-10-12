"""
Results schema: what gets written to disk during a training run, and how
results_webapp reads it back.

Built now despite being explicitly deferred earlier -- results_webapp
can't display anything without SOME on-disk format to read. This is the
minimal version of that, not a fuller redesign; extend it if a later need
doesn't fit.
"""

from .reader import RunResult, load_run, load_runs
from .smoothing import trailing_episode_mean
from .writer import ResultsWriter

__all__ = ["ResultsWriter", "RunResult", "load_run", "load_runs", "trailing_episode_mean"]
