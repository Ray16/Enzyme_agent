"""Evaluation harness for literature exploration.

Scores a completed agent run against a ground-truth benchmark so that any change
to the literature workflow can be judged by its effect on agent behavior
(recall of answer-family papers, breadth, redundancy, tool usage), not just by
unit tests. See ``benchmarks.py`` for the ground-truth campaigns and
``metrics.py`` for ``score_run``.
"""

from enzyme_discovery_agent.eval.benchmarks import BENCHMARKS
from enzyme_discovery_agent.eval.metrics import score_run

__all__ = ["BENCHMARKS", "score_run"]
