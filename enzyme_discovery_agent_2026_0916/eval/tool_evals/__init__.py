"""Isolated per-tool evaluation harness.

Runs a single extraction tool on one curated input and checks its structured output against
known-correct expectations — so a failure points at ONE tool on ONE input, not the confounded
full-agent loop, and costs a handful of LLM calls instead of a whole campaign.

Run:  python -m enzyme_discovery_agent.eval.tool_evals            # all cases (live)
      python -m enzyme_discovery_agent.eval.tool_evals --list
      python -m enzyme_discovery_agent.eval.tool_evals --case L-DEX
      python -m enzyme_discovery_agent.eval.tool_evals --schema-selfcheck   # offline, no network
"""
