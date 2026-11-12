"""Compatibility entry point; implementation moved under structural modeling."""
import importlib
import runpy
import sys

if __name__ == '__main__':
    runpy.run_module('structural_modeling_agent.barrier.sander_worker', run_name='__main__')
else:
    sys.modules[__name__] = importlib.import_module('structural_modeling_agent.barrier.sander_worker')
