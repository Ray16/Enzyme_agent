"""Run the installed authors' NetSolP CLI on CPU without masking/reserving GPUs."""
import runpy
import sys
from pathlib import Path


def main():
    script=Path(sys.argv[1]).resolve()
    args=sys.argv[2:]
    import torch
    # The authors' script otherwise auto-selects CUDA when GPUs are present.
    # This process-local CPU-only override leaves GPU visibility/reservations intact.
    torch.cuda.is_available=lambda:False
    sys.path.insert(0,str(script.parent))
    sys.argv=[str(script),*args]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
