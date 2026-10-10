#!/usr/bin/env python
"""Make the stock ESMFold2 autocast blocks device-agnostic, in place.

This is the only model-code change the Aurora port needs. ESMFold2 hard-codes
`device_type="cuda"` in all seven of its autocast blocks and gates two of them
on `device.type == "cuda"`. On an Intel GPU that silently disables bf16
autocast for the whole trunk, and the bf16 ESMC-6B output then meets the fp32
LayerNorm weights of the pair head:

    RuntimeError: expected scalar type BFloat16 but found Float
      ... modeling_esmfold2.py:983   lm_z = self.language_model(...)
      ... modeling_esmfold2_common.py:2170   self.base_z_linear(hidden_states)

Note what the fix is *for*. Stock ESMFold2 on CUDA runs its trunk under bf16
autocast; that is the reference numerics, not an approximation of it. Leaving
autocast off on XPU would not be "more exact", it would be a different
computation from the one the checkpoint was validated under. So the patch
turns autocast on for XPU rather than casting the LM output down to fp32.

A monkeypatch would not do: two sites pass `enabled=False` deliberately, to
keep the Kabsch realignment in fp32, and a wrapper cannot tell those apart
from the ones disabled merely because the device is not CUDA.

Idempotent -- re-running reports `already patched`. Run it after any
`pip install` that reinstalls transformers.

usage: patch_xpu.py [--site-packages DIR] [--check]
"""
import argparse
import os
import sys

HELPER = '''

def _amp_device_type() -> str:
    """Device string for autocast. Upstream hard-codes "cuda"; the equivalent
    accelerator here is "xpu"."""
    try:
        if torch.xpu.is_available():
            return "xpu"
    except AttributeError:
        pass
    return "cuda"
'''

# (file, before, after) -- each `before` must appear exactly once.
EDITS = [
    ("modeling_esmfold2.py",
     'with torch.amp.autocast("cuda", enabled=pair.is_cuda, dtype=torch.bfloat16):',
     'with torch.amp.autocast(_amp_device_type(),\n'
     '                                enabled=pair.device.type in ("cuda", "xpu"),\n'
     '                                dtype=torch.bfloat16):'),
    ("modeling_esmfold2.py",
     'with torch.autocast(device_type="cuda", dtype=torch.bfloat16):',
     'with torch.autocast(device_type=_amp_device_type(), dtype=torch.bfloat16):'),
    ("modeling_esmfold2.py",
     'use_amp = ref_pos.device.type == "cuda"\n'
     '        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):',
     'use_amp = ref_pos.device.type in ("cuda", "xpu")\n'
     '        with torch.amp.autocast(_amp_device_type(), enabled=use_amp,\n'
     '                                dtype=torch.bfloat16):'),
    ("modeling_esmfold2_common.py",
     'with torch.autocast(device_type="cuda", dtype=torch.bfloat16):',
     'with torch.autocast(device_type=_amp_device_type(), dtype=torch.bfloat16):'),
    # Both of these keep enabled=False on purpose (fp32 Kabsch realignment);
    # only the device name changes, so that it is the *live* autocast they
    # suspend rather than an inactive CUDA one.
    ("modeling_esmfold2_common.py",
     'with torch.autocast(device_type="cuda", enabled=False):',
     'with torch.autocast(device_type=_amp_device_type(), enabled=False):'),
    ("modeling_esmfold2_experimental.py",
     'use_amp = ref_pos.device.type == "cuda"',
     'use_amp = ref_pos.device.type in ("cuda", "xpu")'),
    ("modeling_esmfold2_experimental.py",
     'torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16),',
     'torch.amp.autocast(_amp_device_type(), enabled=use_amp, dtype=torch.bfloat16),'),
]

MARKER = "def _amp_device_type()"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-packages", default=None,
                    help="defaults to the transformers package of the running python")
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    args = ap.parse_args()

    if args.site_packages:
        base = os.path.join(args.site_packages, "transformers", "models", "esmfold2")
    else:
        import transformers
        base = os.path.join(os.path.dirname(transformers.__file__),
                            "models", "esmfold2")
    if not os.path.isdir(base):
        sys.exit(f"no esmfold2 model dir at {base}")
    print(f"patching {base}")

    by_file = {}
    for fn, before, after in EDITS:
        by_file.setdefault(fn, []).append((before, after))

    rc = 0
    for fn, edits in by_file.items():
        path = os.path.join(base, fn)
        src = open(path).read()
        if MARKER in src:
            print(f"  {fn}: already patched")
            continue
        new = src
        for before, after in edits:
            n = new.count(before)
            if n == 0:
                print(f"  {fn}: MISSING anchor -- {before.splitlines()[0][:60]}")
                rc = 1
                continue
            new = new.replace(before, after)
            print(f"  {fn}: {n} site(s) <- {before.splitlines()[0][:60]}")
        # Insert the helper after the import block (before the first def/class).
        lines = new.split("\n")
        at = next(i for i, l in enumerate(lines)
                  if l.startswith("def ") or l.startswith("class "))
        new = "\n".join(lines[:at]) + HELPER + "\n" + "\n".join(lines[at:])
        if args.check:
            print(f"  {fn}: would write {len(new)-len(src):+d} bytes")
        else:
            open(path + ".orig", "w").write(src)
            open(path, "w").write(new)
            print(f"  {fn}: written (original kept as {fn}.orig)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
