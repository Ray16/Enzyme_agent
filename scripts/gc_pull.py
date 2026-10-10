"""Pull files off a Globus Compute endpoint's filesystem (reverse of gc_push.py).

argv: endpoint_id remote_glob local_dir
Each file comes back as a task result, so keep the total well under the task
payload limit -- this is for structures and logs, not checkpoints.
"""
import os
import sys

from globus_compute_sdk import Executor


def fetch(pattern):
    import glob
    import os
    out = []
    for p in sorted(glob.glob(pattern)):
        if os.path.isfile(p):
            out.append((os.path.basename(p), open(p, "rb").read()))
    return out


def main():
    ep, pattern, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(dest, exist_ok=True)
    with Executor(endpoint_id=ep) as ex:
        for name, data in ex.submit(fetch, pattern).result(timeout=1800):
            with open(os.path.join(dest, name), "wb") as f:
                f.write(data)
            print(f"{name} {len(data)} bytes")


if __name__ == "__main__":
    main()
