"""Push a local file to a Globus Compute endpoint's filesystem in chunks.

Aurora mounts neither /grand nor our NFS and no Globus Transfer collection is
wired up here, so the only channel to the compute side is the task payload.
This sends a plain Python function (not a ShellFunction -- bash caps a single
argument at 128 KB, which makes base64-in-a-command useless for anything big).
Chunks land in <dest>.part/NNNN and are concatenated once all are present, so a
re-run only resends what is missing.

argv: endpoint_id local_file remote_path [chunk_mb] [parallel]
"""
import hashlib
import sys

from globus_compute_sdk import Executor


def write_chunk(path, n, data):
    import hashlib
    import os
    d = path + ".part"
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{n:04d}")
    if os.path.exists(p) and os.path.getsize(p) == len(data):
        return f"SKIP {n}"
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)
    return f"OK {n} {len(data)} {hashlib.sha256(data).hexdigest()[:8]}"


def join_parts(path, nparts, expect):
    import hashlib
    import os
    d = path + ".part"
    names = sorted(os.listdir(d))
    if len(names) != nparts:
        return f"INCOMPLETE {len(names)}/{nparts}"
    h = hashlib.sha256()
    with open(path, "wb") as out:
        for n in names:
            b = open(os.path.join(d, n), "rb").read()
            out.write(b)
            h.update(b)
    got = h.hexdigest()
    return f"{'VERIFIED' if got == expect else 'MISMATCH'} {got} size={os.path.getsize(path)}"


def main():
    ep, local, remote = sys.argv[1], sys.argv[2], sys.argv[3]
    chunk_mb = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    par = int(sys.argv[5]) if len(sys.argv) > 5 else 4

    raw = open(local, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    step = int(chunk_mb * 1024 * 1024)
    parts = [raw[i:i + step] for i in range(0, len(raw), step)]
    print(f"{len(raw)/1e6:.1f} MB -> {len(parts)} chunks of {chunk_mb} MB")

    with Executor(endpoint_id=ep) as ex:
        pending = []
        for n, p in enumerate(parts):
            pending.append(ex.submit(write_chunk, remote, n, p))
            if len(pending) >= par:
                for f in pending:
                    f.result(timeout=1800)
                print(f"  ..{n + 1}/{len(parts)}", flush=True)
                pending = []
        for f in pending:
            f.result(timeout=1800)
        print(f"  ..{len(parts)}/{len(parts)}", flush=True)
        print(ex.submit(join_parts, remote, len(parts), digest).result(timeout=1800))


if __name__ == "__main__":
    main()
