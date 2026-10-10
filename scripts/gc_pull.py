"""Pull files off a Globus Compute endpoint's filesystem (reverse of gc_push.py).

Two modes:

  glob  : gc_pull.py EP '<remote glob>' <local dir>
          each matching file comes back whole, in one task result
  chunk : gc_pull.py EP <remote file> <local file> --chunk [MB] [parallel]
          byte ranges fetched as separate tasks and reassembled, sha256-checked

Use `--chunk` for anything above a few MB: a task result has to fit the payload
limit, so a whole-file fetch fails on large archives.
"""
import hashlib
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


def stat_remote(path):
    import hashlib
    import os
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return os.path.getsize(path), h.hexdigest()


def fetch_range(path, off, n):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(n)


def pull_glob(ep, pattern, dest):
    os.makedirs(dest, exist_ok=True)
    with Executor(endpoint_id=ep) as ex:
        for name, data in ex.submit(fetch, pattern).result(timeout=1800):
            with open(os.path.join(dest, name), "wb") as f:
                f.write(data)
            print(f"{name} {len(data)} bytes")


def pull_chunked(ep, remote, local, chunk_mb, par):
    step = int(chunk_mb * 1024 * 1024)
    with Executor(endpoint_id=ep) as ex:
        size, digest = ex.submit(stat_remote, remote).result(timeout=1800)
        nchunks = (size + step - 1) // step
        print(f"{size/1e6:.1f} MB in {nchunks} chunks, sha256 {digest[:16]}")
        part = local + ".part"
        os.makedirs(part, exist_ok=True)

        todo = [i for i in range(nchunks)
                if not (os.path.exists(f"{part}/{i:04d}")
                        and os.path.getsize(f"{part}/{i:04d}")
                        == min(step, size - i * step))]
        pending = []
        for i in todo:
            pending.append((i, ex.submit(fetch_range, remote, i * step, step)))
            if len(pending) >= par:
                for j, f in pending:
                    open(f"{part}/{j:04d}", "wb").write(f.result(timeout=1800))
                print(f"  ..{i + 1}/{nchunks}", flush=True)
                pending = []
        for j, f in pending:
            open(f"{part}/{j:04d}", "wb").write(f.result(timeout=1800))

    h = hashlib.sha256()
    with open(local, "wb") as out:
        for i in range(nchunks):
            b = open(f"{part}/{i:04d}", "rb").read()
            out.write(b)
            h.update(b)
    got = h.hexdigest()
    print(f"{'VERIFIED' if got == digest else 'MISMATCH'} {got} "
          f"size={os.path.getsize(local)}")
    return got == digest


def main():
    ep, remote, local = sys.argv[1], sys.argv[2], sys.argv[3]
    if "--chunk" in sys.argv:
        k = sys.argv.index("--chunk")
        mb = float(sys.argv[k + 1]) if len(sys.argv) > k + 1 else 4.0
        par = int(sys.argv[k + 2]) if len(sys.argv) > k + 2 else 4
        sys.exit(0 if pull_chunked(ep, remote, local, mb, par) else 1)
    pull_glob(ep, remote, local)


if __name__ == "__main__":
    main()
