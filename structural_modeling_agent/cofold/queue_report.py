"""Publish a fixed cofold queue's completed ensemble rankings and coverage."""

import argparse
import csv
import fcntl
import html
import io
import json
from pathlib import Path
import socket
import time

from .queue_worker import write, verify_completion


def inspect_claim(path):
    """Classify a claim without releasing it or assuming its producer has stopped."""
    try:
        owner = json.loads((path / "owner.json").read_text())
        if owner["host"] != socket.gethostname():
            status = "remote_owner_unverified"
        else:
            pid = int(owner["pid"])
            if pid <= 0:
                raise ValueError("invalid process id")
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
            except FileNotFoundError:
                status = "owner_process_absent"
            else:
                current_start = stat.rsplit(")", 1)[1].split()[19]
                status = "owner_process_active" if current_start == str(owner["start_ticks"]) else "owner_pid_reused"
        return {"item": path.name, "status": status, "owner": owner,
                "note": "A missing worker does not establish that its producer or reservation has stopped."}
    except (OSError, ValueError, KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
        return {"item": path.name, "status": "owner_unverified", "reason": str(exc)}


def publish(manifest_path):
    root = manifest_path.resolve().parent
    with (root / "ranking.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest = json.loads(manifest_path.read_text())
        rows, pending, failures, unresolved = [], [], [], []
        for item in manifest["items"]:
            complete = root / "completed" / (item["id"] + ".json")
            if complete.exists():
                try:
                    result = json.loads(complete.read_text())
                    verify_completion(manifest, item, result)
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    pending.append(item["id"])
                    unresolved.append({"id": item["id"], "reason": str(exc), "completion_path": str(complete)})
                    continue
                rows.append({"id": item["id"], "name": item["name"], "evidence_scope": item["priority"],
                             **result["score"], "reactive_phosphate": item["geometry"]["phosphate"],
                             "predictions": result["output"]})
            else:
                pending.append(item["id"])
                failure = root / "failures" / (item["id"] + ".json")
                if failure.exists():
                    failures.append(json.loads(failure.read_text()))
        rows.sort(key=lambda row: (-row["weighted_attack_angle_deg"], row["id"]))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        active = [path.name for path in (root / "claims").glob("*") if path.is_dir()]
        claims = [inspect_claim(root / "claims" / item) for item in sorted(active)]
        report = {"status": "unresolved" if unresolved else "complete" if not pending else "running", "updated_at": time.time(),
                  "metric": manifest["ranking"], "completed": len(rows), "expected": len(manifest["items"]),
                  "active": active, "claim_ownership": claims, "pending": pending, "failures": failures,
                  "ranking_is_provisional": bool(pending), "ranked_candidates": rows,
                  "unresolved_completions": unresolved,
                  "limits": manifest["limits"]}
        write(root / "ranking.json", report)
        buffer = io.StringIO(newline="")
        fields = ["rank", "id", "name", "evidence_scope", "weighted_attack_angle_deg", "harmonic_distance_A", "n_poses", "reactive_phosphate", "predictions"]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        temporary = root / "ranking.csv.tmp"; temporary.write_text(buffer.getvalue()); temporary.replace(root / "ranking.csv")
        body = "".join("<tr>" + "".join("<td>" + html.escape(str(row[k]) if not isinstance(row[k], float) else f"{row[k]:.3f}") + "</td>" for k in fields[:-1]) + "</tr>" for row in rows)
        document = '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Candidate modeling ranking</title><style>body{font:16px system-ui;margin:2rem;max-width:1100px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:.5rem;border-bottom:1px solid #ddd}th{position:sticky;top:0;background:#f5f5f5}p{line-height:1.5}.note{color:#555}</style></head><body>'
        document += f'<h1>Candidate modeling ranking</h1><p>{len(rows)} of {len(manifest["items"])} ensembles complete. ' + ('Provisional ranking; jobs remain.' if pending else 'Full queue completed.') + '</p>'
        document += '<p>Rank: inverse-distance-weighted mean attack angle, in degrees (larger first). Harmonic distance is reported separately. Each ranked ensemble contains the requested complete pose set.</p><table><thead><tr>' + "".join('<th>' + html.escape(k) + '</th>' for k in fields[:-1]) + '</tr></thead><tbody>' + body + '</tbody></table><p class="note">' + html.escape(" ".join(manifest["limits"])) + '</p></body></html>'
        temporary = root / "ranking.html.tmp"; temporary.write_text(document); temporary.replace(root / "ranking.html")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    previous = None
    while True:
        report = publish(args.manifest)
        count = report["completed"]
        if count != previous:
            print(json.dumps({"completed": count, "expected": report["expected"], "status": report["status"]}), flush=True)
            previous = count
        if not args.watch or report["status"] == "complete":
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
