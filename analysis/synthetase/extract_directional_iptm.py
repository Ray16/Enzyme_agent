#!/usr/bin/env python3
"""Extract directional enzyme/substrate pair-chain ipTM from Boltz confidence files.

For pair_chains_iptm[frame_chain][evaluated_chain]:
  [enzyme][substrate] is substrate-to-enzyme ipTM.
  [substrate][enzyme] is enzyme-to-substrate ipTM.

The script preserves both values because pair-chain ipTM is asymmetric.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS = {
    "boltz2": ROOT / "exp_data/amide_synthetase/boltz2/step1/results/output_configs",
    "decaf": ROOT / "exp_data/amide_synthetase/decaf/step1/results",
}


def parse_dataset(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dataset must be LABEL=PATH") from exc
    return label, Path(path)


def chain_indices(result_dir: Path) -> tuple[str, str, str]:
    records = list((result_dir / "processed/records").glob("*.json"))
    if len(records) != 1:
        raise ValueError(f"expected one processed record under {result_dir}, found {len(records)}")
    record = json.loads(records[0].read_text())
    by_name = {chain["chain_name"]: str(chain["chain_id"]) for chain in record["chains"]}
    return record["id"], by_name["A"], by_name["E"]


def extract_dataset(label: str, root: Path) -> list[dict]:
    rows = []
    confidence_files = sorted(root.glob("**/predictions/*/confidence_*_model_*.json"))
    chain_cache = {}
    for confidence_path in confidence_files:
        result_dir = confidence_path.parents[2]
        if result_dir not in chain_cache:
            chain_cache[result_dir] = chain_indices(result_dir)
        record_id, enzyme_idx, substrate_idx = chain_cache[result_dir]
        target = record_id.removeprefix("step1_")
        confidence = json.loads(confidence_path.read_text())
        matrix = confidence["pair_chains_iptm"]

        substrate_to_enzyme = float(matrix[enzyme_idx][substrate_idx])
        enzyme_to_substrate = float(matrix[substrate_idx][enzyme_idx])
        model = int(confidence_path.stem.rsplit("_", 1)[-1])
        rows.append(
            {
                "dataset": label,
                "target": target,
                "model": model,
                "enzyme_chain": "A",
                "enzyme_chain_index": enzyme_idx,
                "substrate_chain": "E",
                "substrate_chain_index": substrate_idx,
                "substrate_to_enzyme_iptm": substrate_to_enzyme,
                "enzyme_to_substrate_iptm": enzyme_to_substrate,
                "delta_enzyme_minus_substrate": enzyme_to_substrate
                - substrate_to_enzyme,
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["target"]), []).append(row)

    summaries = []
    for (dataset, target), group in sorted(grouped.items()):
        substrate_to_enzyme = [row["substrate_to_enzyme_iptm"] for row in group]
        enzyme_to_substrate = [row["enzyme_to_substrate_iptm"] for row in group]
        summaries.append(
            {
                "dataset": dataset,
                "target": target,
                "n_models": len(group),
                "substrate_to_enzyme_mean": statistics.fmean(substrate_to_enzyme),
                "substrate_to_enzyme_median": statistics.median(substrate_to_enzyme),
                "substrate_to_enzyme_max": max(substrate_to_enzyme),
                "enzyme_to_substrate_mean": statistics.fmean(enzyme_to_substrate),
                "enzyme_to_substrate_median": statistics.median(enzyme_to_substrate),
                "enzyme_to_substrate_max": max(enzyme_to_substrate),
                "mean_delta_enzyme_minus_substrate": statistics.fmean(
                    right - left
                    for left, right in zip(substrate_to_enzyme, enzyme_to_substrate)
                ),
                "fraction_substrate_to_enzyme_lower": statistics.fmean(
                    left < right
                    for left, right in zip(substrate_to_enzyme, enzyme_to_substrate)
                ),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        help="dataset as LABEL=PATH; repeatable (defaults to Boltz-2 and production DeCAF)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "exp_data/amide_synthetase/analysis/directional_iptm",
    )
    args = parser.parse_args()
    datasets = dict(args.dataset) if args.dataset else DEFAULT_DATASETS

    rows = []
    for label, root in datasets.items():
        rows.extend(extract_dataset(label, root))
    summaries = summarize(rows)
    write_csv(args.output_dir / "directional_iptm_models.csv", rows)
    write_csv(args.output_dir / "directional_iptm_summary.csv", summaries)

    for dataset in datasets:
        subset = [row for row in rows if row["dataset"] == dataset]
        if not subset:
            print(f"{dataset}: no completed confidence files")
            continue
        substrate = [row["substrate_to_enzyme_iptm"] for row in subset]
        enzyme = [row["enzyme_to_substrate_iptm"] for row in subset]
        lower = sum(left < right for left, right in zip(substrate, enzyme))
        print(
            f"{dataset}: models={len(subset)} "
            f"targets={len({row['target'] for row in subset})} "
            f"substrate_to_enzyme_mean={statistics.fmean(substrate):.4f} "
            f"enzyme_to_substrate_mean={statistics.fmean(enzyme):.4f} "
            f"substrate_to_enzyme_lower={lower / len(subset):.3f}"
        )


if __name__ == "__main__":
    main()
