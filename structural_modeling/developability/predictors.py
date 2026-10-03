#!/usr/bin/env python3
"""Solubility / developability predictors -- real, published tools, not heuristics.

Developability (solubility, expression, stability) is a first-class reward factor. The
predictions here must be GROUNDED in computation or literature, never in guessed
coefficients. This module provides:

  * :class:`NetSolPPredictor` -- a wrapper around **NetSolP** (Thumuluri et al.,
    *Bioinformatics* 2022, doi:10.1093/bioinformatics/btab801), a protein language-model
    (ESM) predictor of *E. coli* solubility and usability. This is the real, benchmarked
    predictor; it runs the authors' own quantized ONNX models via their ``predict.py``.
  * :func:`compute_descriptors` -- grounded biophysical descriptors computed by
    **Biopython** ``ProtParam`` (published scales: Kyte & Doolittle 1982 GRAVY;
    Guruprasad et al. 1990 instability index), reported as CONTEXT only, never as a
    solubility verdict.

Nothing here invents a solubility score from hand-tuned weights. If NetSolP is not
installed, the predictor reports itself unavailable and the assessment fails closed
(abstains) rather than fabricating a number.
"""
from __future__ import annotations

import csv
import io
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

# NetSolP install (cloned from the DTU-linked canonical repo github.com/tvinet/NetSolP-1.0).
# The ESM-based models run under the boltz-2 env (torch + onnxruntime). Threads are capped
# for shared-node courtesy (NetSolP defaults to all cores).
NETSOLP_DIR = os.environ.get(
    "NETSOLP_DIR", os.path.expanduser("~/opt/NetSolP-1.0/PredictionServer"))
NETSOLP_PYTHON = os.environ.get(
    "NETSOLP_PYTHON", os.path.expanduser("~/miniforge3/envs/boltz-2/bin/python"))
NETSOLP_CITATION = ("Thumuluri et al., Bioinformatics 2022, "
                    "doi:10.1093/bioinformatics/btab801 (NetSolP)")


@dataclass
class SolubilityPrediction:
    """A solubility estimate in [0, 1] (1 = most soluble) with its provenance.

    ``threshold`` and ``higher_is_better`` carry the *predictor's own* decision
    semantics, so the assessment never hardcodes a boundary calibrated to one model.
    A candidate is flagged when it falls on the poor side of ``threshold``. The
    defaults (0.5, higher-is-better) match NetSolP's binary soluble/insoluble label.
    """
    score: float | None
    method: str
    features: dict = field(default_factory=dict)
    citation: str | None = None
    note: str | None = None
    threshold: float = 0.5
    higher_is_better: bool = True

    def is_poor(self) -> bool | None:
        """True if the score is on the poor side of the predictor's own threshold.

        Returns None when there is no usable score (caller must fail closed).
        """
        if self.score is None:
            return None
        return (self.score < self.threshold if self.higher_is_better
                else self.score > self.threshold)

    def to_dict(self) -> dict:
        return {"score": self.score, "method": self.method, "features": self.features,
                "citation": self.citation, "note": self.note,
                "threshold": self.threshold, "higher_is_better": self.higher_is_better}


class SolubilityPredictor:
    """Interface: turn a sequence into a SolubilityPrediction. Subclass to add real ones."""

    name = "abstract"

    def available(self) -> bool:
        return True

    def predict(self, sequence: str) -> SolubilityPrediction:  # pragma: no cover - interface
        raise NotImplementedError


# =================================================================================
# Grounded biophysical descriptors (Biopython) -- CONTEXT, not a verdict
# =================================================================================
def compute_descriptors(sequence: str) -> dict:
    """Published sequence descriptors via Biopython ProtParam. Empty dict if unavailable.

    GRAVY uses the Kyte & Doolittle (1982) hydropathy scale; the instability index uses
    Guruprasad et al. (1990) (>40 conventionally flags an unstable protein). These are
    reported for interpretability alongside a real predictor -- they are NOT combined
    into a solubility score here.
    """
    seq = "".join(c for c in (sequence or "").upper() if c.isalpha())
    if len(seq) < 5:
        return {}
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
    except ImportError:
        return {}
    pa = ProteinAnalysis(seq)
    try:
        return {
            "length": len(seq),
            "gravy_kyte_doolittle": round(pa.gravy(), 4),
            "instability_index_guruprasad": round(pa.instability_index(), 2),
            "aromaticity": round(pa.aromaticity(), 4),
            "isoelectric_point": round(pa.isoelectric_point(), 2),
            "net_charge_pH7": round(pa.charge_at_pH(7.0), 2),
        }
    except (KeyError, ValueError):  # non-standard residues
        return {"length": len(seq)}


# =================================================================================
# NetSolP -- the real, published predictor
# =================================================================================
def parse_netsolp_csv(text: str) -> list[dict]:
    """Parse NetSolP predict.py output CSV into per-sequence prediction records."""
    reader = csv.DictReader(io.StringIO(text))
    out: list[dict] = []
    for row in reader:
        rec: dict = {"sid": row.get("sid")}
        for key in ("predicted_solubility", "predicted_usability"):
            if row.get(key) not in (None, ""):
                rec[key] = float(row[key])
        out.append(rec)
    return out


class NetSolPPredictor(SolubilityPredictor):
    """Real solubility/usability prediction via NetSolP's own ONNX models."""

    name = "netsolp"

    def __init__(self, *, model_type: str = "Distilled", prediction_type: str = "SU",
                 num_threads: int = 4, netsolp_dir: str = NETSOLP_DIR,
                 python_bin: str = NETSOLP_PYTHON, timeout_s: int = 900):
        self.model_type = model_type          # Distilled = best speed/perf trade-off (paper)
        self.prediction_type = prediction_type  # S, U, or SU
        self.num_threads = num_threads        # capped: shared-node courtesy
        self.netsolp_dir = netsolp_dir
        self.python_bin = python_bin
        self.timeout_s = timeout_s

    def _models_dir(self) -> str:
        return os.path.join(self.netsolp_dir, "models")

    def available(self) -> bool:
        """True only if the runtime and the required ONNX model files are present."""
        if not (os.path.exists(self.python_bin)
                and os.path.exists(os.path.join(self.netsolp_dir, "predict.py"))):
            return False
        models = self._models_dir()
        if self.model_type == "Distilled":
            needed = [f"{obj}_ESM1b_distilled_quantized.onnx"
                      for obj in self._objectives()]
        else:
            needed = [f"{obj}_{self.model_type}_{i}_quantized.onnx"
                      for obj in self._objectives() for i in range(5)]
        return all(os.path.exists(os.path.join(models, n)) for n in needed)

    def _objectives(self) -> list[str]:
        objs = []
        if "S" in self.prediction_type:
            objs.append("Solubility")
        if "U" in self.prediction_type:
            objs.append("Usability")
        return objs

    def predict(self, sequence: str) -> SolubilityPrediction:
        seq = "".join(c for c in (sequence or "").upper() if c.isalpha())
        if len(seq) < 5:
            return SolubilityPrediction(None, self.name, note="sequence too short -- abstain")
        if not self.available():
            return SolubilityPrediction(
                None, self.name,
                note=f"NetSolP not installed at {self.netsolp_dir} (models missing) -- abstain")
        with tempfile.TemporaryDirectory() as tmp:
            fasta = os.path.join(tmp, "q.fasta")
            out_csv = os.path.join(tmp, "preds.csv")
            with open(fasta, "w") as fh:
                fh.write(f">query\n{seq}\n")
            cmd = [self.python_bin, "predict.py",
                   "--FASTA_PATH", fasta, "--OUTPUT_PATH", out_csv,
                   "--MODELS_PATH", self._models_dir(),
                   "--MODEL_TYPE", self.model_type,
                   "--PREDICTION_TYPE", self.prediction_type,
                   "--NUM_THREADS", str(self.num_threads)]
            subprocess.run(cmd, cwd=self.netsolp_dir, check=True,
                           capture_output=True, timeout=self.timeout_s)
            with open(out_csv) as fh:
                records = parse_netsolp_csv(fh.read())
        if not records:
            return SolubilityPrediction(None, self.name, note="NetSolP returned no rows")
        rec = records[0]
        return SolubilityPrediction(
            score=rec.get("predicted_solubility"),
            method=f"NetSolP-{self.model_type}",
            features={k: v for k, v in rec.items() if k != "sid"},
            citation=NETSOLP_CITATION,
            note="E. coli solubility from a protein language model")
