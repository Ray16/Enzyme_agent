"""CPU-only scoring of every Boltz refold, without an AME or design-QC gate.

Run in the Boltz environment (numpy, gemmi, RDKit). Processed molecule pickles
must be trusted local Boltz outputs. No inference, MSA search or minimization.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import re

# Set before importing numerical libraries. This program never allocates a GPU.
for _var in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"
import gemmi
import numpy as np

VERSION = "as9-boltz-refold-scoring-v1"
BB = ("N", "CA", "C", "O")
AA = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))
SIDE = dict(zip(AA, [s.split() for s in (
    "CB", "CB CG CD NE CZ NH1 NH2", "CB CG OD1 ND2", "CB CG OD1 OD2", "CB SG",
    "CB CG CD OE1 NE2", "CB CG CD OE1 OE2", "", "CB CG ND1 CD2 CE1 NE2",
    "CB CG1 CG2 CD1", "CB CG CD1 CD2", "CB CG CD CE NZ", "CB CG SD CE",
    "CB CG CD1 CD2 CE1 CE2 CZ", "CB CG CD", "CB OG", "CB OG1 CG2",
    "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2", "CB CG CD1 CD2 CE1 CE2 CZ OH",
    "CB CG1 CG2")]))
ATP = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"
ACID = "O=C([O-])CCCCC(=O)[O-]"
LABEL = "annotation_src-coordinate-conditioned-region_1_atom"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(exc):
    return f"{type(exc).__name__}: {exc}"


@dataclass
class Atom:
    chain: str
    residue: str
    insertion: str
    name: str
    resname: str
    element: str
    xyz: np.ndarray
    source: str = ""

    @property
    def resid(self):
        return self.residue, self.insertion


def read_atoms(path):
    """Keep author numbering and source labels, and reject ambiguous atoms."""
    path = Path(path)
    if path.suffix == ".pdb":
        st = gemmi.read_structure(str(path))
        if len(st) != 1:
            raise ValueError("Expected one structural model per pose")
        atoms = [Atom(c.name, str(r.seqid.num), r.seqid.icode.strip(), a.name,
                      r.name, a.element.name, np.array([a.pos.x, a.pos.y, a.pos.z]))
                 for c in st[0] for r in c for a in r]
    else:
        if path.suffix == ".zst":
            import zstandard
            with path.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as stream:
                doc = gemmi.cif.read_string(stream.read().decode())
        elif path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                doc = gemmi.cif.read_string(f.read())
        else:
            doc = gemmi.cif.read_file(str(path))
        d = doc.sole_block().get_mmcif_category("_atom_site.")
        n = len(d["id"])
        if len(set(d.get("pdbx_PDB_model_num", ["1"] * n))) != 1:
            raise ValueError("Expected one structural model")
        def col(auth, label):
            return d.get(auth, d[label])
        chains = col("auth_asym_id", "label_asym_id")
        ids = col("auth_seq_id", "label_seq_id")
        names = col("auth_atom_id", "label_atom_id")
        residues = col("auth_comp_id", "label_comp_id")
        insertions = d.get("pdbx_PDB_ins_code", [None] * n)
        labels = d.get(LABEL, [None] * n)
        atoms = [Atom(str(chains[i]), str(ids[i]), insertions[i] or "", names[i],
                      residues[i], d["type_symbol"][i],
                      np.array([float(d[c][i]) for c in ("Cartn_x", "Cartn_y", "Cartn_z")]),
                      labels[i] or "") for i in range(n)]
    keys = [(a.chain, a.resid, a.name) for a in atoms]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate atom identities/alternate conformers; no silent selection")
    return atoms


def protein(atoms, chain):
    residues = {}
    for a in atoms:
        if a.chain == chain:
            residues.setdefault(a.resid, []).append(a)
    if not residues:
        raise ValueError(f"Missing protein chain {chain}")
    result, sequence, locations = {}, [], {}
    for pos, (rid, group) in enumerate(residues.items(), 1):
        names = {a.resname for a in group}
        if len(names) != 1 or next(iter(names)) not in AA:
            raise ValueError(f"Noncanonical/ambiguous protein residue {rid}: {names}")
        sequence.append(AA[next(iter(names))])
        locations[rid] = pos
        for a in group:
            result[pos, a.name] = a
    return result, "".join(sequence), locations


def xyz(atoms):
    x = np.asarray([a.xyz for a in atoms], float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Empty/nonfinite coordinate selection")
    return x


def fit(x, y):
    """Proper-rotation Kabsch transform, row-vector convention."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Fit requires >=3 finite matched atoms")
    xc, yc = x.mean(0), y.mean(0)
    u, s, vt = np.linalg.svd((x - xc).T @ (y - yc))
    if np.linalg.matrix_rank(x - xc) < 2 or np.linalg.matrix_rank(y - yc) < 2:
        raise ValueError("Collinear fit selection")
    sign = np.eye(3)
    sign[-1, -1] = 1 if np.linalg.det(u @ vt) >= 0 else -1
    rotation = u @ sign @ vt
    return rotation, yc - xc @ rotation


def deviation(x, y):
    return float(np.sqrt(np.mean(np.sum((x - y) ** 2, axis=1))))


def symmetry_rmsd(x, y, descriptors, transform=None):
    """Enumerate ASP OD1/OD2 permutations; never exchange ASN O/N."""
    pairs = []
    for pos in sorted({d[0] for d in descriptors}):
        names = {name: i for i, (p, residue, name) in enumerate(descriptors)
                 if p == pos and residue == "ASP"}
        if {"OD1", "OD2"} <= names.keys():
            pairs.append((names["OD1"], names["OD2"]))
    best, best_flags = None, None
    for flags in itertools.product((False, True), repeat=len(pairs)):
        target = np.array(y, copy=True)
        for swap, (i, j) in zip(flags, pairs):
            if swap:
                target[[i, j]] = target[[j, i]]
        rotation, translation = fit(x, target) if transform is None else transform
        value = deviation(x @ rotation + translation, target)
        if best is None or value < best:
            best, best_flags = value, list(flags)
    return {"rmsd_A": best, "atom_count": len(x), "asp_swaps": best_flags,
            "symmetry_permutations_considered": 2 ** len(pairs)}


def recover_motif_resids(atoms, source, definitions, prot, positions, chain):
    """Recover the motif->design residue mapping when RFD4 drops the source labels.

    RFD4 intermittently emits a sample with both provenance columns empty ("Could not
    apply the insertion mapping"); the motif is still conditioned, only its labelling is
    lost. Superpose on the unchanged ligand frame, then solve a one-to-one assignment
    between conditioned source residues and design residues. Validated against the
    labels on samples that carry them: anchor RMSD agrees to within 0.03 A.
    """
    from scipy.optimize import linear_sum_assignment
    pairs = [(a, b) for a in atoms if a.chain != chain
             for b in source if b.chain == a.chain and b.name == a.name]
    if len(pairs) < 3:
        raise ValueError("Too few ligand atoms to recover the motif frame")
    rotation, translation = fit(xyz([a for a, _ in pairs]), xyz([b for _, b in pairs]))
    residues = sorted({a.resid for a in atoms if a.chain == chain})
    keys = list(definitions)
    cost = np.full((len(keys), len(residues)), np.inf)
    for i, rid in enumerate(keys):
        spec = definitions[rid]
        names = sorted(spec["fixed_atoms"])
        native = xyz([next(a for a in source if a.chain == chain
                           and a.residue == rid and a.name == n) for n in names])
        for j, resid in enumerate(residues):
            pos = positions[resid]
            if any((pos, n) not in prot for n in names):
                continue
            if spec.get("sequence_fixed", True) and prot[pos, names[0]].resname != spec["res_name"]:
                continue
            placed = xyz([prot[pos, n] for n in names]) @ rotation + translation
            cost[i, j] = deviation(placed, native)
    if not np.isfinite(cost).any(axis=1).all():
        raise ValueError("No candidate design residue for at least one conditioned motif residue")
    rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
    if not np.isfinite(cost[rows, cols]).all():
        raise ValueError("Could not assign every conditioned motif residue")
    return {keys[i]: residues[j] for i, j in zip(rows, cols)}


def reference(design, motif_input, metadata, chain="A"):
    atoms, source = read_atoms(design), read_atoms(motif_input)
    prot, sequence, positions = protein(atoms, chain)
    definitions = metadata["motif_fixed_atoms"]
    src = {(a.residue, a.name): a for a in source if a.chain == chain}
    mapping, issues = {}, []
    conditioned, sidechains = [], []
    labelled = {rid: [a for a in atoms if a.chain == chain and a.source == f"{chain}{rid}"]
                for rid in definitions}
    mapping_provenance = "source_coordinate_labels"
    recovered = None
    if not any(labelled.values()):
        mapping_provenance = "recovered_by_ligand_frame_superposition"
        recovered = recover_motif_resids(atoms, source, definitions, prot, positions, chain)
    for rid, spec in definitions.items():
        if recovered is None:
            tagged = labelled[rid]
            if {a.name for a in tagged} != set(spec["fixed_atoms"]) or len({a.resid for a in tagged}) != 1:
                raise ValueError(f"Invalid/missing source-coordinate labels for {chain}{rid}")
            resid = tagged[0].resid
        else:
            resid = recovered[rid]
        pos = positions[resid]
        if pos in mapping.values():
            raise ValueError("Two source motif residues map to one design position")
        mapping[rid] = pos
        fixed = spec.get("sequence_fixed", True)
        observed = prot[pos, sorted(spec["fixed_atoms"])[0]].resname
        if fixed and observed != spec["res_name"]:
            issues.append(f"Fixed identity mismatch at source {rid}: {observed}")
        for name in spec["fixed_atoms"]:
            conditioned.append((pos, spec["res_name"], name, src[rid, name], prot[pos, name]))
        if fixed:
            for name in SIDE[spec["res_name"]]:
                sidechains.append((pos, spec["res_name"], name, src[rid, name], prot.get((pos, name))))
    if len(conditioned) != metadata["fixed_protein_atom_count"]:
        raise ValueError("Conditioned atom count disagrees with motif metadata")
    bb = [(pos, name) for pos in range(1, len(sequence) + 1) for name in BB]
    missing_bb = [f"{p}:{n}" for p, n in bb if (p, n) not in prot]
    return dict(protein=prot, sequence=sequence, backbone_keys=bb,
                missing_backbone=missing_bb, conditioned=conditioned, sidechains=sidechains,
                mapping=mapping, issues=issues, mapping_provenance=mapping_provenance,
                donor_is_proline=(sequence[mapping["311"] - 1] == "P") if "311" in mapping else None)


def structural_scores(atoms, ref, chain):
    pred, sequence, _ = protein(atoms, chain)
    if sequence != ref["sequence"]:
        raise ValueError("Refold protein sequence/order differs from generated design")
    result, errors, transform = {}, {}, None
    try:
        if ref["missing_backbone"]:
            raise ValueError("Incomplete design backbone: " + ",".join(ref["missing_backbone"]))
        x = xyz([pred[k] for k in ref["backbone_keys"]])
        y = xyz([ref["protein"][k] for k in ref["backbone_keys"]])
        transform = fit(x, y)
        result["backbone"] = {"rmsd_A": deviation(x @ transform[0] + transform[1], y),
                              "atom_count": len(x), "fit_frame": "All protein N/CA/C/O fit to generated backbone"}
    except (ValueError, KeyError) as exc:
        errors["backbone"] = error(exc)
    for selection in ("conditioned", "sidechains"):
        try:
            rows = ref[selection]
            if selection == "sidechains" and ref["issues"]:
                raise ValueError("Fixed-identity mismatch; sidechain correspondence invalid")
            descriptors = [(p, residue, name) for p, residue, name, _, _ in rows]
            x = xyz([pred[p, name] for p, _, name, _, _ in rows])
            generated = xyz([g for _, _, _, _, g in rows])
            original = xyz([s for _, _, _, s, _ in rows])
            metrics = {}
            for target, y in (("input", original), ("generated", generated)):
                metrics[f"selection_aligned_to_{target}"] = {
                    **symmetry_rmsd(x, y, descriptors),
                    "fit_frame": f"Independent fit of this atom selection to {target}; ASP symmetry resolved"}
            if transform is not None:
                metrics["backbone_aligned_to_generated"] = {
                    **symmetry_rmsd(x, generated, descriptors, transform),
                    "fit_frame": "Global N/CA/C/O fit to generated backbone; no additional site fit"}
            result[selection] = metrics
        except (ValueError, KeyError, AttributeError) as exc:
            errors[selection] = error(exc)
    return result, errors


def molecule_maps(outputs, explicit=None):
    """Derive atom names from actual processed molecules, never fixed PDB names."""
    from rdkit import Chem
    def canonical(m):
        m = Chem.RemoveHs(Chem.Mol(m))
        for a in m.GetAtoms():
            if a.GetSymbol() != "C":
                a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        return Chem.MolToSmiles(m)
    desired = {"ATP": canonical(Chem.MolFromSmiles(ATP)), "adipate": canonical(Chem.MolFromSmiles(ACID))}
    paths = ([Path(explicit)] if explicit and Path(explicit).is_file() else
             sorted(Path(explicit).rglob("*.pkl")) if explicit else
             sorted(Path(outputs).rglob("processed/mols/*.pkl")))
    found = {k: {} for k in desired}
    errors = []
    for path in paths:
        try:
            with path.open("rb") as f:
                data = pickle.load(f)
            values = data.items() if isinstance(data, dict) else [(path.stem, data)]
            for key, mol in values:
                if not isinstance(mol, Chem.Mol):
                    continue
                identity = canonical(mol)
                for kind, expected in desired.items():
                    if identity != expected:
                        continue
                    heavy = Chem.RemoveHs(Chem.Mol(mol))
                    names = {a.GetIdx(): a.GetProp("name") for a in heavy.GetAtoms()}
                    if len(set(names.values())) != len(names):
                        raise ValueError("Duplicate processed molecule atom names")
                    if kind == "ATP":
                        phosphorus = [a for a in heavy.GetAtoms() if a.GetSymbol() == "P"]
                        alpha = [p for p in phosphorus if any(
                            o.GetSymbol() == "O" and any(c.GetSymbol() == "C" for c in o.GetNeighbors())
                            for o in p.GetNeighbors())]
                        if len(alpha) != 1:
                            raise ValueError("ATP P-O-C topology does not identify one P-alpha")
                        bridge = [o for o in alpha[0].GetNeighbors() if o.GetSymbol() == "O" and
                                  sum(n.GetSymbol() == "P" for n in o.GetNeighbors()) == 2]
                        if len(bridge) != 1:
                            raise ValueError("ATP P-O-P topology does not identify one alpha-beta bridge")
                        mapping = {"alpha_phosphorus": names[alpha[0].GetIdx()],
                                   "alpha_beta_bridge_oxygen": names[bridge[0].GetIdx()]}
                    else:
                        oxygens = [a for a in heavy.GetAtoms() if a.GetSymbol() == "O"]
                        if len(oxygens) != 4:
                            raise ValueError("Adipate must have four oxygens")
                        mapping = {"oxygen_names": sorted(names[a.GetIdx()] for a in oxygens)}
                    found[kind].setdefault(json.dumps(mapping, sort_keys=True), {
                        **mapping, "processed_molecule_key": str(key), "source": str(path),
                        "source_sha256": sha(path), "canonical_smiles": identity,
                        "formal_charge": Chem.GetFormalCharge(heavy)})
        except Exception as exc:
            errors.append({"path": str(path), "error": error(exc)})
    if any(len(v) != 1 for v in found.values()):
        raise ValueError(f"Need unambiguous charged ATP/adipate processed maps; counts "
                         f"{ {k: len(v) for k, v in found.items()} }; parse_errors={errors}")
    return {**{k: next(iter(v.values())) for k, v in found.items()}, "parse_errors": errors}


def reaction_geometry(atoms, maps, atp_chain="B", acid_chain="D"):
    def one(chain, name):
        matches = [a for a in atoms if a.chain == chain and a.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one atom {chain}:{name}, found {len(matches)}")
        return matches[0]
    p = one(atp_chain, maps["ATP"]["alpha_phosphorus"])
    bridge = one(atp_chain, maps["ATP"]["alpha_beta_bridge_oxygen"])
    oxygen = [one(acid_chain, name) for name in maps["adipate"]["oxygen_names"]]
    x = xyz([p, bridge] + oxygen)
    candidates = []
    axis = x[1] - x[0]
    if np.linalg.norm(axis) == 0:
        raise ValueError("Zero length P-alpha/bridge vector")
    for atom, coord in zip(oxygen, x[2:]):
        vector = coord - x[0]
        distance = float(np.linalg.norm(vector))
        if distance == 0:
            raise ValueError("Zero attacking O/P-alpha distance")
        angle = float(np.degrees(np.arccos(np.clip(vector @ axis / distance / np.linalg.norm(axis), -1, 1))))
        candidates.append({"atom": atom.name, "distance_A": distance, "angle_deg": angle})
    selected = min(candidates, key=lambda a: (a["distance_A"], a["atom"]))
    return {**selected, "attack_chain": acid_chain, "center_chain": atp_chain,
            "center_atom": p.name, "angle_reference_atom": bridge.name,
            "selection": "Nearest of all four adipate oxygens to P-alpha; lexical name tie-break only",
            "all_oxygen_geometries": candidates}


def aggregate(rows, expected):
    present = [r for r in rows if r.get("path")]
    counts = Counter(r["pose_index"] for r in present)
    duplicate = any(n > 1 for n in counts.values())
    # Wrong/missing sequence identity is not a QC failure: it is not the
    # requested molecular system. Keep its measurements but never rank it as one.
    valid = [r for r in present if r.get("geometry") and r.get("sequence_match", True)]
    complete = (not duplicate and set(counts) == set(range(expected)) and
                len(valid) == expected and all(n == 1 for n in counts.values()))
    # Do not silently overweight duplicated model indices, even with different files.
    values = None
    if valid and not duplicate:
        distances = np.array([r["geometry"]["distance_A"] for r in valid])
        angles = np.array([r["geometry"]["angle_deg"] for r in valid])
        if np.isfinite(distances).all() and np.isfinite(angles).all() and (distances > 0).all():
            values = {"weighted_attack_angle_deg": float(np.sum(angles / distances) / np.sum(1 / distances)),
                      "harmonic_distance_A": float(len(distances) / np.sum(1 / distances)),
                      "pose_count": len(valid)}
    return {"expected_poses": expected, "observed_coordinate_files": len(present),
            "geometry_valid_poses": len(valid), "missing_pose_indices": sorted(set(range(expected)) - counts.keys()),
            "sequence_mismatch_or_unverified_poses": sum(not r.get("sequence_match", True) for r in present),
            "duplicate_pose_indices": sorted(i for i, n in counts.items() if n > 1),
            "unexpected_pose_indices": sorted(counts.keys() - set(range(expected))),
            "complete_geometry_ensemble": complete,
            "all_expected_poses": values if complete else None,
            "available_pose_subset": values,
            "definition": "sum(angle/distance)/sum(1/distance); harmonic distance=n/sum(1/distance). No QC filtering.",
            "confidence_files_present": sum(r.get("confidence") is not None for r in present),
            "fully_scored_poses": sum(r.get("status") == "complete" for r in present)}


def score(design, motif_input, metadata_path, outputs, expected=50, processed_mols=None,
          protein_chain="A", atp_chain="B", acid_chain="D"):
    metadata = json.loads(Path(metadata_path).read_text())
    result = {"scoring_version": VERSION, "design": str(design), "design_sha256": sha(design),
              "motif_input": str(motif_input), "motif_input_sha256": sha(motif_input),
              "metadata": str(metadata_path), "metadata_sha256": sha(metadata_path),
              "outputs": str(outputs), "expected_poses": expected,
              "inclusion_policy": "All generated sequences and all observed refolds, including Pro311 and QC failures",
              "reference_errors": {}, "poses": []}
    ref, maps, design_sequence = None, None, None
    try:
        _, design_sequence, _ = protein(read_atoms(design), protein_chain)
    except Exception as exc:
        result["reference_errors"]["sequence"] = error(exc)
    try:
        ref = reference(design, motif_input, metadata, protein_chain)
        result["reference"] = {"sequence": ref["sequence"], "length": len(ref["sequence"]),
                               "motif_mapping_provenance": ref["mapping_provenance"],
                               "source_residue_to_sequence_position": ref["mapping"],
                               "fixed_identity_issues": ref["issues"], "donor_is_proline": ref["donor_is_proline"],
                               "conditioned_atom_count": len(ref["conditioned"]),
                               "fixed_residue_sidechain_atom_count": len(ref["sidechains"])}
    except Exception as exc:
        result["reference_errors"]["structure"] = error(exc)
    try:
        maps = molecule_maps(outputs, processed_mols)
        result["reaction_atom_maps"] = maps
    except Exception as exc:
        result["reference_errors"]["chemistry"] = error(exc)
    seen, paths = set(), []
    for path in sorted(Path(outputs).rglob("*_model_*.pdb")):
        if path.resolve() not in seen:
            paths.append(path)
            seen.add(path.resolve())
    for path in paths:
        match = re.search(r"_model_(\d+)\.pdb$", path.name)
        if not match:
            continue
        row = {"pose_index": int(match.group(1)), "path": str(path), "sha256": sha(path),
               "errors": {}, "geometry": None, "structural": {}, "confidence": None,
               "sequence_match": False}
        confidence = path.with_name("confidence_" + path.stem + ".json")
        # An ensemble symlink may be outside the actual prediction directory.
        if not confidence.exists():
            confidence = path.resolve().with_name("confidence_" + path.stem + ".json")
        try:
            row["confidence"] = json.loads(confidence.read_text())
            row["confidence_path"] = str(confidence)
        except Exception as exc:
            row["errors"]["confidence"] = error(exc)
        try:
            atoms = read_atoms(path)
        except Exception as exc:
            row["errors"]["structure_parse"] = error(exc)
            atoms = None
        if atoms is not None:
            try:
                _, pose_sequence, _ = protein(atoms, protein_chain)
                row["sequence_match"] = design_sequence is not None and pose_sequence == design_sequence
                if not row["sequence_match"]:
                    raise ValueError("Requested protein sequence mismatch or unverifiable design sequence")
            except Exception as exc:
                row["errors"]["sequence_identity"] = error(exc)
            if ref is not None:
                try:
                    row["structural"], errors = structural_scores(atoms, ref, protein_chain)
                    row["errors"].update(errors)
                except Exception as exc:
                    row["errors"]["structural"] = error(exc)
            else:
                row["errors"]["structural"] = result["reference_errors"]["structure"]
            if maps is not None:
                try:
                    row["geometry"] = reaction_geometry(atoms, maps, atp_chain, acid_chain)
                except Exception as exc:
                    row["errors"]["geometry"] = error(exc)
            else:
                row["errors"]["geometry"] = result["reference_errors"]["chemistry"]
        row["status"] = "complete" if not row["errors"] else "partial" if row["geometry"] or row["structural"] else "failed"
        result["poses"].append(row)
    present = {r["pose_index"] for r in result["poses"]}
    for index in sorted(set(range(expected)) - present):
        result["poses"].append({"pose_index": index, "path": None, "status": "missing",
                                "errors": {"coordinates": "Expected pose not present"}, "geometry": None})
    result["poses"].sort(key=lambda r: (r["pose_index"], r.get("path") or ""))
    result["ensemble"] = aggregate(result["poses"], expected)
    result["status"] = "complete" if (result["ensemble"]["complete_geometry_ensemble"] and
                                          result["ensemble"]["fully_scored_poses"] == expected) else "incomplete"
    result["interpretation"] = ("RMSDs and geometry describe refolded structures, not activity. "
        "Selection-aligned site RMSDs remove rigid site movement; backbone-aligned site RMSDs retain it. "
        "Input full sidechains include unconditioned template atoms. Glycine contributes no sidechain atoms. "
        "Only ASP OD1/OD2 equivalence is resolved. Input donor identities310/311 are free. "
        "Incomplete valid-pose aggregates are explicitly labeled and are not all50-pose rankings.")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design", type=Path, required=True)
    p.add_argument("--motif-input", "--motif", dest="motif_input", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--boltz-outputs", "--boltz-dir", dest="boltz_outputs", type=Path, required=True)
    p.add_argument("--processed-mols", type=Path, help="Optional trusted processed mol pickle/directory")
    p.add_argument("--expected-poses", type=int, default=50)
    p.add_argument("--protein-chain", default="A")
    p.add_argument("--atp-chain", default="B")
    p.add_argument("--acid-chain", default="D")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.expected_poses < 1:
        p.error("--expected-poses must be positive")
    try:
        result = score(a.design, a.motif_input, a.metadata, a.boltz_outputs,
                       a.expected_poses, a.processed_mols, a.protein_chain, a.atp_chain, a.acid_chain)
    except Exception as exc:
        result = {"scoring_version": VERSION, "status": "failed", "error": error(exc)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_name(a.output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(a.output)
    print(json.dumps({"output": str(a.output), "status": result["status"], "ensemble": result.get("ensemble")}))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
