"""Adapter: resolved handoff geometry -> ReactiveSite, for the generic proxy library.

This is the seam between the two halves of the structural-modeling pipeline:

    modeling_reasoning   mechanism -> how to COFOLD   (chains, metals, SMILES, step)
                         mechanism -> how to INTERPRET (which atom plays which role)
                                          |
                                          v   this module
    proxies              geometry + statistics over the pose ensemble

The point of routing scoring through the SAME resolved record that built the
Boltz inputs is that the geometry which gets scored is the geometry that was
modelled. When setup and scoring keep separate mechanism tables they drift, and
nothing in the output reveals it.

Atom resolution is done against the actual structure rather than by trusting an
RDKit atom index to line up with PDB atom order (it does not in general):

    ligand role   -> take the ligand chain's HETATM block, read it with RDKit,
                     restore bond orders from the participant's modeling SMILES,
                     re-run the role's SMARTS, take the first matched atom and
                     read its coordinate straight off the conformer
    cofactor role -> look up the CCD atom name in the chain carrying that cofactor

Chains are matched to participants by COMPOSITION (atom count + a successful
bond-order assignment), never by chain order, because chain order differs between
campaigns modelling the same chemistry.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = AllChem = None

from structural_modeling.proxies import Pose, ReactiveSite

ROLE_KEYS = ("nucleophile", "electrophile", "leaving_group", "charge_centre")


class RoleResolutionError(RuntimeError):
    """Raised when the handoff cannot be turned into a scorable geometry."""


def _ligand_block(pose_path: str, chain: str) -> str:
    with open(pose_path) as fh:
        body = "".join(l for l in fh if l.startswith("HETATM") and l[21] == chain)
    return body + "END\n"


def _mol_for_chain(pose_path: str, chain: str, smiles: str):
    """RDKit mol for one ligand chain, with PDB coordinates and correct bond orders."""
    if Chem is None:
        raise RoleResolutionError("RDKit is required to resolve ligand atom roles")
    pm = Chem.MolFromPDBBlock(_ligand_block(pose_path, chain), removeHs=True, sanitize=False)
    tpl = Chem.MolFromSmiles(smiles)
    if pm is None or tpl is None or pm.GetNumAtoms() != tpl.GetNumAtoms():
        return None
    try:
        return AllChem.AssignBondOrdersFromTemplate(tpl, pm)
    except Exception:
        return None


def _match_chain(pose: Pose, pose_path: str, smiles: str) -> tuple[str, Any] | None:
    """Find which ligand chain IS this participant, by composition not by position."""
    if Chem is None:
        return None
    tpl = Chem.MolFromSmiles(smiles)
    if tpl is None:
        return None
    n = tpl.GetNumAtoms()
    for chain, atoms in pose.ligands.items():
        if len(atoms) != n:
            continue
        mol = _mol_for_chain(pose_path, chain, smiles)
        if mol is not None:
            return chain, mol
    return None


def _ligand_atom(pose: Pose, pose_path: str, smiles: str, smarts: str) -> np.ndarray:
    hit = _match_chain(pose, pose_path, smiles)
    if hit is None:
        raise RoleResolutionError(
            f"no ligand chain matches the participant SMILES {smiles!r}")
    _chain, mol = hit
    q = Chem.MolFromSmarts(smarts)
    if q is None:
        raise RoleResolutionError(f"invalid SMARTS {smarts!r}")
    matches = mol.GetSubstructMatches(q)
    if not matches:
        raise RoleResolutionError(f"SMARTS {smarts!r} matched nothing on the participant")
    if len(matches) > 1:
        # Same fail-closed rule the handoff resolver applies: an ambiguous selector
        # means an arbitrary pick, and an arbitrary pick scores a different reaction.
        raise RoleResolutionError(
            f"SMARTS {smarts!r} is ambiguous ({len(matches)} matches); tighten it")
    conf = mol.GetConformer()
    p = conf.GetAtomPosition(matches[0][0])
    return np.array([p.x, p.y, p.z])


def _cofactor_atom(pose: Pose, atom_name: str) -> np.ndarray:
    for _chain, atoms in pose.ligands.items():
        for name, _elem, xyz in atoms:
            if name == atom_name:
                return xyz
    raise RoleResolutionError(f"cofactor atom {atom_name!r} not present in the pose")


def _substrate_smiles(record: dict) -> dict[Any, str]:
    out = {}
    for s in record.get("target_substrates") or []:
        smi = s.get("modeling_smiles")
        if smi:
            out[s.get("component_index")] = smi
            if s.get("role"):
                out[s["role"]] = smi
    return out


def resolver_from_handoff(record: dict) -> Callable[[Pose], ReactiveSite | None]:
    """Build a resolver from a RESOLVED handoff (the record the renderer consumes).

    The returned callable has the signature the proxy library expects, so it drops
    straight into ``check_resolver`` and ``build_ensemble``. Poses must carry a
    ``source_path`` attribute (``parse_pose`` sets it) so the ligand HETATM block
    can be re-read for bond perception.
    """
    geometry = record.get("geometry") or {}
    if not geometry:
        raise RoleResolutionError("handoff has no resolved geometry block; "
                                  "run resolve_handoff first")
    smiles_by = _substrate_smiles(record)
    ideal = float(geometry.get("ideal_angle", 180.0))

    def resolve(pose: Pose) -> ReactiveSite | None:
        path = getattr(pose, "source_path", None)
        coords: dict[str, np.ndarray | None] = {}
        for key in ROLE_KEYS:
            spec = geometry.get(key)
            if not spec:
                coords[key] = None
                continue
            try:
                if spec.get("kind") == "cofactor":
                    coords[key] = _cofactor_atom(pose, spec.get("atom"))
                else:
                    smi = smiles_by.get(spec.get("component_index")) or \
                        smiles_by.get(spec.get("role"))
                    if smi is None or path is None:
                        coords[key] = None
                        continue
                    coords[key] = _ligand_atom(pose, path, smi, spec["smarts"])
            except RoleResolutionError:
                coords[key] = None
        if coords.get("nucleophile") is None or coords.get("electrophile") is None:
            return None
        frame = ()
        elec = coords["electrophile"]
        for chain in pose.ligands:
            nb = pose.bonded(elec, chain, cut=1.8)
            if nb:
                frame = tuple(x for _n, _e, x in nb)
                break
        return ReactiveSite(
            nucleophile=coords["nucleophile"], electrophile=elec,
            leaving=coords.get("leaving_group"), charge_centre=coords.get("charge_centre"),
            frame=frame, ideal_angle=ideal)

    return resolve


def attach_smarts(record: dict, selectors: dict) -> dict:
    """Copy each role's SMARTS string into the geometry block.

    The handoff stores an ``atom_selector`` NAME; scoring needs the pattern. Doing
    the lookup once here keeps ``proxies`` free of any selector vocabulary.
    """
    geometry = record.get("geometry") or {}
    lig = selectors.get("ligand_smarts") or {}
    for key in ROLE_KEYS:
        spec = geometry.get(key)
        if spec and spec.get("kind") != "cofactor":
            name = spec.get("atom_selector")
            if name in lig:
                spec["smarts"] = lig[name]
    return record


def load_resolved(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)
