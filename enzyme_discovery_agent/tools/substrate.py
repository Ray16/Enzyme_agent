"""Substrate property analysis using RDKit.

Computes physicochemical properties that determine how a substrate interacts
with an enzyme binding pocket: molecular volume, logP, TPSA, functional groups,
longest aliphatic chain, and ECFP4 fingerprints.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Optional

try:
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import AllChem, Descriptors
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    Chem = None  # type: ignore
    DataStructs = None  # type: ignore

from enzyme_discovery_agent.tool_registry import tool

logger = logging.getLogger("enzyme_agent")


@dataclass
class SubstrateProperties:
    """Physicochemical properties of a single substrate."""
    name: str
    smiles: str
    molecular_weight: float = 0.0
    molecular_volume: float = 0.0
    molecular_volume_method: str = "unknown"
    logP: float = 0.0
    tpsa: float = 0.0
    num_hba: int = 0
    num_hbd: int = 0
    num_rotatable_bonds: int = 0
    longest_aliphatic_chain: int = 0
    is_aromatic: bool = False
    is_cyclic: bool = False
    functional_groups: list[str] = field(default_factory=list)
    # Comprehensive functional group counts (each tracked separately)
    num_carboxyl: int = 0
    num_primary_amine: int = 0
    num_secondary_amine: int = 0
    num_tertiary_amine: int = 0
    num_amide: int = 0
    num_hydroxyl: int = 0
    num_thiol: int = 0
    num_ketone: int = 0
    num_aldehyde: int = 0
    num_ester: int = 0
    num_ether: int = 0
    num_epoxide: int = 0
    num_phosphate: int = 0
    num_sulfonate: int = 0
    num_nitro: int = 0
    num_nitrile: int = 0
    num_imine: int = 0
    num_guanidinium: int = 0
    num_aromatic_ring: int = 0
    net_charge_ph8: int = 0
    ecfp4: Optional[DataStructs.ExplicitBitVect] = field(default=None, repr=False)
    mol: Optional[Chem.Mol] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict (excluding mol/fp objects)."""
        return {
            "name": self.name,
            "smiles": self.smiles,
            "molecular_weight": round(self.molecular_weight, 2),
            "molecular_volume": round(self.molecular_volume, 1),
            "molecular_volume_method": self.molecular_volume_method,
            "logP": round(self.logP, 2),
            "tpsa": round(self.tpsa, 1),
            "num_hba": self.num_hba,
            "num_hbd": self.num_hbd,
            "num_rotatable_bonds": self.num_rotatable_bonds,
            "longest_aliphatic_chain": self.longest_aliphatic_chain,
            "is_aromatic": self.is_aromatic,
            "is_cyclic": self.is_cyclic,
            "functional_groups": self.functional_groups,
            "num_carboxyl": self.num_carboxyl,
            "num_primary_amine": self.num_primary_amine,
            "num_secondary_amine": self.num_secondary_amine,
            "num_tertiary_amine": self.num_tertiary_amine,
            "num_amide": self.num_amide,
            "num_hydroxyl": self.num_hydroxyl,
            "num_thiol": self.num_thiol,
            "num_ketone": self.num_ketone,
            "num_aldehyde": self.num_aldehyde,
            "num_ester": self.num_ester,
            "num_ether": self.num_ether,
            "num_epoxide": self.num_epoxide,
            "num_phosphate": self.num_phosphate,
            "num_sulfonate": self.num_sulfonate,
            "num_nitro": self.num_nitro,
            "num_nitrile": self.num_nitrile,
            "num_imine": self.num_imine,
            "num_guanidinium": self.num_guanidinium,
            "num_aromatic_ring": self.num_aromatic_ring,
            "net_charge_ph8": self.net_charge_ph8,
        }


# Module-level cache
_analyzer_cache: dict[str, SubstrateProperties] = {}
_invalid_smiles_cache: set[str] = set()
_invalid_smiles_lock = threading.Lock()

# Persistent (cross-run) cache of computed SubstrateProperties, keyed by SMILES.
# The per-member RDKit scoring (ETKDGv3 embed + MMFF + ComputeMolVolume) is the
# CPU bottleneck of the family-survey breadth pass. The same molecules recur
# across families and resumed runs, so reuse their target-independent properties
# while the descriptor implementation and RDKit version remain unchanged.
# A disk hit returns the same values the in-memory cache would. Lives outside
# the repo so the auto-push
# hook never commits it; override or disable with ENZYME_SUBSTRATE_CACHE_DIR
# (set to empty to disable).  A corrupt/partial entry is silently treated as a
# miss and recomputed, never a hard failure.
_SMILES_CACHE_SCHEMA = 1
_CACHE_CONTEXT = {
    "rdkit_version": rdBase.rdkitVersion if _HAS_RDKIT else None,
    "descriptor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
_env_cache_dir = os.environ.get("ENZYME_SUBSTRATE_CACHE_DIR")
_DISK_CACHE_DIR: Optional[Path] = (
    None if _env_cache_dir == ""
    else Path(_env_cache_dir) if _env_cache_dir
    else Path.home() / ".cache" / "enzyme_discovery_agent" / "substrate_props"
)
# Scalar dataclass fields persisted verbatim (everything except the RDKit objects).
_SCALAR_FIELDS = tuple(f.name for f in dataclass_fields(SubstrateProperties)
                       if f.name not in {"ecfp4", "mol"}) if _HAS_RDKIT else ()


def _disk_cache_path(smiles: str) -> Optional[Path]:
    if _DISK_CACHE_DIR is None:
        return None
    identity = json.dumps({"smiles": smiles, "schema": _SMILES_CACHE_SCHEMA,
                           "context": _CACHE_CONTEXT}, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return _DISK_CACHE_DIR / digest[:2] / f"{digest}.json"


def _load_from_disk(smiles: str, name: str) -> Optional[SubstrateProperties]:
    """Return cached properties for ``smiles`` from disk, or None on any miss."""
    path = _disk_cache_path(smiles)
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        if (data.get("schema") != _SMILES_CACHE_SCHEMA or data.get("smiles") != smiles
                or data.get("context") != _CACHE_CONTEXT):
            return None
        kwargs = {k: data["props"][k] for k in _SCALAR_FIELDS if k in data["props"]}
        props = SubstrateProperties(**kwargs)
        fp = DataStructs.ExplicitBitVect(2048)
        fp.FromBase64(data["ecfp4"])
        props.ecfp4 = fp
        props.mol = Chem.Mol(base64.b64decode(data["mol"]))
        # The molecule is shared by SMILES; the display name is per-call.
        props.name = name
        return props
    except Exception:  # noqa: BLE001 - any corruption/version skew -> recompute
        return None


def _store_to_disk(props: SubstrateProperties) -> None:
    path = _disk_cache_path(props.smiles)
    if path is None or props.ecfp4 is None or props.mol is None:
        return
    try:
        payload = {
            "schema": _SMILES_CACHE_SCHEMA,
            "context": _CACHE_CONTEXT,
            "smiles": props.smiles,
            "props": {k: getattr(props, k) for k in _SCALAR_FIELDS},
            "ecfp4": props.ecfp4.ToBase64(),
            "mol": base64.b64encode(props.mol.ToBinary()).decode("ascii"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)  # atomic; concurrent writers of the same key both valid
    except Exception:  # noqa: BLE001 - caching must never break analysis
        logger.debug("substrate disk-cache store failed for %s", props.smiles, exc_info=True)


def _detect_functional_groups(mol: Chem.Mol) -> dict:
    """Detect functional groups from molecular structure using SMARTS.

    Returns a dict with ``groups`` (list of detected group names) and
    individual ``num_*`` counts for every tracked functional group.
    """
    patterns = {
        # --- Nitrogen groups ---
        "carboxyl": "[CX3](=O)[OX2H1,OX1-]",            # -COOH / -COO-
        "primary_amine": "[NX3H2;!$(NC=O);!$([NH3+])]",  # free -NH2
        "protonated_amine": "[NX4H3+]",                  # [NH3+] ionized
        "secondary_amine": "[NX3H1;!$(NC=O)]([#6])[#6]", # R-NH-R'
        "protonated_secondary_amine": "[NX4H2+]([#6])[#6]",  # R-[NH2+]-R'
        "tertiary_amine": "[NX3;!$(NC=O);!$([NX4])]([#6])([#6])[#6]",  # R3N
        "amide": "[NX3;!$([NX4])][CX3](=O)",             # -NH-C(=O)- / -N(R)-C(=O)-
        "imine": "[CX3;!$(C=O)]=[NX2]",                  # C=N
        "nitrile": "[NX1]#[CX2]",                         # C≡N
        "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",   # -NO2
        "guanidinium": "[NX3]C(=[NH2+,NH])[NX3]",        # guanidinium
        # --- Oxygen groups ---
        "hydroxyl": "[OX2H1;!$(OC=O);!$(OP)]",           # -OH (not ester/phosphate)
        "ketone": "[CX3](=O)([#6])[#6]",                 # R-CO-R'
        "aldehyde": "[CX3H1](=O)[#6]",                   # R-CHO
        "ester": "[CX3](=O)[OX2][#6]",                   # R-COO-R'
        "ether": "[OX2]([#6;!$(C=O)])[#6;!$(C=O)]",      # R-O-R' (not ester/acid)
        "epoxide": "[OX2r3]",                             # 3-membered ring O
        # --- Sulfur groups ---
        "thiol": "[SX2H1]",                               # -SH
        "sulfonate": "[SX4](=O)(=O)([OX2,OX1-])",        # -SO3H / -SO3-
        # --- Phosphorus groups ---
        "phosphate": "[PX4](=O)([OX2,OX1-])([OX2,OX1-])[OX2,OX1-]",  # -PO4
        # --- Ring systems ---
        "aromatic_ring": "a1aaaaa1",                      # 6-membered aromatic
    }

    counts = {}
    groups = []
    for name, smarts in patterns.items():
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat)
        counts[name] = len(matches)
        if matches:
            groups.append(name)

    # Combine primary amine counts (neutral + protonated forms)
    total_primary = counts.get("primary_amine", 0) + counts.get("protonated_amine", 0)
    # Combine secondary amine counts (neutral + protonated forms)
    total_secondary = (
        counts.get("secondary_amine", 0) + counts.get("protonated_secondary_amine", 0)
    )

    return {
        "groups": groups,
        "num_carboxyl": counts.get("carboxyl", 0),
        "num_primary_amine": total_primary,
        "num_secondary_amine": total_secondary,
        "num_tertiary_amine": counts.get("tertiary_amine", 0),
        "num_amide": counts.get("amide", 0),
        "num_hydroxyl": counts.get("hydroxyl", 0),
        "num_thiol": counts.get("thiol", 0),
        "num_ketone": counts.get("ketone", 0),
        "num_aldehyde": counts.get("aldehyde", 0),
        "num_ester": counts.get("ester", 0),
        "num_ether": counts.get("ether", 0),
        "num_epoxide": counts.get("epoxide", 0),
        "num_phosphate": counts.get("phosphate", 0),
        "num_sulfonate": counts.get("sulfonate", 0),
        "num_nitro": counts.get("nitro", 0),
        "num_nitrile": counts.get("nitrile", 0),
        "num_imine": counts.get("imine", 0),
        "num_guanidinium": counts.get("guanidinium", 0),
        "num_aromatic_ring": counts.get("aromatic_ring", 0),
    }


def _longest_chain(mol: Chem.Mol) -> int:
    """Find the longest contiguous chain of non-ring heavy atoms."""
    if mol is None:
        return 0

    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)

    adj: dict[int, list[int]] = {}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in ring_atoms and b not in ring_atoms:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

    if not adj:
        return 0

    max_len = 0
    for start in adj:
        visited = {start}
        queue = [(start, 1)]
        while queue:
            node, depth = queue.pop(0)
            max_len = max(max_len, depth)
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))
    return max_len


def compute_substrate_properties(smiles: str, name: str = "") -> SubstrateProperties:
    """Compute and cache substrate properties for a SMILES string."""
    if not _HAS_RDKIT:
        raise RuntimeError("RDKit is not installed — substrate analysis unavailable")
    if smiles in _analyzer_cache:
        return _analyzer_cache[smiles]

    disk_hit = _load_from_disk(smiles, name)
    if disk_hit is not None:
        _analyzer_cache[smiles] = disk_hit
        return disk_hit

    # A single malformed source structure can recur in hundreds of family
    # members. Cache only deterministic parse failures within this process.
    with _invalid_smiles_lock:
        if smiles in _invalid_smiles_cache:
            raise ValueError(f"Invalid SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if len(_invalid_smiles_cache) < 10000:
                _invalid_smiles_cache.add(smiles)
            raise ValueError(f"Invalid SMILES: {smiles}")

    mol_h = Chem.AddHs(mol)
    try:
        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            raise ValueError("unspecified structure contains a dummy atom")
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) != 0:
            raise ValueError("conformer generation failed")
        if AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500) != 0:
            raise ValueError("conformer optimization did not converge")
        vol = AllChem.ComputeMolVolume(mol_h)
        volume_method = "3d_conformer"
    except Exception:
        vol = mol.GetNumHeavyAtoms() * 10.0
        volume_method = "heavy_atom_estimate"
        logger.warning("3D embedding failed for %s; using volume estimate", smiles)

    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    # Neutralize before FG detection so ionization state differences
    # (e.g., -NH2 vs -NH3+, -COOH vs -COO-) don't affect group matching.
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        uncharger = rdMolStandardize.Uncharger()
        neutral_mol = uncharger.uncharge(mol)
    except Exception:
        neutral_mol = mol
    fg = _detect_functional_groups(neutral_mol)

    props = SubstrateProperties(
        name=name,
        smiles=smiles,
        molecular_weight=Descriptors.MolWt(mol),
        molecular_volume=vol,
        molecular_volume_method=volume_method,
        logP=Descriptors.MolLogP(mol),
        tpsa=Descriptors.TPSA(mol),
        num_hba=Descriptors.NumHAcceptors(mol),
        num_hbd=Descriptors.NumHDonors(mol),
        num_rotatable_bonds=Descriptors.NumRotatableBonds(mol),
        longest_aliphatic_chain=_longest_chain(mol),
        is_aromatic=any(atom.GetIsAromatic() for atom in mol.GetAtoms()),
        is_cyclic=mol.GetRingInfo().NumRings() > 0,
        functional_groups=fg["groups"],
        num_carboxyl=fg["num_carboxyl"],
        num_primary_amine=fg["num_primary_amine"],
        num_secondary_amine=fg["num_secondary_amine"],
        num_tertiary_amine=fg["num_tertiary_amine"],
        num_amide=fg["num_amide"],
        num_hydroxyl=fg["num_hydroxyl"],
        num_thiol=fg["num_thiol"],
        num_ketone=fg["num_ketone"],
        num_aldehyde=fg["num_aldehyde"],
        num_ester=fg["num_ester"],
        num_ether=fg["num_ether"],
        num_epoxide=fg["num_epoxide"],
        num_phosphate=fg["num_phosphate"],
        num_sulfonate=fg["num_sulfonate"],
        num_nitro=fg["num_nitro"],
        num_nitrile=fg["num_nitrile"],
        num_imine=fg["num_imine"],
        num_guanidinium=fg["num_guanidinium"],
        num_aromatic_ring=fg["num_aromatic_ring"],
        # At pH 8: primary/secondary/tertiary amines protonated (+1),
        # guanidinium protonated (+1), carboxyls deprotonated (-1),
        # phosphates/sulfonates deprotonated (-1/-2).
        net_charge_ph8=(
            -fg["num_carboxyl"]
            + fg["num_primary_amine"]
            + fg["num_secondary_amine"]
            + fg["num_tertiary_amine"]
            + fg["num_guanidinium"]
            - fg["num_phosphate"] * 2
            - fg["num_sulfonate"]
        ),
        ecfp4=ecfp4,
        mol=mol,
    )

    _analyzer_cache[smiles] = props
    _store_to_disk(props)
    return props


@tool(
    description=(
        "Analyze a substrate's physicochemical properties: molecular weight, volume, "
        "logP, TPSA, functional groups, chain length, and more. Useful for understanding "
        "what an enzyme's binding pocket needs to accommodate."
    ),
    category="substrate",
    parallel_safe=True,
    params={
        "smiles": {
            "type": "string",
            "description": "SMILES string of the substrate to analyze.",
        },
        "name": {
            "type": "string",
            "description": "Optional name for the substrate.",
            "default": "",
        },
    },
)
def analyze_substrate(smiles: str, name: str = "") -> dict:
    """Analyze a substrate's physicochemical properties."""
    try:
        props = compute_substrate_properties(smiles, name)
        return {
            "status": "success",
            "summary": (
                f"Substrate {name or smiles[:30]}: MW={props.molecular_weight:.1f}, "
                f"Vol={props.molecular_volume:.1f}A3, LogP={props.logP:.2f}, "
                f"groups={props.functional_groups}"
            ),
            "data": props.to_dict(),
        }
    except ValueError as e:
        return {
            "status": "error",
            "summary": str(e),
            "data": None,
            "suggestions": ["Check SMILES validity with validate_smiles."],
        }
