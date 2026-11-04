"""Evidence checks before cofolding and declared physical checks after cofolding.

No assembly, cofactor inventory, reaction atoms or physical tolerances are inferred
here. A campaign supplies reviewed expectations and source snapshots. Passing
checks establishes agreement with those expectations, not catalytic activity or
the truth of a literature interpretation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SETUP_CHOICES = ("assembly", "enzyme_state", "cofactors", "substrates", "reaction_step")
SOURCE_KINDS = {"primary_literature", "experimental_structure", "solution_measurement",
                "curated_annotation"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value):
    return type(value) is int and value > 0


def _check(name, status, reason, **measurements):
    return {"check": name, "status": status, "reason": reason, **measurements}


def _status(checks):
    return "fail" if any(c["status"] == "fail" for c in checks) else (
        "abstain" if not checks or any(c["status"] == "abstain" for c in checks) else "pass")


def _source_checks(evidence, prefix):
    if not isinstance(evidence, list) or not evidence:
        return [_check(prefix, "abstain", "reviewed source evidence is missing")]
    checks = []
    for i, source in enumerate(evidence):
        name = f"{prefix}.{i}"
        if not isinstance(source, dict):
            checks.append(_check(name, "fail", "source record must be an object"))
            continue
        if source.get("kind") not in SOURCE_KINDS or source.get("derived_from_prediction"):
            checks.append(_check(name, "abstain", "a prediction, hypothesis or transfer cannot independently validate its own setup"))
            continue
        if not source.get("source_id") or not source.get("rationale") or source.get("reviewed") is not True:
            checks.append(_check(name, "abstain", "source identity, reviewed interpretation and rationale are required"))
            continue
        path, expected, quote = source.get("path"), source.get("sha256"), source.get("quote")
        if not path or not expected or not isinstance(quote, str) or not quote.strip():
            checks.append(_check(name, "abstain", "a fingerprinted local source and verbatim supporting passage are required"))
            continue
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            checks.append(_check(name, "abstain", f"source unavailable: {exc}"))
            continue
        if hashlib.sha256(raw).hexdigest() != expected:
            checks.append(_check(name, "fail", "source fingerprint changed"))
        elif quote not in raw.decode("utf-8", errors="replace"):
            checks.append(_check(name, "fail", "supporting passage is absent from the source snapshot"))
        else:
            checks.append(_check(name, "pass", "source fingerprint and passage verified; source identity and interpretation are asserted, not independently verified",
                                 source_id=source["source_id"], source_sha256=expected,
                                 source_integrity="verified", attribution="asserted",
                                 interpretation="asserted", independent_review="not_verified"))
    return checks


def _chain_spec(chain):
    return {k: chain[k] for k in ("source", "start", "end") if k in chain}


def _substrate_spec(item):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(item.get("modeling_smiles", ""))
    if molecule is None or not item.get("modeling_smiles"):
        raise ValueError("each substrate needs valid modeling SMILES")
    if Chem.FindMolChiralCenters(molecule, includeUnassigned=True) and any(
            label == "?" for _, label in Chem.FindMolChiralCenters(molecule, includeUnassigned=True)):
        raise ValueError("modeling stereochemistry is unspecified")
    return {"component_index": item["component_index"], "role": item["role"],
            "include_in_model": item.get("include_in_model", True),
            "modeling_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True)}


def validate_setup(record, sequence=None):
    """Recompute setup checks; never trust a saved pass after an input changes.

    ``record`` is a modeling_input. Expectations belong in
    validation_plan.setup.<choice>.{expected,evidence}. Source paths are explicit
    filesystem paths, resolved from the invoking process's working directory.
    """
    checks = []
    plan = record.get("validation_plan") or {}
    if not isinstance(plan, dict) or not isinstance(plan.get("setup", {}), dict):
        raise ValueError("validation_plan and its setup must be objects")
    if plan.get("schema_version") != "1.0":
        checks.append(_check("validation_plan", "abstain", "a schema-1.0 validation plan is required"))
    setup = plan.get("setup") or {}
    assembly = (setup.get("assembly") or {}).get("expected") or {}
    for choice in SETUP_CHOICES:
        declaration = setup.get(choice) or {}
        expected = declaration.get("expected")
        checks.extend(_source_checks(declaration.get("evidence"), f"{choice}.evidence"))
        if not isinstance(expected, dict):
            checks.append(_check(choice, "abstain", "independently reviewed expectations are missing"))
            continue
        try:
            if choice == "assembly":
                counts = expected.get("protomer_chain_counts")
                sites = expected.get("active_sites")
                if not isinstance(counts, list) or not counts or not all(_positive_int(n) for n in counts) or not _positive_int(sites):
                    raise ValueError("declare chain counts per protomer and active-site count separately")
                if expected.get("site_context") != "complete":
                    checks.append(_check(choice, "abstain", "completeness of the modeled ligand site is unresolved"))
                    continue
                observed = len((record.get("enzyme_state") or {}).get("chains") or [])
                checks.append(_check(choice, "pass" if observed == sum(counts) else "fail",
                                     "modeled chains compared with the reviewed functional assembly",
                                     modeled_chains=observed, expected_chains=sum(counts),
                                     protomers=len(counts), active_sites=sites))
            elif choice == "enzyme_state":
                actual = record.get("enzyme_state") or {}
                want = expected.get("chains")
                if not isinstance(want, list) or not want or not expected.get("state"):
                    raise ValueError("declare the state and every chain construct")
                if expected["state"] not in {"intact", "processed"} or any(
                        c.get("source") not in {"full_sequence", "sequence_range"} for c in want):
                    raise ValueError("unsupported enzyme state or chain source")
                same = actual.get("state") == expected["state"] and [
                    _chain_spec(c) for c in actual.get("chains", [])] == [_chain_spec(c) for c in want]
                checks.append(_check(choice, "pass" if same else "fail", "construct/state compared with reviewed chain boundaries"))
                seq_hash = expected.get("precursor_sha256")
                if not seq_hash or sequence is None:
                    checks.append(_check("sequence_identity", "abstain", "full precursor sequence and reviewed sequence fingerprint are required"))
                else:
                    digest = hashlib.sha256(sequence.encode()).hexdigest()
                    checks.append(_check("sequence_identity", "pass" if digest == seq_hash else "fail",
                                         "input sequence compared with reviewed precursor fingerprint", observed_sha256=digest))
                    for chain in actual.get("chains", []):
                        if chain.get("source") == "sequence_range" and not (
                                _positive_int(chain.get("start")) and _positive_int(chain.get("end"))
                                and chain["start"] <= chain["end"] <= len(sequence)):
                            raise ValueError("processed-chain bounds fall outside the precursor")
            elif choice == "cofactors":
                if expected.get("complete_for_step") is not True or not isinstance(expected.get("species"), list):
                    checks.append(_check(choice, "abstain", "a complete inventory, including an explicit empty inventory, is required for this step"))
                    continue
                wanted, observed = Counter(), Counter()
                for item in expected["species"]:
                    ccd = item.get("ccd")
                    if not ccd or not item.get("role"):
                        raise ValueError("cofactor identity and role must be explicit")
                    if ("count_per_active_site" in item) == ("count_per_model" in item):
                        raise ValueError("declare exactly one cofactor counting basis")
                    if "count_per_active_site" in item:
                        count = item["count_per_active_site"]
                        if not _positive_int(assembly.get("active_sites")):
                            raise ValueError("per-site counts require the independently declared active-site count")
                        count = count * assembly["active_sites"] if type(count) in (int, float) else None
                    else:
                        count = item.get("count_per_model")
                    if type(count) not in (int, float) or not math.isfinite(count) or count <= 0 or int(count) != count:
                        raise ValueError("cofactor count must resolve to a positive integer per modeled system")
                    wanted[(ccd, item["role"])] += int(count)
                for item in record.get("cofactors") or []:
                    if not item.get("ccd") or not item.get("role") or not _positive_int(item.get("count")):
                        raise ValueError("modeled cofactor identity, role or count is missing")
                    observed[(item["ccd"], item["role"])] += item["count"]
                checks.append(_check(choice, "pass" if wanted == observed else "fail",
                                     "complete cofactor identities, roles and stoichiometry compared for the declared step",
                                     expected_inventory={f"{k[0]}:{k[1]}": v for k, v in wanted.items()},
                                     observed_inventory={f"{k[0]}:{k[1]}": v for k, v in observed.items()}))
            elif choice == "substrates":
                want = expected.get("components")
                if not isinstance(want, list) or not want:
                    raise ValueError("declare every substrate component, its role and modeled chemical state")
                wanted = [_substrate_spec(c) for c in want]
                actual = [_substrate_spec(c) for c in record.get("target_substrates") or []]
                checks.append(_check(choice, "pass" if actual == wanted else "fail",
                                     "molecular identities, roles, inclusion and charge/stereochemical states compared"))
            else:
                actual = record.get("modeled_step") or {}
                if not _positive_int(expected.get("index")) or not expected.get("mechanism"):
                    raise ValueError("declare the reaction-step index and mechanism")
                same = all(actual.get(k) == expected[k] for k in ("index", "mechanism"))
                geometry_step = (record.get("geometry") or {}).get("step")
                same = same and (geometry_step is None or geometry_step == expected["index"])
                checks.append(_check(choice, "pass" if same else "fail", "modeled mechanism and step compared with reviewed reaction coordinate"))
        except (KeyError, TypeError, ValueError) as exc:
            checks.append(_check(choice, "fail", str(exc)))
    return {"schema_version": "1.0", "status": _status(checks), "checks": checks,
            "record_sha256": fingerprint(record),
            "validator_sha256": file_digest(__file__),
            "note": "Checks verify source integrity and consistency with reviewed expectations; they do not establish source entailment or experimental activity."}


def entity_map_for(chains, cofactors, substrates):
    """Mirror the renderer's chain allocation so prediction identity is checkable."""
    from rdkit import Chem
    import string
    ids = iter(string.ascii_uppercase)
    entities = []
    for chain in chains:
        entities.append({"chain": next(ids), "kind": "protein", "sequence": chain["sequence"],
                         "sequence_sha256": hashlib.sha256(chain["sequence"].encode()).hexdigest()})
    for item in cofactors:
        for _ in range(item["count"]):
            entities.append({"chain": next(ids), "kind": "cofactor", "ccd": item["ccd"], "role": item.get("role")})
    for item in substrates:
        mol = Chem.MolFromSmiles(item["modeling_smiles"])
        if mol is None:
            raise ValueError("invalid substrate SMILES")
        entities.append({"chain": next(ids), "kind": "substrate", "component_index": item.get("component_index"),
                         "elements": dict(Counter(a.GetSymbol().upper() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)),
                         "modeling_smiles": item["modeling_smiles"]})
    return entities


def validate_msa(path, sequence):
    """Check the query of a supplied Boltz CSV or FASTA/A3M MSA."""
    import csv
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    if text.startswith(">"):
        lines = text.splitlines()[1:]
        query = ""
        for line in lines:
            if line.startswith(">"):
                break
            query += line.strip()
    else:
        rows = csv.DictReader(text.splitlines())
        first = next(rows, None)
        query = (first or {}).get("sequence", "")
    # Lowercase A3M letters are insertion columns; gaps are not residues.
    query = "".join(c for c in query if c.isupper())
    if not query or query != sequence:
        raise ValueError("MSA query does not match the modeled chain sequence")
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "query_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()}


# Nonterminal heavy atoms of the 20 standard protein residues. OXT is optional;
# hydrogen naming/protonation is outside this completeness check.
_SIDECHAINS = {
    "ALA": "CB", "ARG": "CB CG CD NE CZ NH1 NH2", "ASN": "CB CG OD1 ND2",
    "ASP": "CB CG OD1 OD2", "CYS": "CB SG", "GLN": "CB CG CD OE1 NE2",
    "GLU": "CB CG CD OE1 OE2", "GLY": "", "HIS": "CB CG ND1 CD2 CE1 NE2",
    "ILE": "CB CG1 CG2 CD1", "LEU": "CB CG CD1 CD2", "LYS": "CB CG CD CE NZ",
    "MET": "CB CG SD CE", "PHE": "CB CG CD1 CD2 CE1 CE2 CZ", "PRO": "CB CG CD",
    "SER": "CB OG", "THR": "CB OG1 CG2", "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH", "VAL": "CB CG1 CG2"}


def _ref_key(ref):
    if not isinstance(ref, dict) or any(not str(ref.get(k, "")).strip() for k in ("chain", "residue", "atom")):
        raise ValueError("atom references require chain, residue and atom")
    return (str(ref["chain"]), str(ref["residue"]), str(ref["atom"]))


def validate_pose_plan(record, entities=None):
    """Check acceptance declarations before compute, independently of predictions."""
    checks = []
    try:
        plan = record["validation_plan"]["poses"]
        if not isinstance(plan, dict):
            raise ValueError("poses must be an object")
        expected, minimum = plan.get("expected_predictions"), plan.get("minimum_passing_predictions")
        if not _positive_int(expected) or not _positive_int(minimum) or minimum > expected:
            raise ValueError("declare positive expected_predictions and minimum_passing_predictions <= expected")
        checks.extend(_source_checks(plan.get("evidence"), "pose_plan.evidence"))
        if entities is None:
            chains = [{"sequence": ""} for _ in record["enzyme_state"]["chains"]]
            entities = entity_map_for(chains, record.get("cofactors", []),
                                      [s for s in record["target_substrates"] if s.get("include_in_model", True)])
        chain_ids = {e["chain"] for e in entities}
        proteins = {e["chain"] for e in entities if e["kind"] == "protein"}
        substrates = {e["chain"] for e in entities if e["kind"] == "substrate"}
        cofactors = {e["chain"] for e in entities if e["kind"] == "cofactor"}
        def ref(value):
            key = _ref_key(value)
            if key[0] not in chain_ids:
                raise ValueError("atom reference names an unrendered chain")
            return key
        sites = plan.get("sites")
        count = record["validation_plan"]["setup"]["assembly"]["expected"]["active_sites"]
        if not isinstance(sites, list) or not _positive_int(count) or len(sites) != count:
            raise ValueError("declare one physical geometry for every expected active site")
        ids, pairs, geometries = set(), set(), set()
        for site in sites:
            sid = site.get("id")
            if not isinstance(sid, str) or not sid.strip() or sid in ids:
                raise ValueError("active site IDs must be nonempty and distinct")
            ids.add(sid)
            keys = [ref(site[k]) for k in ("nucleophile", "electrophile", "angle_reference")]
            alternatives=site.get('nucleophile_candidates')
            if alternatives is not None:
                candidates=[ref(candidate) for candidate in alternatives]
                if (not candidates or len(set(candidates))!=len(candidates) or keys[0] not in candidates
                        or any(key[0]!=keys[0][0] for key in candidates)
                        or any(key in keys[1:] for key in candidates)):
                    raise ValueError('Candidate nucleophiles must be distinct declared atoms of the same participant, including its reference atom')
            pair, geometry = tuple(sorted(keys[:2])), tuple(sorted(keys))
            if len(set(keys)) != 3 or pair in pairs or geometry in geometries:
                raise ValueError("duplicate physical catalytic site or non-distinct reactive atom assignments")
            pairs.add(pair); geometries.add(geometry)
            if site.get("substrate_chain") not in substrates or site["substrate_chain"] not in {k[0] for k in keys}:
                raise ValueError("each site must identify its participating substrate chain")
            _range(site.get("distance_range_A"), "attack distance")
            if _range(site.get("angle_range_deg"), "attack angle")[1] > 180:
                raise ValueError("angle bounds must lie in [0,180]")
        contacted = set()
        for group in plan.get("distance_checks", []):
            a, b = ref(group["a"]), ref(group["b"])
            if a == b:
                raise ValueError("distance endpoints must differ")
            _range(group.get("range_A"), "contact distance")
            if a[0] != b[0]:
                contacted.update((a[0], b[0]))
        if cofactors - contacted:
            raise ValueError("every cofactor requires a declared environmental contact")
        import gemmi
        metals = {e["chain"] for e in entities if e["kind"] == "cofactor" and
                  (e.get("role") in {"metal", "catalytic_metal"} or
                   (gemmi.Element(e["ccd"]).name.upper() == e["ccd"].upper() and gemmi.Element(e["ccd"]).is_metal))}
        coordinated = set()
        for group in plan.get("coordination_checks", []):
            center = ref(group["center"])
            donors = [ref(d) for d in group["donors"]]
            if center[0] not in metals or not donors or len(set(donors)) != len(donors) or center in donors:
                raise ValueError("declare a metal center and distinct donors")
            if not isinstance(group.get("element"), str) or not group["element"]:
                raise ValueError("declare coordination element")
            _range(group.get("distance_range_A"), "coordination distance")
            counts = _range(group.get("donor_count_range"), "donor count")
            if any(type(v) is not int for v in counts) or counts[1] > len(donors):
                raise ValueError("donor count bounds must be integers within the declared donor inventory")
            coordinated.add(center[0])
        if metals - coordinated:
            raise ValueError("every metal requires declared coordination criteria")
        edges = {p: {p} for p in proteins}
        for interface in plan.get("interfaces", []):
            a, b = interface["chains"]
            radius = interface.get("contact_radius_A")
            if a == b or a not in proteins or b not in proteins or type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0 or not _positive_int(interface.get("minimum_residue_pairs")):
                raise ValueError("invalid protein interface declaration")
            edges[a].add(b); edges[b].add(a)
        reached, frontier = set(), {next(iter(proteins))} if proteins else set()
        while frontier:
            reached.update(frontier)
            frontier = set().union(*(edges[p] for p in frontier)) - reached
        if not proteins or reached != proteins:
            raise ValueError("declared interfaces must connect every protein chain")
        environment = plan["ligand_environment"]
        for key in ("contact_radius_A", "clash_distance_A"):
            value = environment[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("ligand distance criteria must be finite and positive")
        if not _positive_int(environment.get("minimum_contact_atoms")) or type(environment.get("maximum_clashing_atoms")) is not int or environment["maximum_clashing_atoms"] < 0:
            raise ValueError("declare contact and clash atom counts")
        policy = plan.get("protein_atom_policy") or {}
        if not isinstance(policy, dict):
            raise ValueError("protein_atom_policy must be an object")
        for omission in policy.get("allowed_missing", []):
            if ref(omission)[0] not in proteins or not omission.get("rationale"):
                raise ValueError("protein omissions need an exact protein atom and rationale")
            checks.extend(_source_checks(omission.get("evidence"), "pose_plan.protein_omission"))
        names = set()
        for modification in policy.get("modified_residues", []):
            name, heavy = modification.get("resname"), modification.get("heavy_atoms")
            if (not isinstance(name, str) or not name or name in names or name in _SIDECHAINS or
                    not isinstance(heavy, list) or not heavy or any(not isinstance(a, str) or not a for a in heavy) or
                    len(set(heavy)) != len(heavy) or not {"N", "CA", "C", "O"}.issubset(heavy) or not modification.get("rationale")):
                raise ValueError("modified residue rules need a distinct nonstandard name, heavy-atom inventory and rationale")
            names.add(name)
            checks.extend(_source_checks(modification.get("evidence"), "pose_plan.protein_modification"))
        checks.append(_check("pose_plan", "pass", "complete physical acceptance criteria declared before prediction"))
    except (KeyError, ValueError, TypeError, RuntimeError) as exc:
        checks.append(_check("pose_plan", "abstain", str(exc)))
    return {"status": _status(checks), "checks": checks, "record_sha256": fingerprint(record)}


def _protein_completeness(atoms, chain, policy):
    residues = {}
    for key, atom in atoms.items():
        if key[0] == chain and atom["protein"]:
            residues.setdefault((key[1], atom["resname"]), set()).add(key[2])
    checks = []
    for (number, name), present in residues.items():
        modifications = [m for m in policy.get("modified_residues", []) if m.get("resname") == name]
        if name in _SIDECHAINS:
            expected = set(("N CA C O " + _SIDECHAINS[name]).split())
        elif len(modifications) == 1 and isinstance(modifications[0].get("heavy_atoms"), list) and modifications[0]["heavy_atoms"]:
            modification = modifications[0]
            checks.extend(_source_checks(modification.get("evidence"), f"protein_modification.{chain}.{number}"))
            expected = set(modification["heavy_atoms"])
        else:
            checks.append(_check(f"protein_completeness.{chain}.{number}", "abstain", f"heavy-atom inventory for modified residue {name} is undeclared"))
            continue
        missing = expected - present
        wrong_elements = []
        if name in _SIDECHAINS:
            wrong_elements = [atom for atom in expected & present
                              if atoms[(chain, number, atom)]["element"] != atom[0]]
        declared_missing = []
        for exception in policy.get("allowed_missing", []):
            if (exception.get("chain"), str(exception.get("residue"))) == (chain, number) and exception.get("atom") in missing:
                evidence = _source_checks(exception.get("evidence"), f"protein_omission.{chain}.{number}")
                checks.extend(evidence)
                declared_missing.append(exception["atom"])
        checks.append(_check(f"protein_completeness.{chain}.{number}", "fail" if wrong_elements else "abstain" if missing else "pass",
                             "required residue heavy atoms compared with observed inventory", residue_name=name,
                             missing_atoms=sorted(missing), declared_missing_atoms=sorted(declared_missing),
                             wrong_element_atoms=sorted(wrong_elements)))
    return checks


def _read_structure(path):
    import gemmi
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError("one prediction model per structure file is required")
    atoms, sequences = {}, {}
    chain_names = set()
    for chain in structure[0]:
        if chain.name in chain_names:
            raise ValueError("duplicate chain identity can hide molecular inventory")
        chain_names.add(chain.name)
        letters = []
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info.is_amino_acid():
                letters.append(info.one_letter_code.upper())
            number = str(residue.seqid.num) + residue.seqid.icode.strip()
            for atom in residue:
                if atom.altloc not in ("\x00", " ", "A") or atom.element.is_hydrogen:
                    continue
                key = (chain.name, number, atom.name)
                xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                if key in atoms or not np.isfinite(xyz).all():
                    raise ValueError("duplicate atom identity or nonfinite coordinates")
                atoms[key] = {"xyz": xyz, "element": atom.element.name.upper(), "resname": residue.name,
                              "protein": info.is_amino_acid()}
        if letters:
            sequences[chain.name] = "".join(letters)
    if not atoms:
        raise ValueError("prediction contains no heavy atoms")
    return atoms, sequences


def _atom(atoms, ref):
    key = (ref["chain"], str(ref["residue"]), ref["atom"])
    if key not in atoms:
        raise LookupError(f"declared atom {key} is absent")
    if ref.get("element") and atoms[key]["element"] != ref["element"].upper():
        raise ValueError(f"declared element does not match atom {key}")
    return atoms[key]


def _geometry_refs(atoms,site):
    """Resolve a predeclared nearest-nucleophile rule without changing its candidates."""
    refs=[site[key] for key in ('nucleophile','electrophile','angle_reference')]
    alternatives=site.get('nucleophile_candidates')
    if alternatives:
        center=_atom(atoms,refs[1])['xyz']
        # Missing candidates remain unresolved; do not choose only among surviving atoms.
        choices=[(candidate,_atom(atoms,candidate)['xyz']) for candidate in alternatives]
        refs[0]=min(choices,key=lambda item:float(np.linalg.norm(item[1]-center)))[0]
    return refs


def _range(value, name):
    if not isinstance(value, list) or len(value) != 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in value) or not 0 <= value[0] <= value[1]:
        raise ValueError(f"{name} requires an explicit finite nonnegative [minimum, maximum]")
    return value


def _angle(a, b, c):
    u, v = a - b, c - b
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("attack-angle atoms overlap")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / norm, -1., 1.))))


def component_inventories(sources):
    """Resolve explicitly fingerprinted CCD atom inventories, never residue-name guesses."""
    import gemmi
    result={}
    for ccd,source in (sources or {}).items():
        path=Path(source['path'])
        if file_digest(path)!=source.get('sha256'):raise ValueError('Chemical component source fingerprint changed')
        block=gemmi.cif.read_file(str(path)).sole_block()
        if gemmi.cif.as_string(block.find_value('_chem_comp.id'))!=ccd:
            raise ValueError('Chemical component source identifies a different CCD')
        atoms={gemmi.cif.as_string(row[0]):gemmi.cif.as_string(row[1]).upper() for row in block.find(['_chem_comp_atom.atom_id','_chem_comp_atom.type_symbol'])
               if gemmi.cif.as_string(row[1]).upper() not in {'H','D'}}
        if not atoms:raise ValueError('Chemical component has no heavy-atom inventory')
        result[ccd]=atoms
    return result


def validate_pose(path, execution, *, component_atoms=None, generic_ligand_names=()):
    """Measure one pose using exact atom identities and reviewed tolerances."""
    record = execution["modeling_input"]
    plan = (record.get("validation_plan") or {}).get("poses") or {}
    checks = []
    try:
        source_hash = file_digest(path)
    except OSError:
        source_hash = None
    try:
        atoms, sequences = _read_structure(path)
    except (OSError, ValueError, RuntimeError) as exc:
        return {"path": str(path), "sha256": source_hash, "status": "fail", "checks": [_check("structure", "fail", str(exc))]}
    entities = execution.get("entity_map") or []
    if not entities:
        checks.append(_check("inventory", "abstain", "renderer entity map is missing"))
    wanted_chains = {e["chain"] for e in entities}
    actual_chains = {k[0] for k in atoms}
    if entities:
        checks.append(_check("chain_inventory", "pass" if wanted_chains == actual_chains else "fail",
                             "predicted chain set compared with rendered input",
                             expected=sorted(wanted_chains), observed=sorted(actual_chains)))
    for entity in entities:
        chain = entity["chain"]
        selected = [a for k, a in atoms.items() if k[0] == chain]
        if entity["kind"] == "protein":
            same = sequences.get(chain) == entity["sequence"]
            checks.extend(_protein_completeness(atoms, chain, plan.get("protein_atom_policy") or {}))
        elif entity["kind"] == "cofactor":
            residues = {k[1] for k in atoms if k[0] == chain}
            names={a['resname'] for a in selected}
            same=bool(selected) and len(residues)==1 and names=={entity['ccd']}
            expected=(component_atoms or {}).get(entity['ccd'])
            if expected is not None:
                observed={key[2]:atom['element'] for key,atom in atoms.items() if key[0]==chain}
                recognized=(names=={entity['ccd']} or len(names)==1 and names<=set(generic_ligand_names))
                same=recognized and len(residues)==1 and observed==expected
        else:
            same = Counter(a["element"] for a in selected) == Counter(entity["elements"])
        checks.append(_check(f"identity.{chain}", "pass" if same else "fail",
                             "predicted entity compared with rendered identity (substrate elemental inventory only)"))
    checks.extend(_source_checks(plan.get("evidence"), "pose_criteria.evidence"))
    checks.extend(validate_pose_plan(record, entities)["checks"])
    sites = plan.get("sites") or []
    if not sites:
        checks.append(_check("attack_geometry", "abstain", "site-specific reaction atoms and tolerances are missing"))
    expected_sites = (((record.get("validation_plan") or {}).get("setup") or {}).get("assembly") or {}).get("expected", {}).get("active_sites")
    if not _positive_int(expected_sites) or len(sites) != expected_sites or len({s.get("id") for s in sites}) != len(sites):
        checks.append(_check("active_site_coverage", "abstain", "declare a distinct geometry check for every modeled active site"))
    for site in sites:
        name = f"site.{site.get('id', 'unnamed')}"
        try:
            lo, hi = _range(site.get("distance_range_A"), "attack distance")
            alo, ahi = _range(site.get("angle_range_deg"), "attack angle")
            if ahi > 180:
                raise ValueError("attack angles must lie in [0, 180]")
            refs = _geometry_refs(atoms,site)
            substrate_chains = {e["chain"] for e in entities if e["kind"] == "substrate"}
            if site.get("substrate_chain") not in substrate_chains or not any(r["chain"] == site["substrate_chain"] for r in refs):
                raise ValueError("attack geometry must identify the participating rendered substrate chain")
            keys = [(r["chain"], str(r["residue"]), r["atom"]) for r in refs]
            if len(set(keys)) != 3:
                raise ValueError("three distinct attack-angle atom identities are required")
            nu, el, ref = [_atom(atoms, r)["xyz"] for r in refs]
            distance, angle = float(np.linalg.norm(nu-el)), _angle(nu, el, ref)
            checks.append(_check(name, "pass" if lo <= distance <= hi and alo <= angle <= ahi else "fail",
                                 "distance and mechanism-specific attack angle evaluated jointly",
                                 distance_A=distance, angle_deg=angle,nucleophile=refs[0]))
        except LookupError as exc:
            checks.append(_check(name, "abstain", str(exc)))
        except (KeyError, TypeError, ValueError) as exc:
            checks.append(_check(name, "fail", str(exc)))
    # Each distance group can declare metal donor distances, required catalytic
    # contacts or a reference-supported interchain contact. No universal cutoffs.
    for group in plan.get("distance_checks") or []:
        name = f"distance.{group.get('id', 'unnamed')}"
        try:
            low, high = _range(group.get("range_A"), "contact distance")
            first, second = _atom(atoms, group["a"]), _atom(atoms, group["b"])
            if group["a"] == group["b"]:
                raise ValueError("contact endpoints must be distinct")
            distance = float(np.linalg.norm(first["xyz"] - second["xyz"]))
            checks.append(_check(name, "pass" if low <= distance <= high else "fail",
                                 "declared cofactor, catalytic or interface contact", distance_A=distance))
        except LookupError as exc:
            checks.append(_check(name, "abstain", str(exc)))
        except (KeyError, TypeError, ValueError) as exc:
            checks.append(_check(name, "fail", str(exc)))
    covered_cofactors = set()
    for group in plan.get("distance_checks") or []:
        a, b = group.get("a") or {}, group.get("b") or {}
        if a.get("chain") != b.get("chain"):
            covered_cofactors.update((a.get("chain"), b.get("chain")))
    required_cofactors = {e["chain"] for e in entities if e["kind"] == "cofactor"}
    if required_cofactors - covered_cofactors:
        checks.append(_check("cofactor_contact_coverage", "abstain", "each cofactor needs a declared contact to its modeled environment",
                             missing=sorted(required_cofactors-covered_cofactors)))
    metal_chains = {e["chain"] for e in entities if e["kind"] == "cofactor" and e.get("role") in {"metal", "catalytic_metal"}}
    import gemmi
    for entity in entities:
        if entity["kind"] == "cofactor":
            selected = [a for k, a in atoms.items() if k[0] == entity["chain"]]
            if len(selected) == 1 and gemmi.Element(selected[0]["element"]).is_metal:
                metal_chains.add(entity["chain"])
    coordinated = set()
    for group in plan.get("coordination_checks") or []:
        name = f"coordination.{group.get('id', 'unnamed')}"
        try:
            center = _atom(atoms, group["center"])
            if group["center"]["chain"] not in metal_chains or center["element"] != group["element"].upper():
                raise ValueError("coordination center must match a declared metal entity and element")
            low, high = _range(group.get("distance_range_A"), "coordination distance")
            nlo, nhi = _range(group.get("donor_count_range"), "donor count")
            refs = group["donors"]
            keys = [(r["chain"], str(r["residue"]), r["atom"]) for r in refs]
            if not refs or len(set(keys)) != len(keys) or any(r == group["center"] for r in refs):
                raise ValueError("declare distinct donor atoms excluding the metal center")
            distances = [float(np.linalg.norm(_atom(atoms, r)["xyz"]-center["xyz"])) for r in refs]
            count = sum(low <= d <= high for d in distances)
            checks.append(_check(name, "pass" if nlo <= count <= nhi else "fail", "coordination of declared modeled donors only",
                                 donor_distances_A=distances, donors_in_range=count))
            coordinated.add(group["center"]["chain"])
        except LookupError as exc:
            checks.append(_check(name, "abstain", str(exc)))
        except (KeyError, TypeError, ValueError) as exc:
            checks.append(_check(name, "fail", str(exc)))
    if metal_chains-coordinated:
        checks.append(_check("metal_coordination_coverage", "abstain", "explicit donor checks are missing for modeled metal sites",
                             missing=sorted(metal_chains-coordinated)))
    proteins = {e["chain"] for e in entities if e["kind"] == "protein"}
    connected = {c: {c} for c in proteins}
    for interface in plan.get("interfaces") or []:
        name = f"interface.{interface.get('id', 'unnamed')}"
        try:
            first, second = interface["chains"]
            radius, minimum = interface["contact_radius_A"], interface["minimum_residue_pairs"]
            if first == second or first not in proteins or second not in proteins:
                raise ValueError("interface must identify two rendered protein chains")
            if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0 or not _positive_int(minimum):
                raise ValueError("explicit positive interface radius and residue-pair count are required")
            left = [(k, a) for k, a in atoms.items() if k[0] == first]
            right = [(k, a) for k, a in atoms.items() if k[0] == second]
            from scipy.spatial import cKDTree
            pairs = set()
            if left and right:
                neighbors = cKDTree([a["xyz"] for _, a in left]).query_ball_tree(
                    cKDTree([a["xyz"] for _, a in right]), radius)
                pairs = {(left[i][0][1], right[j][0][1])
                         for i, hits in enumerate(neighbors) for j in hits}
            checks.append(_check(name, "pass" if len(pairs) >= minimum else "fail", "predicted interface compared with declared contact requirements",
                                 residue_pairs=len(pairs)))
            connected[first].add(second)
            connected[second].add(first)
        except (KeyError, TypeError, ValueError) as exc:
            checks.append(_check(name, "fail", str(exc)))
    if len(proteins) > 1:
        reached, frontier = set(), {next(iter(proteins))}
        while frontier:
            reached.update(frontier)
            frontier = set().union(*(connected[c] for c in frontier))-reached
        if reached != proteins:
            checks.append(_check("assembly_interface_coverage", "abstain", "declared interface checks do not connect the modeled protein chains"))
    clashes = plan.get("ligand_environment")
    if not clashes:
        checks.append(_check("ligand_environment", "abstain", "ligand contact and clash criteria are missing"))
    else:
        try:
            radius = clashes["contact_radius_A"]
            clash_distance = clashes["clash_distance_A"]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (radius, clash_distance)):
                raise ValueError("contact and clash distances must be finite and positive")
            minimum = clashes["minimum_contact_atoms"]
            maximum = clashes["maximum_clashing_atoms"]
            if type(minimum) is not int or minimum < 1 or type(maximum) is not int or maximum < 0:
                raise ValueError("declare positive contact and nonnegative clash counts")
            substrate_chains = {e["chain"] for e in entities if e["kind"] == "substrate"}
            protein_chains = {e["chain"] for e in entities if e["kind"] == "protein"}
            if not substrate_chains or not protein_chains:
                raise LookupError("substrate and protein entity assignments are required")
            protein = np.array([v["xyz"] for k, v in atoms.items() if k[0] in protein_chains])
            if len(protein) == 0:
                raise LookupError("protein atoms are absent")
            for chain in sorted(substrate_chains):
                ligand = [v for k, v in atoms.items() if k[0] == chain]
                if not ligand:
                    raise LookupError(f"substrate chain {chain} is absent")
                distances = np.array([np.linalg.norm(protein-a["xyz"], axis=1).min() for a in ligand])
                contacts, overlaps = int((distances <= radius).sum()), int((distances < clash_distance).sum())
                checks.append(_check(f"environment.{chain}", "pass" if contacts >= minimum and overlaps <= maximum else "fail",
                                     "ligand-protein contacts and heavy-atom overlaps", contact_atoms=contacts,
                                     clashing_atoms=overlaps, minimum_distance_A=float(distances.min())))
        except LookupError as exc:
            checks.append(_check("ligand_environment", "abstain", str(exc)))
        except (KeyError, TypeError, ValueError) as exc:
            checks.append(_check("ligand_environment", "fail", str(exc)))
    try:
        current_hash = file_digest(path)
    except OSError:
        current_hash = None
    if current_hash != source_hash:
        checks.append(_check("source_integrity", "abstain", "prediction bytes changed or became unavailable during validation"))
    return {"path": str(path), "sha256": source_hash, "status": _status(checks), "checks": checks}


def validate_ensemble(paths, execution, *, supplemental_checks=None, chemical_component_sources=None,generic_ligand_names=()):
    """Report every pose, sampling coverage and the jointly supported fraction."""
    paths = [Path(p).resolve() for p in paths]
    plan = ((execution.get("modeling_input") or {}).get("validation_plan") or {}).get("poses") or {}
    checks = []
    expected = plan.get("expected_predictions")
    minimum = plan.get("minimum_passing_predictions")
    if not _positive_int(expected) or not _positive_int(minimum) or minimum > expected:
        checks.append(_check("sampling_plan", "abstain", "declare expected predictions and a minimum passing count"))
    else:
        checks.append(_check("sampling_count", "pass" if len(paths) == expected else "abstain",
                             "complete declared ensemble required", expected=expected, observed=len(paths)))
    if len({str(p.resolve()) for p in paths}) != len(paths):
        checks.append(_check("prediction_identity", "fail", "duplicate prediction paths would inflate coverage"))
    component_atoms=component_inventories(chemical_component_sources)
    rows = [validate_pose(p, execution,component_atoms=component_atoms,generic_ligand_names=generic_ligand_names) for p in paths]
    for row in rows:
        row["checks"].extend((supplemental_checks or {}).get(str(Path(row["path"]).resolve()), []))
        row["status"] = _status(row["checks"])
    digests = [r["sha256"] for r in rows if r.get("sha256")]
    if len(set(digests)) != len(digests):
        checks.append(_check("distinct_samples", "abstain", "byte-identical prediction copies do not establish distinct generated samples"))
    passing = sum(r["status"] == "pass" for r in rows)
    unresolved = sum(r["status"] == "abstain" for r in rows)
    if _positive_int(minimum):
        verdict = "pass" if passing >= minimum else "abstain" if unresolved or len(paths) < (expected or 0) else "fail"
        checks.append(_check("reasonable_poses", verdict, "count of poses jointly meeting all declared checks",
                             passing=passing, unresolved=unresolved, required=minimum))
    if not paths:
        checks.append(_check("predictions", "abstain", "no prediction files supplied"))
    if any(not r.get("sha256") for r in rows):
        checks.append(_check("prediction_fingerprints", "abstain", "some prediction inputs cannot be fingerprinted"))
    if any(c["check"] == "source_integrity" and c["status"] != "pass" for row in rows for c in row["checks"]):
        checks.append(_check("prediction_fingerprints", "abstain", "prediction bytes changed while measurements were computed"))
    if any(file_digest(source['path'])!=source['sha256'] for source in (chemical_component_sources or {}).values()):
        checks.append(_check('chemical_component_integrity','abstain','Chemical component bytes changed during validation'))
    return {"schema_version": "1.0", "status": _status(checks), "checks": checks,
            "execution_sha256": fingerprint(execution), "n_predictions": len(paths),
            "validator_sha256": file_digest(__file__), "n_unique_file_contents": len(set(digests)),
            "n_passing": passing, "n_unresolved": unresolved,
            "passing_fraction": passing/len(paths) if paths else None,
            "chemical_component_sources":chemical_component_sources or {},
            "poses": rows, "note": "A failure concerns these model inputs/poses, not enzyme inactivity. Confidence does not override physical checks. Atom inventories do not establish connectivity; metal contacts do not establish a complete hydration sphere."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup")
    setup.add_argument("--record", required=True, type=Path, help="resolved modeling_input JSON")
    setup.add_argument("--sequence", help="full precursor amino-acid sequence")
    poses = sub.add_parser("poses")
    poses.add_argument("--manifest", required=True, type=Path, help="renderer campaign_manifest.json")
    poses.add_argument("--accession", required=True)
    poses.add_argument("--predictions", required=True, type=Path)
    for command in (setup, poses):
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "setup":
        result = validate_setup(json.loads(args.record.read_text()), args.sequence)
    else:
        manifest = json.loads(args.manifest.read_text())
        matches = [e for e in manifest["executions"] if e["accession"] == args.accession]
        if len(matches) != 1:
            parser.error("accession must identify exactly one rendered execution")
        paths = sorted(p for p in args.predictions.iterdir() if p.suffix.lower() in {".pdb", ".cif", ".mmcif"})
        result = validate_ensemble(paths, matches[0])
    from enzyme_discovery_agent.state_store import atomic_write_json
    atomic_write_json(args.output, result, allow_nan=False)
    print(f"{result['status']}: {args.output}")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
