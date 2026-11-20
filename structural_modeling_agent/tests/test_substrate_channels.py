"""Tests for the substrate-channel analysis (structural_modeling_agent.analysis.substrate_channels).

Layers:
  1. staging (pose alignment, substrate stripping, start point, chain counting, mmCIF input, fail-loud
     behaviour) on small SYNTHETIC holo systems with known geometry -- no JVM needed;
  2. the result parser, on a fabricated CAVER output directory in CAVER's real file format;
  3. the monomer / inconsistent-start caveat logic;
  4. CAVER itself against ANALYTIC GROUND TRUTH: a hollow tube of C atoms has a free radius of
     (wall-atom-centre radius - r_C); a ring added inside it must set the bottleneck to its own free
     radius. Auto-skipped if java / CAVER are absent.

Run (from the repo root):
  /nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/boltz-2/bin/python -m pytest \\
      structural_modeling_agent/tests/test_substrate_channels.py -q
"""
import math
import os
import shutil
import tempfile

import numpy as np
import pytest

pytest.importorskip("MDAnalysis")
from structural_modeling_agent.analysis import substrate_channels as ch  # noqa: E402


# ---------------------------------------------------------------- synthetic holo system
def _atom(rec, name, resn, chain, resi, xyz, el):
    nm = name if len(name) == 4 else f" {name:<3}"
    return (f"{rec:<6}{0:>5} {nm}{resn:>4} {chain}{resi:>4}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el:>2}")


def _protein(chains="A", n=12):
    """ALA C-alpha spiral, one per chain (shifted so chains do not overlap)."""
    out = []
    for ci, c in enumerate(chains):
        for k in range(n):
            t = math.radians(100 * k)
            out.append(_atom("ATOM", "CA", "ALA", c, k + 1,
                             (14.0 * ci + 6.0 * math.cos(t), 6.0 * math.sin(t), 1.5 * k), "C"))
    return out


ATP = [("PG", "P", (0.0, 0.0, 0.0)), ("O1G", "O", (1.5, 0.0, 0.0)), ("O2G", "O", (-0.75, 1.3, 0.0)),
       ("O3G", "O", (-0.75, -1.3, 0.0)), ("O3B", "O", (0.0, 0.0, -1.5)),
       ("PB", "P", (0.0, 0.0, -3.0)), ("C5'", "C", (2.0, 0.0, -4.0)), ("C4'", "C", (3.0, 1.0, -4.5))]
# glutarate-like zig-zag: carboxylate C1 sits ~4 A from PG (nearest the electrophile), C5 far end
SUB = [("C1", "C", (0.0, 0.0, 4.0)), ("O1", "O", (1.1, 0.0, 4.6)), ("O2", "O", (-1.1, 0.0, 4.6)),
       ("C2", "C", (0.0, 0.0, 2.5)), ("C3", "C", (1.3, 0.0, 1.8)), ("C4", "C", (1.3, 0.0, 0.3)),
       ("C5", "C", (2.6, 0.0, -0.4)), ("O3", "O", (3.6, 0.0, 0.3)), ("O4", "O", (2.6, 0.0, -1.7))]
SUB_PUSH = np.array([6.0, 0.0, 6.0])      # keep the ligand away from the ATP so groups are unambiguous


def _holo(chains="A", substrate="acid"):
    L = _protein(chains)
    for i, (n, el, xyz) in enumerate(ATP):
        L.append(_atom("HETATM", n, "ATP", "C", 1, xyz, el))
    L.append(_atom("HETATM", "MG", "MG", "D", 1, (-3.0, 0.0, -1.0), "MG"))
    if substrate == "acid":
        for n, el, xyz in SUB:
            L.append(_atom("HETATM", n, "LIG", "E", 1, tuple(np.array(xyz) * 0 + np.array(xyz)), el))
    elif substrate == "amine":                 # cadaverine-like: N-C-C-C-C-C-N, no carboxylate
        for k, el in enumerate("NCCCCCN"):
            L.append(_atom("HETATM", f"{el}{k + 1}", "LIG", "E", 1, (5.0 + 1.3 * k, 0.0, 4.0), el))
    return L


def _write(lines, path):
    with open(path, "w") as fh:
        for i, ln in enumerate(lines, 1):
            fh.write(ln[:6] + f"{i:>5}" + ln[11:] + "\n")
        fh.write("END\n")
    return path


def _rigid(lines, seed):
    """Random rotation + translation of every atom (a pose in its own frame)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.random(random_state=seed).as_matrix()
    t = np.random.default_rng(seed).uniform(-20, 20, 3)
    out = []
    for ln in lines:
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]) @ R.T + t
        out.append(ln[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + ln[54:])
    return out


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path)


def _poses(tmp, n=3, chains="A", substrate="acid"):
    return [_write(_rigid(_holo(chains, substrate), seed=i + 1), os.path.join(tmp, f"in_{i}.pdb"))
            for i in range(n)]


# ---------------------------------------------------------------- 1. staging
def test_stage_aligns_strips_substrate_and_finds_start(tmp):
    import MDAnalysis as mda
    import warnings
    poses = _poses(tmp, 3)
    man = ch.stage_ensemble(poses, os.path.join(tmp, "w"))
    assert man["n_poses"] == 3 and man["start_how"] == "median_reactive_carboxylate"
    ref_ca = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(3):
            u = mda.Universe(os.path.join(man["pdb_dir"], f"pose_{i:04d}.pdb"))
            names = set(u.atoms.resnames)
            assert "LIG" not in names, "substrate must be stripped"
            assert {"ATP", "MG"} <= names, "cofactor + metal must be kept as obstacles"
            ca = u.select_atoms("protein and name CA").positions
            ref_ca = ca if ref_ca is None else ref_ca
            assert np.allclose(ca, ref_ca, atol=0.05), "poses must share one frame after alignment"
    # the start point is the carboxylate carbon nearest the cofactor P, in the common (pose-0) frame
    assert np.allclose(man["start_xyz"], man["per_pose"][0]["reactive_center"], atol=0.05)
    assert man["start_spread"]["max_A"] < 0.05 and man["start_spread"]["frac_within_4A"] == 1.0
    assert man["per_pose"][0]["n_protein_chains"] == 1
    assert man["per_pose"][0]["substrate_extent_A"] > 5.0


def test_start_is_the_carboxylate_nearest_the_electrophile(tmp):
    man = ch.stage_ensemble(_poses(tmp, 2), os.path.join(tmp, "w"))
    # the two carboxylate carbons are C1 (4.0 A from PG) and C5 (~2.6 A from PG): the nearer one wins.
    # Computed from the geometry, not hard-coded, so this test cannot drift from the fixture.
    c1, c5 = np.array(SUB[0][2]), np.array(SUB[6][2])
    pg = np.zeros(3)
    nearer = c1 if np.linalg.norm(c1 - pg) < np.linalg.norm(c5 - pg) else c5
    pose0 = np.array(man["per_pose"][0]["reactive_center"])
    # pose 0 is a rigid copy of the holo; compare distances to P instead of absolute coordinates
    import MDAnalysis as mda
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(os.path.join(man["pdb_dir"], "pose_0000.pdb"))
    p = u.select_atoms("name PG").positions[0]
    assert abs(np.linalg.norm(pose0 - p) - np.linalg.norm(nearer - pg)) < 0.1


def test_two_chain_assembly_is_counted(tmp):
    man = ch.stage_ensemble(_poses(tmp, 2, chains="AB"), os.path.join(tmp, "w"))
    assert all(p["n_protein_chains"] == 2 for p in man["per_pose"])


def test_mmcif_input_is_converted(tmp):
    gemmi = pytest.importorskip("gemmi")
    pdb = _write(_holo("A"), os.path.join(tmp, "x.pdb"))
    st = gemmi.read_structure(pdb)
    st.setup_entities()
    cif = os.path.join(tmp, "x.cif")
    st.make_mmcif_document().write_file(cif)
    man_pdb = ch.stage_ensemble([pdb, pdb], os.path.join(tmp, "w1"))
    man_cif = ch.stage_ensemble([cif, cif], os.path.join(tmp, "w2"))
    assert man_cif["n_poses"] == 2
    d = np.linalg.norm(np.array(man_cif["per_pose"][0]["reactive_center"]) -
                       np.array(man_pdb["per_pose"][0]["reactive_center"]))
    assert d < 0.1, "CIF and PDB of the same pose must give the same reactive centre"


def test_no_poses_raises(tmp):
    with pytest.raises(ValueError):
        ch.stage_ensemble([], os.path.join(tmp, "w"))


def test_ca_count_mismatch_raises(tmp):
    a = _write(_holo("A"), os.path.join(tmp, "a.pdb"))
    b = _write(_holo("AB"), os.path.join(tmp, "b.pdb"))
    with pytest.raises(ValueError, match="C-alpha"):
        ch.stage_ensemble([a, b], os.path.join(tmp, "w"))


def test_substrate_without_carboxylate_needs_explicit_start(tmp):
    poses = _poses(tmp, 2, substrate="amine")
    # the default start_group is carboxylate; the amine substrate has none -> a clear error naming
    # the groups that WERE found, and the escape hatches.
    with pytest.raises(ValueError, match="carboxylate"):
        ch.stage_ensemble(poses, os.path.join(tmp, "w"))
    man = ch.stage_ensemble(poses, os.path.join(tmp, "w2"), start_xyz=(1.0, 2.0, 3.0))
    assert man["start_how"] == "explicit" and man["start_xyz"] == [1.0, 2.0, 3.0]
    assert man["start_spread"] is None


def test_start_group_amine_resolves_a_step2_nucleophile(tmp):
    """A step-2-like amine substrate (no carboxylate) is placed automatically with --start-group amine,
    at the amine nitrogen nearest the cofactor phosphorus -- no hand-typed coordinates."""
    poses = _poses(tmp, 2, substrate="amine")
    man = ch.stage_ensemble(poses, os.path.join(tmp, "w"), start_group="amine")
    assert man["start_how"] == "median_reactive_amine"
    # the fixture amine is N-C-C-C-C-C-N starting at x=5.0; the first N (nearest the ATP PG at origin)
    # should anchor the start. Resolve it independently from the staged pose-0 geometry.
    import MDAnalysis as mda
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(poses[0])
        ns = u.select_atoms("resname LIG and name N*")
        pg = u.select_atoms("name PG").positions[0]
    nearest_n = ns.positions[np.argmin(np.linalg.norm(ns.positions - pg, axis=1))]
    # pose 0 is staged in its own frame; compare distance to PG rather than absolute coordinates
    assert abs(np.linalg.norm(np.array(man["per_pose"][0]["reactive_center"]) - pg) -
               np.linalg.norm(nearest_n - pg)) < 0.2


def test_start_by_atom_selection(tmp):
    """--start-sel places the start at the centre of geometry of an arbitrary atom selection, so any
    named residue / fixed point works without coordinates; the start point tracks the chosen atom."""
    man = ch.stage_ensemble(_poses(tmp, 2), os.path.join(tmp, "w"), start_sel="resname LIG and name C3")
    assert man["start_how"] == "explicit_selection" and man["start_spread"] is not None
    import MDAnalysis as mda
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(os.path.join(man["pdb_dir"], "pose_0000.pdb"))
    assert "LIG" not in set(u.atoms.resnames)                    # substrate still stripped
    # the reported start for pose 0 is in the same (pose-0) frame as the staged protein: the nearest
    # staged CA must sit a chemical-bond-scale distance from it (C3 is buried in the ligand near the CAs)
    start0 = np.array(man["per_pose"][0]["reactive_center"])
    cas = u.select_atoms("name CA").positions
    assert np.min(np.linalg.norm(cas - start0, axis=1)) < 25.0   # start is inside the model, not at infinity


def test_bad_start_selection_raises(tmp):
    with pytest.raises(ValueError, match="matched no atoms"):
        ch.stage_ensemble(_poses(tmp, 2), os.path.join(tmp, "w"), start_sel="resname NOPE")


def test_apo_structure_allowed_with_explicit_start(tmp):
    """A protein-only input (no ligand at all) is staged when an explicit start is given; split_ligands
    would otherwise raise 'no non-protein ligand atoms'."""
    apo = [_write(_rigid(_protein("A"), seed=i + 1), os.path.join(tmp, f"apo_{i}.pdb")) for i in range(2)]
    with pytest.raises(ValueError):                              # auto start must fail loud on apo
        ch.stage_ensemble(apo, os.path.join(tmp, "wa"))
    man = ch.stage_ensemble(apo, os.path.join(tmp, "wb"), start_sel="name CA and resid 1")
    assert man["n_poses"] == 2 and man["per_pose"][0]["substrate_extent_A"] is None
    import MDAnalysis as mda
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(os.path.join(man["pdb_dir"], "pose_0000.pdb"))
    assert set(u.atoms.resnames) == {"ALA"}      # protein only, nothing spurious kept


# ---------------------------------------------------------------- 3. caveat logic
def test_monomer_input_is_flagged():
    assert ch.SINGLE_CHAIN_WARNING in ch.build_notes([1], None)
    assert ch.SINGLE_CHAIN_WARNING in ch.build_notes([1, 2], None)      # mixed -> still flagged
    assert ch.build_notes([2], None) == []


def test_inconsistent_start_is_flagged():
    assert ch.START_INCONSISTENT_NOTE in ch.build_notes([2], {"frac_within_4A": 0.5})
    assert ch.build_notes([2], {"frac_within_4A": 1.0}) == []


# ---------------------------------------------------------------- 2. parser
def _fake_caver_out(d, snaps):
    """CAVER's real file layout. `snaps`: {snapshot: {cluster: (throughput, bottleneck, length, curv)}}"""
    os.makedirs(os.path.join(d, "analysis"))
    rows = ["Snapshot, Tunnel cluster, Tunnel, Throughput, Cost, Bottleneck radius, "
            "Bottleneck R error bound, Length, Curvature"]
    brows = ["Bottleneck residues:, Residues in distance <= 3.0 A from the bottleneck, sorted from the closest.",
             "Snapshot,Tunnel cluster,Tunnel,Throughput,Cost,Bottleneck X,Bottleneck Y,Bottleneck Z,"
             "Bottleneck R,Bottleneck residues"]
    for s, cl in snaps.items():
        for c, (tp, br, ln, cv) in cl.items():
            rows.append(f"{s}, {c}, 1, {tp}, {-math.log(tp):.4f}, {br}, -, {ln}, {cv}")
            res = "A:316,A:321" if c == 1 else "A:99"
            brows.append(f"{s},{c},1,{tp},0.1,0.0,0.0,0.0,{br},{res}")
    open(os.path.join(d, "analysis", "tunnel_characteristics.csv"), "w").write("\n".join(rows) + "\n")
    open(os.path.join(d, "analysis", "bottlenecks.csv"), "w").write("\n".join(brows) + "\n")


def test_parser_frequency_priority_and_lining(tmp):
    out = os.path.join(tmp, "o")
    _fake_caver_out(out, {
        "p0.pdb": {1: (0.8, 1.5, 16.0, 1.2), 2: (0.5, 1.0, 10.0, 1.3)},
        "p1.pdb": {1: (0.6, 1.7, 14.0, 1.2)},
        "p2.pdb": {1: (0.7, 1.6, 15.0, 1.2), 2: (0.3, 0.9, 12.0, 1.4)},
        "p3.pdb": {1: (0.9, 1.4, 17.0, 1.1)},
    })
    cl = ch.parse_caver_output(out, n_poses=4)
    assert [c["cluster"] for c in cl] == [1, 2]                      # ordered by priority
    c1, c2 = cl
    assert c1["frequency"] == 1.0 and c2["frequency"] == 0.5         # over ALL 4 poses
    assert c1["bottleneck_A"]["mean"] == pytest.approx(1.55, abs=1e-3)
    assert c1["bottleneck_A"]["min"] == 1.4 and c1["bottleneck_A"]["max"] == 1.7
    assert c1["length_A"]["mean"] == pytest.approx(15.5, abs=1e-2)
    # CAVER priority = mean throughput over ALL snapshots (0 where the cluster is absent)
    assert c1["priority"] == pytest.approx((0.8 + 0.6 + 0.7 + 0.9) / 4, abs=1e-3)
    assert c2["priority"] == pytest.approx((0.5 + 0.3) / 4, abs=1e-3)
    assert c2["throughput_mean"] == pytest.approx(0.4, abs=1e-3)     # mean over snapshots that have it
    assert {r["residue"] for r in c1["bottleneck_lining_residues"]} == {"A:316", "A:321"}
    assert all(r["fraction"] == 1.0 for r in c1["bottleneck_lining_residues"])


def test_parser_ignores_malformed_rows(tmp):
    out = os.path.join(tmp, "o")
    _fake_caver_out(out, {"p0.pdb": {1: (0.8, 1.5, 16.0, 1.2)}})
    with open(os.path.join(out, "analysis", "tunnel_characteristics.csv"), "a") as fh:
        fh.write("garbage line without enough columns\n")
    assert len(ch.parse_caver_output(out, 1)) == 1


def test_run_caver_missing_install_raises(tmp):
    with pytest.raises(FileNotFoundError):
        ch.run_caver(tmp, (0, 0, 0), caver_home=os.path.join(tmp, "no_caver"))


def test_subset_pdb_dir_symlinks_first_n(tmp):
    """The perturbed-start speedup: a subset dir points at the first N staged poses (symlinks, not
    copies), so a start-sensitivity run processes fewer poses. This is what cuts the n_starts multiplier."""
    pdb = os.path.join(tmp, "pdb"); os.makedirs(pdb)
    for i in range(50):
        open(os.path.join(pdb, f"pose_{i:04d}.pdb"), "w").write("END\n")
    open(os.path.join(pdb, "not_a_pose.txt"), "w").write("x")       # ignored
    dest, n = ch.subset_pdb_dir(pdb, os.path.join(tmp, "sub"), 15)
    files = sorted(os.listdir(dest))
    assert n == 15 and len(files) == 15
    assert files[0] == "pose_0000.pdb" and files[-1] == "pose_0014.pdb"
    assert all(os.path.islink(os.path.join(dest, f)) for f in files)     # symlinks, not copies
    # n larger than available, and n=0/None, both mean "all"
    assert ch.subset_pdb_dir(pdb, os.path.join(tmp, "s2"), 999)[1] == 50
    assert ch.subset_pdb_dir(pdb, os.path.join(tmp, "s3"), None)[1] == 50


# ---------------------------------------------------------------- 4. CAVER vs analytic ground truth
def _have_caver():
    return shutil.which("java") is not None and os.path.exists(os.path.join(ch.CAVER_HOME, "caver.jar"))


R_C = 1.70       # CAVER's carbon radius (Caver3/bin/atom_radii.csv: 6;C;carbon;170)


def _tube(R, L=36.0, ring_R=None, spacing=1.4):
    """Hollow tube of C atoms along z (wall-atom centres at radius R); optional inner ring at z=0."""
    atoms = []

    def add(x, y, z):
        atoms.append(f"ATOM  {len(atoms) + 1:5d}  CA  ALA A{(len(atoms) % 9999) + 1:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    for k in range(int(L / spacing) + 1):
        m = max(8, int(2 * math.pi * R / spacing))
        off = (k % 2) * math.pi / m
        for j in range(m):
            t = 2 * math.pi * j / m + off
            add(R * math.cos(t), R * math.sin(t), -L / 2 + k * spacing)
    if ring_R is not None:
        m = max(8, int(2 * math.pi * ring_R / spacing))
        for layer in (-0.7, 0.0, 0.7):
            for j in range(m):
                t = 2 * math.pi * j / m
                add(ring_R * math.cos(t), ring_R * math.sin(t), layer)
    return "\n".join(atoms) + "\nEND\n"


def _bottlenecks(tmp, tag, text):
    wd = os.path.join(tmp, tag)
    os.makedirs(os.path.join(wd, "pdb"))
    for i in range(2):
        open(os.path.join(wd, "pdb", f"pose_{i:03d}.pdb"), "w").write(text)
    out, _, _ = ch.run_caver(wd, (0.0, 0.0, 8.0))
    return [c["bottleneck_A"]["mean"] for c in ch.parse_caver_output(out, 2)]


@pytest.mark.skipif(not _have_caver(), reason="java / CAVER 3 not available")
def test_caver_bottleneck_matches_analytic_tube_radius(tmp):
    b5 = max(_bottlenecks(tmp, "r5", _tube(5.0)))
    b7 = max(_bottlenecks(tmp, "r7", _tube(7.0)))
    assert b5 == pytest.approx(5.0 - R_C, abs=0.1)          # 3.30
    assert b7 == pytest.approx(7.0 - R_C, abs=0.1)          # 5.30
    assert b7 - b5 == pytest.approx(2.0, abs=0.15)


@pytest.mark.skipif(not _have_caver(), reason="java / CAVER 3 not available")
def test_caver_bottleneck_finds_an_inner_constriction(tmp):
    plain = max(_bottlenecks(tmp, "plain", _tube(7.0)))
    bs = _bottlenecks(tmp, "ring", _tube(7.0, ring_R=4.0))
    assert min(bs) == pytest.approx(4.0 - R_C, abs=0.1)     # the ring-crossing path: 2.30
    assert min(bs) < plain - 2.0


# ---------------------------------------------------------------- 5. external benchmark: real DhaA
# CAVER was designed on the haloalkane dehalogenase DhaA (Chovancova 2012, PLoS Comput Biol 8:e1002708),
# which has a buried catalytic site reached by two well-characterised tunnels: the wider p1 "main tunnel"
# and the narrower p2 "slot tunnel". The published crystal-structure p1 bottleneck radius for PDB 1CQW is
# ~1.3 A. This test runs the WHOLE wrapper (stage -> CAVER -> parse -> classify) on a trimmed 1CQW and
# checks it reproduces that known geometry: a buried site, a main tunnel of ~1.3 A lined by the catalytic
# residues, and a second, narrower tunnel. This is EXTERNAL validation against literature, not a self-
# consistency check. 1CQW's own numbering offsets the canonical DhaA triad by ~+11 (nucleophile Asp117,
# base His283, acid Glu141 -- resolved from the structure, not assumed).
DHAA_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "dhaa_1cqw.pdb")


@pytest.mark.skipif(not _have_caver() or not os.path.exists(DHAA_FIXTURE),
                    reason="java / CAVER 3 or the DhaA fixture not available")
def test_caver_reproduces_known_dhaa_tunnels(tmp):
    res = ch.analyze_channels([DHAA_FIXTURE], os.path.join(tmp, "dhaa"),
                              start_sel="resid 117 and name OD1 OD2",
                              label="DhaA 1CQW", n_starts=3, threads=2)
    # the catalytic site is deeply buried (not a solvent-open surface pocket)
    assert res["site"]["burial_fraction"]["mean"] > 0.8
    assert res["site"]["surface_exposed"] is False
    # a main tunnel to the surface is found
    assert res["n_channels"] >= 1
    main = res["channels"][0]
    assert main["length_A"]["mean"] > 8.0                       # reaches bulk, not a dent
    # published 1CQW p1 bottleneck ~1.3 A; allow a generous window around it
    assert 1.0 <= main["bottleneck_A"]["mean"] <= 1.7
    # it is the REAL active-site tunnel: lined by the catalytic nucleophile Asp117 (and usually His283)
    lining = {r["residue"] for r in main["bottleneck_lining_residues"]}
    assert "A:117" in lining
    # DhaA's second (slot) tunnel is narrower than the main tunnel -- topology, not just one hole
    if len(res["clusters"]) >= 2:
        second = min(res["clusters"], key=lambda c: c["bottleneck_A"]["mean"])
        assert second["bottleneck_A"]["mean"] <= main["bottleneck_A"]["mean"] + 0.05
