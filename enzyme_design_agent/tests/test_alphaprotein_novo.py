"""Offline tests for the de novo design tool (AlphaProtein Novo wrapper).

These never launch AP Novo or touch a GPU: the subprocess boundary and the
reservation gate are mocked, so what is under test is the wrapper's own logic --
manifest translation, GPU pinning, release-on-every-exit-path, and result
parsing.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from enzyme_design_agent.tools import alphaprotein_novo as dd


def _motif_cif(tmp: Path) -> Path:
    """A stand-in motif file; the wrapper only checks that it exists."""
    p = tmp / "motif.cif"
    p.write_text("data_motif\n#\n")
    return p


class TestManifestBuilding(unittest.TestCase):
    def _manifest(self, **over):
        base = dict(
            motif_cif_path="/abs/motif.cif",
            motif_str="A1,A2|10-100,{},2-80/B1",
            motif_atoms="",
            seq_length="",
            unindexed_motif_residues="",
            num_designs=4,
            num_sampling_steps=50,
            seed_start=0,
            is_author_naming=True,
            design_name="camp",
            resequence=True,
            resequence_temperature=0.1,
            num_sequences=2,
            folding_states=[],
            fold_apo_monomer=True,
            fold_seeds=1,
            evaluation_suite="",
            partial_diffusion_input_file="",
            partial_diffusion_num_steps=600,
        )
        base.update(over)
        return dd._build_manifest(**base)

    def test_core_fields_and_absolute_input_path(self):
        m = self._manifest()
        design = m["designs"][0]
        self.assertEqual(design["motif_str"], "A1,A2|10-100,{},2-80/B1")
        self.assertEqual(design["num_designs"], 4)
        self.assertEqual(m["defaults"]["num_sampling_steps"], 50)
        # AP Novo resolves relative manifest paths against the manifest's own
        # directory, so the motif path must be absolute.
        self.assertTrue(Path(design["input_file"]).is_absolute())

    def test_optional_motif_fields_omitted_when_empty(self):
        design = self._manifest()["designs"][0]
        for key in ("motif_atoms", "seq_length", "unindexed_motif_residues",
                    "partial_diffusion_input_file"):
            self.assertNotIn(key, design)

    def test_optional_motif_fields_passed_through(self):
        design = self._manifest(motif_atoms="A1:NE2,ND1", seq_length="120-160")["designs"][0]
        self.assertEqual(design["motif_atoms"], "A1:NE2,ND1")
        self.assertEqual(design["seq_length"], "120-160")

    def test_ligand_adds_holo_state_and_apo_can_be_dropped(self):
        m = self._manifest(fold_apo_monomer=False, folding_states=[
            {"name": "complex", "ligands": [{"id": "B", "ccd_code": "6NT"}]}])
        states = m["folding"]["states"]
        self.assertEqual([s["name"] for s in states], ["complex"])
        self.assertEqual(states[0]["ligands"], [{"id": "B", "ccd_code": "6NT"}])

    def test_multi_state_walks_the_reaction_coordinate(self):
        """Several states = the cross-state consistency filter from the paper."""
        m = self._manifest(fold_apo_monomer=True, folding_states=[
            {"name": "michaelis", "ligands": [
                {"id": "T", "ccd_code": "ATP"}, {"id": "M", "ccd_code": "MG"},
                {"id": "N", "ccd_code": "MG"},
                {"id": "L", "ccd_code": "pd00", "smiles": "CC(O)CC(=O)O"}]},
            {"name": "intermediate", "ligands": [
                {"id": "T", "ccd_code": "ADP"}, {"id": "M", "ccd_code": "MG"},
                {"id": "N", "ccd_code": "MG"},
                {"id": "L", "ccd_code": "pd00", "smiles": "CC(OP(=O)(O)O)CC(=O)O"}]},
        ])
        states = m["folding"]["states"]
        self.assertEqual([s["name"] for s in states],
                         ["monomer", "michaelis", "intermediate"])
        # same chain id carries different chemistry per state
        self.assertEqual(states[1]["ligands"][0]["ccd_code"], "ATP")
        self.assertEqual(states[2]["ligands"][0]["ccd_code"], "ADP")
        self.assertEqual(len(states[1]["ligands"]), 4)

    def test_unnamed_states_get_stable_fallback_names(self):
        m = self._manifest(fold_apo_monomer=False, folding_states=[
            {"ligands": [{"id": "B", "ccd_code": "ATP"}]},
            {"ligands": [{"id": "B", "ccd_code": "ADP"}]}])
        self.assertEqual([s["name"] for s in m["folding"]["states"]], ["state1", "state2"])

    def test_multiple_ligand_chains_and_custom_smiles(self):
        """A cofactor + two metals + a non-CCD substrate = four ligand chains."""
        m = self._manifest(fold_apo_monomer=False, folding_states=[{"name": "c", "ligands": [
            {"id": "B", "ccd_code": "ATP"},
            {"id": "C", "ccd_code": "MG"},
            {"id": "D", "ccd_code": "MG"},
            {"id": "E", "ccd_code": "LG1", "smiles": "CC(O)CC(=O)O"},
        ]}])
        ligs = m["folding"]["states"][0]["ligands"]
        self.assertEqual([l["id"] for l in ligs], ["B", "C", "D", "E"])
        self.assertEqual(ligs[3]["smiles"], "CC(O)CC(=O)O")
        # A real CCD code carries no smiles key.
        self.assertNotIn("smiles", ligs[0])

    def test_unknown_ligand_keys_are_dropped(self):
        m = self._manifest(fold_apo_monomer=False, folding_states=[{"name": "c", "ligands": [
            {"id": "B", "ccd_code": "ATP", "bogus": "x", "smiles": None},
        ]}])
        self.assertEqual(m["folding"]["states"][0]["ligands"], [{"id": "B", "ccd_code": "ATP"}])

    def test_no_folding_block_when_no_states_requested(self):
        m = self._manifest(fold_apo_monomer=False, folding_states=[])
        self.assertNotIn("folding", m)

    def test_folding_input_follows_resequence_flag(self):
        self.assertEqual(self._manifest(resequence=True)["folding"]["inputs"], ["resequenced"])
        self.assertEqual(self._manifest(resequence=False)["folding"]["inputs"], ["generated"])

    def test_disabled_resequence_carries_no_sampling_params(self):
        r = self._manifest(resequence=False)["resequence"]
        self.assertFalse(r["enabled"])
        self.assertNotIn("temperature", r)

    def test_partial_diffusion_pairs_file_with_step_count(self):
        design = self._manifest(
            partial_diffusion_input_file="/abs/parent.cif",
            partial_diffusion_num_steps=900,
        )["designs"][0]
        self.assertTrue(Path(design["partial_diffusion_input_file"]).is_absolute())
        self.assertEqual(design["partial_diffusion_num_steps"], 900)

    def test_fold_seeds_expand_to_seed_list(self):
        self.assertEqual(self._manifest(fold_seeds=3)["folding"]["seeds"], [0, 1, 2])


class TestResultParsing(unittest.TestCase):
    def test_eval_summary_coerces_numbers_and_counts_all_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = Path(tmp)
            (camp / "04_eval").mkdir()
            (camp / "04_eval" / "evaluation_summary.csv").write_text(
                "design,motif_rmsd,plddt,note\n"
                "camp_0000,0.42,88.1,ok\n"
                "camp_0001,1.90,71.0,\n"
            )
            rows, total = dd._read_eval_summary(camp, max_rows=1)
            self.assertEqual(total, 2)          # total is the full count...
            self.assertEqual(len(rows), 1)      # ...even when rows are capped
            self.assertAlmostEqual(rows[0]["motif_rmsd"], 0.42)
            self.assertEqual(rows[0]["note"], "ok")

    def test_eval_summary_drops_empty_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = Path(tmp)
            (camp / "04_eval").mkdir()
            (camp / "04_eval" / "evaluation_summary.csv").write_text(
                "design,motif_rmsd\ncamp_0000,\n"
            )
            rows, _ = dd._read_eval_summary(camp)
            self.assertNotIn("motif_rmsd", rows[0])

    def test_missing_summary_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(dd._read_eval_summary(Path(tmp)), ([], 0))

    def test_generated_count_excludes_the_motif_sidecar(self):
        """Stage 1 writes <prefix>.cif and <prefix>_motif.cif; only the first is
        a design. Counting both doubles the count and halves per-design timings."""
        with tempfile.TemporaryDirectory() as tmp:
            camp = Path(tmp)
            (camp / "01_generation").mkdir()
            for n in ("c_0000.cif", "c_0000_motif.cif", "c_0001.cif", "c_0001_motif.cif"):
                (camp / "01_generation" / n).write_text("x")
            self.assertEqual(dd._collect_designs(camp)["generated"], 2)

    def test_artifact_counts_per_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = Path(tmp)
            for sub, name in (
                ("01_generation", "c_0000.cif"),
                ("01_generation", "c_0000_motif.cif"),
                ("01_generation", "c_0001.cif"),
                ("02_resequence", "c_0000_seq00_reseq.cif"),
                ("03_folded", "c_0000_seq00_complex_folded_seed0.cif"),
                ("04_eval", "c_0000_seq00_evaluation.json"),
            ):
                (camp / sub).mkdir(exist_ok=True)
                (camp / sub / name).write_text("x")
            self.assertEqual(
                dd._collect_designs(camp),
                {"generated": 2, "resequenced": 1, "folded": 1, "evaluated": 1},
            )


class TestGpuDiscipline(unittest.TestCase):
    """The node is shared: AP Novo must only ever see reserved devices."""

    def test_free_gpu_indices_only_takes_this_host(self):
        listing = "lambda2 3\nlambda2 5\nlambda7 0\nbogus\n"
        with mock.patch.object(dd.socket, "gethostname", return_value="lambda2.example"), \
             mock.patch.object(dd.subprocess, "run",
                               return_value=mock.Mock(stdout=listing, returncode=0)):
            self.assertEqual(dd._free_gpu_indices(8), ["3", "5"])
            self.assertEqual(dd._free_gpu_indices(1), ["3"])

    def test_acquire_skips_devices_lost_to_a_race(self):
        # gpu 3 is claimed by someone else between `list` and `acquire`; 5 works.
        def fake_run(cmd, **kw):
            if cmd[1] == "acquire":
                return mock.Mock(returncode=0 if cmd[2] == "5" else 1, stdout="", stderr="")
            return mock.Mock(returncode=0, stdout="lambda2 3\nlambda2 5\n")

        with mock.patch.object(dd.socket, "gethostname", return_value="lambda2"), \
             mock.patch.object(dd.subprocess, "run", side_effect=fake_run):
            self.assertEqual(dd._acquire_gpus(1, "t"), ["5"])

    def test_acquire_records_this_process_as_lock_owner(self):
        seen = []

        def fake_run(cmd, **kw):
            if cmd[1] == "acquire":
                seen.append(cmd)
                return mock.Mock(returncode=0, stdout="", stderr="")
            return mock.Mock(returncode=0, stdout="lambda2 0\n")

        with mock.patch.object(dd.socket, "gethostname", return_value="lambda2"), \
             mock.patch.object(dd.subprocess, "run", side_effect=fake_run), \
             mock.patch.object(dd.os, "getpid", return_value=4242):
            dd._acquire_gpus(1, "lbl")
        self.assertIn("--pid", seen[0])
        self.assertEqual(seen[0][seen[0].index("--pid") + 1], "4242")

    def test_no_free_gpu_means_no_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(dd, "_readiness",
                                   return_value={"ready": True, "blocking": [], "optional": [],
                                                 "paths": {}}), \
                 mock.patch.object(dd, "_local_gpu_capability", return_value=9.0), \
                 mock.patch.object(dd, "_acquire_gpus", return_value=[]), \
                 mock.patch.object(dd.subprocess, "run") as run:
                out = dd.design_de_novo_enzyme(
                    motif_cif_path=str(_motif_cif(tmp)),
                    motif_str="A1/B1", run_dir=str(tmp),
                )
            self.assertEqual(out["status"], "error")
            self.assertIn("No free GPU", out["summary"])
            run.assert_not_called()
            # The manifest is still written, so a retry is free.
            self.assertTrue(Path(out["data"]["manifest"]).exists())

    def _run_with_fake_pipeline(self, tmp: Path, returncode=0, raises=None, generated=1):
        """Drive the tool with a stubbed pipeline; returns (result, popen_kwargs)."""
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            captured["env"] = kw.get("env", {})
            if raises is not None:
                raise raises
            camp = Path([c for c in cmd if c.startswith("--output_dir=")][0].split("=", 1)[1])
            (camp / "01_generation").mkdir(parents=True, exist_ok=True)
            for i in range(generated):
                (camp / "01_generation" / f"d_{i:04d}.cif").write_text("x")
            return mock.Mock(returncode=returncode, stdout="", stderr="boom")

        with mock.patch.object(dd, "_readiness",
                               return_value={"ready": True, "blocking": [], "optional": [],
                                             "paths": {}}), \
             mock.patch.object(dd, "_local_gpu_capability", return_value=9.0), \
             mock.patch.object(dd, "_acquire_gpus", return_value=["2", "6"]), \
             mock.patch.object(dd, "_release_gpus") as release, \
             mock.patch.object(dd.subprocess, "run", side_effect=fake_run):
            out = dd.design_de_novo_enzyme(
                motif_cif_path=str(_motif_cif(tmp)),
                motif_str="A1/B1", num_gpus=2, resequence=False, run_dir=str(tmp),
            )
        captured["released"] = release.call_args[0][0] if release.called else None
        return out, captured

    def test_pins_cuda_visible_devices_to_reserved_set_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cap = self._run_with_fake_pipeline(Path(tmp))
            self.assertEqual(out["status"], "success")
            # Exactly the reserved devices -- otherwise AP Novo forks a worker
            # onto every GPU on the node.
            self.assertEqual(cap["env"]["CUDA_VISIBLE_DEVICES"], "2,6")
            self.assertEqual(cap["env"]["OMP_NUM_THREADS"],
                             str(dd.config.APNOVO_THREADS_PER_WORKER))
            self.assertEqual(cap["released"], ["2", "6"])

    def test_gpus_released_when_pipeline_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cap = self._run_with_fake_pipeline(Path(tmp), returncode=1)
            self.assertEqual(out["status"], "error")
            self.assertEqual(cap["released"], ["2", "6"])

    def test_gpus_released_when_launch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cap = self._run_with_fake_pipeline(Path(tmp), raises=OSError("no exec"))
            self.assertEqual(out["status"], "error")
            self.assertEqual(cap["released"], ["2", "6"])

    def test_cpu_only_stages_reserve_no_gpu(self):
        """Stages 2 and 4 run on CPU -- holding a GPU for them starves the node."""
        for stage in ("resequence", "evaluation"):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                captured = {}

                def fake_run(cmd, **kw):
                    captured["env"] = kw.get("env", {})
                    camp = Path([c for c in cmd if c.startswith("--output_dir=")][0]
                                .split("=", 1)[1])
                    (camp / "01_generation").mkdir(parents=True, exist_ok=True)
                    (camp / "01_generation" / "d.cif").write_text("x")
                    return mock.Mock(returncode=0, stdout="", stderr="")

                with mock.patch.object(dd, "_readiness",
                                       return_value={"ready": True, "blocking": [],
                                                     "optional": [], "paths": {}}), \
                     mock.patch.object(dd, "_acquire_gpus") as acq, \
                     mock.patch.object(dd.subprocess, "run", side_effect=fake_run):
                    out = dd.design_de_novo_enzyme(
                        motif_cif_path=str(_motif_cif(tmp)), motif_str="A1/B1",
                        only_stage=stage, resequence=False, run_dir=str(tmp),
                    )
                acq.assert_not_called()
                self.assertEqual(out["status"], "success")
                self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertIn("on CPU", out["summary"])

    def test_gpu_stages_still_require_a_reservation(self):
        """A CPU-only carve-out must not leak into the GPU stages."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(dd, "_readiness",
                                   return_value={"ready": True, "blocking": [],
                                                 "optional": [], "paths": {}}), \
                 mock.patch.object(dd, "_local_gpu_capability", return_value=9.0), \
                 mock.patch.object(dd, "_acquire_gpus", return_value=[]), \
                 mock.patch.object(dd.subprocess, "run") as run:
                for stage in ("", "generation", "folding"):
                    out = dd.design_de_novo_enzyme(
                        motif_cif_path=str(_motif_cif(tmp)), motif_str="A1/B1",
                        only_stage=stage, resequence=False, run_dir=str(tmp),
                    )
                    self.assertEqual(out["status"], "error")
                    self.assertIn("No free GPU", out["summary"])
                run.assert_not_called()

    def test_num_gpus_is_clamped_to_the_cluster_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(dd, "_readiness",
                                   return_value={"ready": True, "blocking": [], "optional": [],
                                                 "paths": {}}), \
                 mock.patch.object(dd, "_local_gpu_capability", return_value=9.0), \
                 mock.patch.object(dd, "_acquire_gpus", return_value=[]) as acq:
                dd.design_de_novo_enzyme(motif_cif_path=str(_motif_cif(tmp)),
                                         motif_str="A1/B1", num_gpus=99, run_dir=str(tmp))
                self.assertEqual(acq.call_args[0][0], dd.config.APNOVO_MAX_GPUS)
                dd.design_de_novo_enzyme(motif_cif_path=str(_motif_cif(tmp)),
                                         motif_str="A1/B1", num_gpus=0, run_dir=str(tmp))
                self.assertEqual(acq.call_args[0][0], 1)


class TestSubprocessEnv(unittest.TestCase):
    def test_pins_devices_and_caps_threads(self):
        with mock.patch.dict(dd.os.environ, {}, clear=True):
            env = dd._subprocess_env(["2", "5"])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,5")
        self.assertEqual(env["OMP_NUM_THREADS"], str(dd.config.APNOVO_THREADS_PER_WORKER))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,5")

    def test_cpu_only_gets_empty_device_list(self):
        with mock.patch.dict(dd.os.environ, {}, clear=True):
            self.assertEqual(dd._subprocess_env([])["CUDA_VISIBLE_DEVICES"], "")

    def test_system_cuda_is_stripped_from_library_path(self):
        """A node's system CUDA shadows the venv's wheels -> JAX silently drops
        to CPU and the folding stage dies with "Not supported on cpu"."""
        with mock.patch.dict(
            dd.os.environ,
            {"LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:/opt/mylib:/usr/local/CUDA/lib"},
            clear=True,
        ):
            env = dd._subprocess_env(["0"])
        self.assertEqual(env["LD_LIBRARY_PATH"], "/opt/mylib")

    def test_library_path_dropped_when_only_cuda_entries(self):
        with mock.patch.dict(
            dd.os.environ, {"LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:"}, clear=True
        ):
            self.assertNotIn("LD_LIBRARY_PATH", dd._subprocess_env(["0"]))

    def test_non_cuda_library_path_untouched(self):
        with mock.patch.dict(
            dd.os.environ, {"LD_LIBRARY_PATH": "/opt/a:/opt/b"}, clear=True
        ):
            self.assertEqual(dd._subprocess_env(["0"])["LD_LIBRARY_PATH"], "/opt/a:/opt/b")


class TestFoldingHardwareGate(unittest.TestCase):
    """AF3's attention kernels need Ampere+; V100/T4 hosts must fail fast."""

    def test_folding_scope_detection(self):
        # Whole pipeline: folding happens if any state was requested.
        self.assertTrue(dd._folding_in_scope("", True, []))
        self.assertTrue(dd._folding_in_scope("", False, [{"name": "c", "ligands": []}]))
        self.assertFalse(dd._folding_in_scope("", False, []))
        # Single-stage runs: only the folding stage itself needs the GPU arch.
        self.assertTrue(dd._folding_in_scope("folding", False, []))
        for stage in ("generation", "resequence", "evaluation"):
            self.assertFalse(dd._folding_in_scope(stage, True, [{"name": "c"}]))

    def test_capability_parsing(self):
        with mock.patch.object(dd.subprocess, "run",
                               return_value=mock.Mock(stdout="7.0\n7.0\n9.0\n", returncode=0)):
            self.assertEqual(dd._local_gpu_capability(), 9.0)
        with mock.patch.object(dd.subprocess, "run",
                               return_value=mock.Mock(stdout="", returncode=0)):
            self.assertIsNone(dd._local_gpu_capability())
        with mock.patch.object(dd.subprocess, "run", side_effect=OSError):
            self.assertIsNone(dd._local_gpu_capability())

    def _attempt(self, cap, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(dd, "_readiness",
                                   return_value={"ready": True, "blocking": [],
                                                 "optional": [], "paths": {}}), \
                 mock.patch.object(dd, "_local_gpu_capability", return_value=cap), \
                 mock.patch.object(dd, "_acquire_gpus", return_value=[]) as acq:
                out = dd.design_de_novo_enzyme(
                    motif_cif_path=str(_motif_cif(tmp)), motif_str="A1/B1",
                    resequence=False, run_dir=str(tmp), **kw)
            return out, acq

    def test_folding_refused_on_too_old_gpu_before_any_work(self):
        out, acq = self._attempt(7.0, fold_apo_monomer=True)
        self.assertEqual(out["status"], "error")
        self.assertIn("7.0", out["summary"])
        acq.assert_not_called()   # nothing reserved, nothing generated

    def test_generation_only_still_allowed_on_old_gpu(self):
        out, acq = self._attempt(7.0, fold_apo_monomer=False, only_stage="generation")
        # Gets past the hardware gate and on to reserving a GPU.
        acq.assert_called()
        self.assertIn("No free GPU", out["summary"])

    def test_folding_allowed_on_new_gpu_and_when_undetectable(self):
        for cap in (9.0, None):
            out, acq = self._attempt(cap, fold_apo_monomer=True)
            acq.assert_called()
            self.assertIn("No free GPU", out["summary"])


class TestGuardsAndSummaries(unittest.TestCase):
    def test_missing_motif_file_is_rejected_before_reserving(self):
        with mock.patch.object(dd, "_readiness",
                               return_value={"ready": True, "blocking": [], "optional": [],
                                             "paths": {}}), \
             mock.patch.object(dd, "_acquire_gpus") as acq:
            out = dd.design_de_novo_enzyme(motif_cif_path="/nope/missing.cif", motif_str="A1/B1")
        self.assertEqual(out["status"], "error")
        acq.assert_not_called()

    def test_blank_motif_str_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(dd, "_readiness",
                                   return_value={"ready": True, "blocking": [], "optional": [],
                                                 "paths": {}}), \
                 mock.patch.object(dd, "_acquire_gpus") as acq:
                out = dd.design_de_novo_enzyme(
                    motif_cif_path=str(_motif_cif(Path(tmp))), motif_str="   ")
            self.assertEqual(out["status"], "error")
            acq.assert_not_called()

    def test_resequence_without_ligandmpnn_is_refused_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                dd, "_readiness",
                return_value={"ready": True, "blocking": [],
                              "optional": ["LigandMPNN env missing -- ..."], "paths": {}}), \
                 mock.patch.object(dd, "_acquire_gpus") as acq:
                out = dd.design_de_novo_enzyme(
                    motif_cif_path=str(_motif_cif(Path(tmp))),
                    motif_str="A1/B1", resequence=True)
            self.assertEqual(out["status"], "error")
            self.assertIn("LigandMPNN", out["summary"])
            acq.assert_not_called()

    def test_unready_backend_reports_blocking_reasons(self):
        with mock.patch.object(dd, "_readiness",
                               return_value={"ready": False, "blocking": ["python missing at /x"],
                                             "optional": [], "paths": {}}):
            out = dd.design_de_novo_enzyme(motif_cif_path="/any.cif", motif_str="A1/B1")
        self.assertEqual(out["status"], "error")
        self.assertIn("python missing", out["summary"])

    def test_summarize_reads_campaign_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            camp = Path(tmp) / "camp"
            (camp / "01_generation").mkdir(parents=True)
            (camp / "01_generation" / "c_0000.cif").write_text("x")
            (camp / "pipeline_index.json").write_text(
                json.dumps({"completed_stages": ["generation"]}))
            out = dd.summarize_de_novo_campaign(str(camp))
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["data"]["artifact_counts"]["generated"], 1)
        self.assertEqual(out["data"]["completed_stages"], ["generation"])

    def test_summarize_rejects_missing_and_empty_campaigns(self):
        self.assertEqual(dd.summarize_de_novo_campaign("/nope/none")["status"], "error")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(dd.summarize_de_novo_campaign(tmp)["status"], "error")

    def test_readiness_flags_missing_pieces(self):
        with mock.patch.object(dd.Path, "exists", return_value=False):
            info = dd._readiness()
        self.assertFalse(info["ready"])
        self.assertTrue(info["blocking"])


if __name__ == "__main__":
    unittest.main()


class TestStagePlan(unittest.TestCase):
    """Which stages the wrapper actually asks run_pipeline.py to run.

    Regression for a real failure (2026-10-06, campaign pmd_clamp6_v2): asking for
    generation + resequencing only still ran the AF3 folding stage, because
    `run_pipeline.py` has --only_stage and --from_stage but no "stop after" flag and
    its `_should_run` returns True for FOLDING on a default invocation. Generation
    (24/24) and resequencing (96/96) both completed, then AF3 raised
    NotImplementedError on the V100 and the whole call returned rc=1.
    """

    def _stage_flags(self, tmp: Path, **kwargs) -> list[list[str]]:
        """The --only_stage values passed across all subprocess invocations."""
        seen: list[list[str]] = []

        def fake_run(cmd, **_):
            seen.append([c for c in cmd if c.startswith("--only_stage=")])
            camp = Path([c for c in cmd if c.startswith("--output_dir=")][0].split("=", 1)[1])
            (camp / "01_generation").mkdir(parents=True, exist_ok=True)
            (camp / "01_generation" / "d_0000.cif").write_text("x")
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(dd, "_readiness",
                               return_value={"ready": True, "blocking": [], "optional": [],
                                             "paths": {}}), \
             mock.patch.object(dd, "_local_gpu_capability", return_value=9.0), \
             mock.patch.object(dd, "_acquire_gpus", return_value=["0"]), \
             mock.patch.object(dd, "_release_gpus"), \
             mock.patch.object(dd.subprocess, "run", side_effect=fake_run):
            dd.design_de_novo_enzyme(
                motif_cif_path=str(_motif_cif(tmp)),
                motif_str="A1/B1", run_dir=str(tmp), **kwargs)
        return seen

    def test_no_folding_requested_means_folding_is_never_invoked(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = self._stage_flags(Path(tmp), fold_apo_monomer=False,
                                     folding_states=[], resequence=False)
        self.assertEqual(seen, [["--only_stage=generation"]])

    def test_resequencing_without_folding_runs_exactly_two_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = self._stage_flags(Path(tmp), fold_apo_monomer=False,
                                     folding_states=[], resequence=True)
        self.assertEqual(seen, [["--only_stage=generation"],
                                ["--only_stage=resequence"]])
        self.assertNotIn(["--only_stage=folding"], seen)

    def test_folding_requested_still_uses_one_default_invocation(self):
        # When folding IS wanted the pipeline's own ordering is left alone, so a
        # single unrestricted call must remain the behaviour.
        with tempfile.TemporaryDirectory() as tmp:
            seen = self._stage_flags(Path(tmp), fold_apo_monomer=True,
                                     folding_states=[], resequence=False)
        self.assertEqual(seen, [[]])

    def test_explicit_only_stage_is_passed_through_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = self._stage_flags(Path(tmp), only_stage="resequence")
        self.assertEqual(seen, [["--only_stage=resequence"]])
