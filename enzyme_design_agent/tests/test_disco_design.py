"""Offline tests for the DISCO design tool.

These never launch DISCO or touch a GPU: the subprocess boundary and the
reservation gate are mocked, so what is under test is the wrapper's own logic --
ligand/input-JSON translation, GPU pinning, release-on-every-exit-path, and
result parsing.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from enzyme_design_agent.tools import disco_design as dc


class TestLigandSpec(unittest.TestCase):
    def test_smiles_passes_through_raw(self):
        self.assertEqual(dc._ligand_spec({"smiles": "c1ccccc1"}), "c1ccccc1")

    def test_ccd_code_is_prefixed(self):
        self.assertEqual(dc._ligand_spec({"ccd_code": "ATP"}), "CCD_ATP")

    def test_sdf_path_is_prefixed_and_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "lig.sdf"
            p.write_text("x")
            spec = dc._ligand_spec({"sdf_path": str(p)})
            self.assertTrue(spec.startswith("FILE_"))
            self.assertTrue(Path(spec[5:]).is_absolute())

    def test_precedence_file_over_ccd_over_smiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "lig.sdf"
            p.write_text("x")
            spec = dc._ligand_spec({"sdf_path": str(p), "ccd_code": "ATP", "smiles": "C"})
            self.assertTrue(spec.startswith("FILE_"))

    def test_empty_ligand_returns_none(self):
        self.assertIsNone(dc._ligand_spec({}))
        self.assertIsNone(dc._ligand_spec({"smiles": "", "ccd_code": "  "}))


class TestInputJobs(unittest.TestCase):
    def test_one_job_per_length_with_masked_chain(self):
        jobs = dc._build_input_jobs(slug="camp", lengths=[150, 200], ligand_specs=["CCD_ATP"])
        self.assertEqual([j["name"] for j in jobs], ["camp_len150", "camp_len200"])
        # the protein chain is fully masked to exactly the requested length
        chain = jobs[0]["sequences"][0]["proteinChain"]
        self.assertEqual(chain["sequence"], "-" * 150)
        self.assertEqual(set(jobs[0]["sequences"][0]["proteinChain"]["sequence"]), {"-"})

    def test_every_ligand_becomes_its_own_entry(self):
        jobs = dc._build_input_jobs(
            slug="c", lengths=[100], ligand_specs=["CCD_ATP", "CCD_MG", "OC(=O)CCCCC(=O)O"]
        )
        seqs = jobs[0]["sequences"]
        # one protein chain + three ligand entries, in order
        self.assertEqual(len(seqs), 4)
        self.assertIn("proteinChain", seqs[0])
        self.assertEqual([s["ligand"]["ligand"] for s in seqs[1:]],
                         ["CCD_ATP", "CCD_MG", "OC(=O)CCCCC(=O)O"])

    def test_unconditional_has_no_ligand_entries(self):
        jobs = dc._build_input_jobs(slug="u", lengths=[70], ligand_specs=[])
        self.assertEqual(len(jobs[0]["sequences"]), 1)
        self.assertIn("proteinChain", jobs[0]["sequences"][0])


class TestResultParsing(unittest.TestCase):
    def _dump(self, tmp: Path, n_designs=2, n_errors=0):
        pdbs = tmp / "pdbs"
        seqs = tmp / "sequences"
        pdbs.mkdir(parents=True)
        seqs.mkdir(parents=True)
        for i in range(n_designs):
            stem = f"camp_len200_sample_{i}"
            (pdbs / f"{stem}.pdb").write_text("ATOM\n")
            (seqs / f"{stem}.txt").write_text(f">cogen_seq 0\nACDEFGHIKL\nligand_smiles CCO\n")
            (pdbs / f"{stem}_ligands.txt").write_text("ligand_smiles CCO\n")
        if n_errors:
            err = tmp / "ERR"
            err.mkdir()
            for j in range(n_errors):
                (err / f"camp_len300_{j}.txt").write_text("boom traceback")
        return tmp

    def test_counts_designs_and_reads_sequence_and_ligands(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = self._dump(Path(tmp), n_designs=2)
            rows, n, n_err = dc._collect_designs(dump)
            self.assertEqual((n, n_err), (2, 0))
            self.assertEqual(rows[0]["sequence"], "ACDEFGHIKL")
            self.assertEqual(rows[0]["sequence_length"], 10)
            self.assertEqual(rows[0]["ligands"], ["CCO"])

    def test_row_cap_keeps_total_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = self._dump(Path(tmp), n_designs=5)
            rows, n, _ = dc._collect_designs(dump, max_rows=2)
            self.assertEqual(n, 5)          # full count...
            self.assertEqual(len(rows), 2)  # ...even when rows are capped

    def test_errors_counted_and_tailed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = self._dump(Path(tmp), n_designs=1, n_errors=2)
            _, _, n_err = dc._collect_designs(dump)
            self.assertEqual(n_err, 2)
            tails = dc._error_tail(dump)
            self.assertEqual(len(tails), 2)
            self.assertTrue(all("boom" in v for v in tails.values()))

    def test_read_sequence_stops_at_second_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.txt"
            p.write_text(">cogen_seq 0\nACDE\n>cogen_seq 1\nWWWW\n")
            self.assertEqual(dc._read_sequence(p), "ACDE")


class TestGpuDiscipline(unittest.TestCase):
    """The node is shared: DISCO must only ever see its one reserved device."""

    def test_free_gpu_indices_only_takes_this_host(self):
        listing = "lambda2 3\nlambda2 5\nlambda7 0\nbogus\n"
        with mock.patch.object(dc.socket, "gethostname", return_value="lambda2.x"), \
             mock.patch.object(dc.subprocess, "run",
                               return_value=mock.Mock(stdout=listing, returncode=0)):
            self.assertEqual(dc._free_gpu_indices(8), ["3", "5"])

    def test_acquire_records_pid_and_skips_lost_race(self):
        def fake_run(cmd, **kw):
            if cmd[1] == "acquire":
                return mock.Mock(returncode=0 if cmd[2] == "5" else 1, stdout="", stderr="")
            return mock.Mock(returncode=0, stdout="lambda2 3\nlambda2 5\n")
        with mock.patch.object(dc.socket, "gethostname", return_value="lambda2"), \
             mock.patch.object(dc.subprocess, "run", side_effect=fake_run), \
             mock.patch.object(dc.os, "getpid", return_value=4242):
            self.assertEqual(dc._acquire_gpus(1, "t"), ["5"])

    def test_no_free_gpu_means_no_launch_but_input_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(dc, "_readiness",
                                   return_value={"ready": True, "blocking": [], "optional": [],
                                                 "paths": {}}), \
                 mock.patch.object(dc, "_local_gpu_capability", return_value=7.0), \
                 mock.patch.object(dc, "_acquire_gpus", return_value=[]), \
                 mock.patch.object(dc.subprocess, "run") as run:
                out = dc.design_enzyme_with_disco(
                    ligands=[{"ccd_code": "ATP"}], lengths=[150], run_dir=str(tmp),
                )
            self.assertEqual(out["status"], "error")
            self.assertIn("No free GPU", out["summary"])
            run.assert_not_called()
            self.assertTrue(Path(out["data"]["input_json"]).exists())

    def _run_with_fake_disco(self, tmp: Path, returncode=0, raises=None, n_designs=1):
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            captured["env"] = kw.get("env", {})
            if raises is not None:
                raise raises
            dump = Path([c for c in cmd if c.startswith("dump_dir=")][0].split("=", 1)[1])
            pdbs = dump / "pdbs"
            seqs = dump / "sequences"
            pdbs.mkdir(parents=True, exist_ok=True)
            seqs.mkdir(parents=True, exist_ok=True)
            for i in range(n_designs):
                stem = f"camp_len150_sample_{i}"
                (pdbs / f"{stem}.pdb").write_text("ATOM\n")
                (seqs / f"{stem}.txt").write_text(">cogen_seq 0\nACDE\n")
            return mock.Mock(returncode=returncode, stdout="", stderr="boom")

        with mock.patch.object(dc, "_readiness",
                               return_value={"ready": True, "blocking": [], "optional": [],
                                             "paths": {}}), \
             mock.patch.object(dc, "_local_gpu_capability", return_value=7.0), \
             mock.patch.object(dc, "_acquire_gpus", return_value=["4"]), \
             mock.patch.object(dc, "_release_gpus") as release, \
             mock.patch.object(dc.subprocess, "run", side_effect=fake_run):
            out = dc.design_enzyme_with_disco(
                ligands=[{"ccd_code": "ATP"}], lengths=[150], num_seeds=1,
                campaign_name="camp", run_dir=str(tmp),
            )
        captured["released"] = release.call_args[0][0] if release.called else None
        return out, captured

    def test_pins_single_reserved_device_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cap = self._run_with_fake_disco(Path(tmp))
            self.assertEqual(out["status"], "success")
            self.assertEqual(cap["env"]["CUDA_VISIBLE_DEVICES"], "4")
            self.assertEqual(cap["released"], ["4"])

    def test_released_when_disco_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cap = self._run_with_fake_disco(Path(tmp), returncode=1, n_designs=0)
            self.assertEqual(out["status"], "error")
            self.assertEqual(cap["released"], ["4"])

    def test_released_when_launch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, cap = self._run_with_fake_disco(Path(tmp), raises=OSError("no exec"))
            self.assertEqual(out["status"], "error")
            self.assertEqual(cap["released"], ["4"])

    def test_command_carries_presets_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, cap = self._run_with_fake_disco(Path(tmp))
            cmd = cap["cmd"]
            self.assertTrue(any(c == "experiment=diverse" for c in cmd))
            self.assertTrue(any(c == "seeds=[0]" for c in cmd))
            # default host here is sm7.0 -> naive attention
            self.assertIn("use_deepspeed_evo_attention=false", cmd)


class TestFastAttentionGate(unittest.TestCase):
    def test_disabled_without_cutlass_even_on_ampere(self):
        with mock.patch.object(dc.config, "DISCO_CUTLASS_PATH", ""):
            self.assertFalse(dc._fast_attention_available(9.0))

    def test_disabled_below_ampere_even_with_cutlass(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(dc.config, "DISCO_CUTLASS_PATH", tmp):
            self.assertFalse(dc._fast_attention_available(7.0))

    def test_enabled_with_cutlass_on_ampere(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(dc.config, "DISCO_CUTLASS_PATH", tmp):
            self.assertTrue(dc._fast_attention_available(8.0))


class TestValidation(unittest.TestCase):
    def test_bad_ligand_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(dc, "_readiness",
                                   return_value={"ready": True, "blocking": [], "optional": [],
                                                 "paths": {}}):
                out = dc.design_enzyme_with_disco(
                    ligands=[{"bogus": "x"}], lengths=[150], run_dir=str(tmp),
                )
            self.assertEqual(out["status"], "error")
            self.assertIn("none of", out["summary"])

    def test_missing_sdf_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(dc, "_readiness",
                                   return_value={"ready": True, "blocking": [], "optional": [],
                                                 "paths": {}}):
                out = dc.design_enzyme_with_disco(
                    ligands=[{"sdf_path": "/no/such/file.sdf"}], lengths=[150], run_dir=str(tmp),
                )
            self.assertEqual(out["status"], "error")
            self.assertIn("not found", out["summary"])


if __name__ == "__main__":
    unittest.main()
