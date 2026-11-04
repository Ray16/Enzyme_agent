"""Mixed CUDA/XPU provenance, references and split-position regression checks."""
import hashlib
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

EXT=Path(__file__).resolve().parents[1]/"campaigns/amide_interface_20261010/substrate_extension_20261026"
sys.path.insert(0,str(EXT))
import merge_analysis as merger
import audit_completion as audit
import validate_WT_references as validation
import campaign_figures as figures


class MixedBackendExtensionTest(unittest.TestCase):
    def test_split_position_references_and_permutation(self):
        result=validation.validate_mixed_fixture(20261010)
        self.assertEqual(result["substitutions_per_position"],19)
        self.assertEqual(result["backend_cells"],4)

    def test_legacy_cuda_attempt_and_aurora_remote_input_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            here=Path(directory);(here/"inputs").mkdir();(here/"logs").mkdir()
            ident="synthetic_A1C";config=here/"inputs"/(ident+".yaml")
            config.write_text('{"sequence":"AC","msa":"/local/query.csv"}')
            local_hash=hashlib.sha256(config.read_bytes()).hexdigest()
            argv=["python","predict",str(config)]
            for flag,value in {"--diffusion_samples":"50","--max_parallel_samples":"1","--seed":"20261010",
                "--recycling_steps":"3","--sampling_steps":"200","--num_workers":"0","--preprocessing-threads":"1"}.items():
                argv.extend([flag,value])
            attempt=dict(complete=True,count=1,argv=argv,input_sha256=local_hash)
            log=here/"logs"/(ident+".attempt1.log")
            log.write_text("PREDICTION_PRECISION 32-true\nPROCESS_LOCAL_SAMPLE_CAP source_sha256=abc\n")
            self.assertEqual(audit.audit_attempt(ident,attempt,local_hash,20261010,here),
                             ("Lambda_CUDA_FP32","Lambda fleet"))
            remote_hash="f"*64
            aurora=dict(attempt,backend="Aurora_XPU_FP32",provider="Aurora XPU FP32",
                        input_sha256=remote_hash,local_equivalent_input_sha256=local_hash,
                        auxiliary_status="diagnostic_warning",confidence_warnings=["optional NPZ unavailable"])
            marker=dict(precision="FP32",autocast=False,sample_chunking="split(max_parallel_samples)")
            log.write_text("AURORA_IMPLEMENTATION "+json.dumps(marker)+"\nAURORA_FP32_INFERENCE autocast_disabled=True\n")
            self.assertEqual(audit.audit_attempt(ident,aurora,local_hash,20261010,here),
                             ("Aurora_XPU_FP32","Aurora XPU FP32"))
            with self.assertRaises(AssertionError):
                audit.audit_attempt(ident,dict(aurora,local_equivalent_input_sha256="0"*64),local_hash,20261010,here)
            with self.assertRaises(AssertionError):
                audit.audit_attempt(ident,dict(aurora,backend="unknown"),local_hash,20261010,here)
            marker["autocast"]=True
            log.write_text("AURORA_IMPLEMENTATION "+json.dumps(marker)+"\nAURORA_FP32_INFERENCE autocast_disabled=True\n")
            with self.assertRaises(AssertionError):
                audit.audit_attempt(ident,aurora,local_hash,20261010,here)

    def test_reference_audit_uses_backend_and_rejects_corrupted_control(self):
        values={"candidate_WT":(100.,2.)}
        controls={"candidate":dict(n_models=50,poses=[[2.,110.]]*50,
                                   angle_deg=110.,harmonic_distance_A=2.)}
        row=dict(candidate="candidate",backend="Aurora_XPU_FP32")
        self.assertEqual(audit.reference_metric(row,values,controls),((110.,2.),"Aurora_XPU_FP32"))
        self.assertEqual(audit.reference_metric(dict(row,backend="Lambda_CUDA_FP32"),values,controls),
                         ((100.,2.),"legacy_CUDA_FP32"))
        controls["candidate"]["angle_deg"]=111.
        with self.assertRaises(AssertionError):audit.reference_metric(row,values,controls)

    def test_merge_keeps_full_tree_actual_backend_and_cache_input_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            here=Path(directory);canonical=here/"canonical"
            ident="synthetic_A1C";prediction=here/"results"/ident/("boltz_results_"+ident)/"predictions"/ident
            prediction.mkdir(parents=True)
            for sample in range(50):
                (prediction/f"{ident}_model_{sample}.pdb").write_text(f"pose {sample}\n")
                (prediction/f"confidence_{ident}_model_{sample}.json").write_text(json.dumps({"sample":sample}))
            processed=prediction.parent.parent/"processed"/"mols";processed.mkdir(parents=True)
            (processed/"source.bin").write_bytes(b"preserve predictor-produced ligand evidence")
            (here/"attempts").mkdir()
            (here/"inputs").mkdir();config=here/"inputs"/(ident+".yaml");config.write_text('{"sequence":"AC"}')
            actual_hash="a"*64;local_hash=hashlib.sha256(config.read_bytes()).hexdigest()
            attempt=dict(complete=True,backend="Aurora_XPU_FP32",provider="Aurora XPU FP32",
                         input_sha256=actual_hash,local_equivalent_input_sha256=local_hash,argv=["actual","remote"],count=1,
                         auxiliary_status={"npz":"warning"},confidence_warnings=["optional auxiliary array is nonfinite"])
            (here/"attempts"/(ident+".json")).write_text(json.dumps(attempt))
            job=dict(id=ident,candidate="candidate",mutation="A1C",sequence="AC",
                     sequence_sha256=hashlib.sha256(b"AC").hexdigest(),expected_components={},geometry={})
            qc=dict(n_models=50,poses=[[2.,110.]]*50,angle_deg=110.,harmonic_distance_A=2.,qc="fixture QC")
            with patch.object(merger,"HERE",here),patch.object(merger,"BASE",canonical),patch.object(merger,"load_cached_qc",return_value=qc) as cached,patch.object(merger,"verify",return_value=qc) as verify:
                receipt=merger.merge_one(job)
                verify.assert_not_called()
                self.assertEqual(cached.call_args.args[1],actual_hash)
                self.assertEqual(receipt["backend"],"Aurora_XPU_FP32")
                self.assertEqual(receipt["input_sha256"],actual_hash)
                self.assertEqual(receipt["local_equivalent_input_sha256"],local_hash)
                self.assertEqual(receipt["auxiliary_status"],attempt["auxiliary_status"])
                self.assertEqual(receipt["confidence_warnings"],attempt["confidence_warnings"])
                self.assertEqual(merger.auxiliary_diagnostics({ident:receipt})[0]["confidence_warnings"],attempt["confidence_warnings"])
                dest=canonical/"results"/ident/("boltz_results_"+ident)
                self.assertTrue((dest/"processed/mols/source.bin").samefile(processed/"source.bin"))
                verified=json.loads((canonical/"results"/ident/"verified.json").read_text())
                self.assertEqual(verified["provider"],"Aurora XPU FP32")
                self.assertEqual(verified["auxiliary_status"],attempt["auxiliary_status"])
                self.assertEqual(len(verified["retained_result_files_sha256"]),101)
                merger.merge_one(job)
                self.assertEqual(verify.call_count,1)
                collision=dest/"predictions"/ident/f"confidence_{ident}_model_0.json"
                changed=collision.with_suffix(".changed");changed.write_text('{"different":true}');changed.replace(collision)
                with self.assertRaisesRegex(AssertionError,"refusing overwrite"):merger.merge_one(job)

    def test_unknown_variant_backend_fails_closed(self):
        fixture,arrays,backends,references=validation.mixed_fixture(20261010)
        backends[fixture["jobs"][1]["id"]]="unknown"
        with self.assertRaises(AssertionError):
            merger.score_arrays(fixture,arrays,backend_references=references,backend_by_id=backends)

    def test_full_figure_pipeline_handles_split_sites(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign=Path(directory);extension=campaign/"extension";out=campaign/"figures"
            extension.mkdir()
            records=[];controls=[]
            for candidate,n_positions in [("P0DXD6",21),("P78061",34),("Q46E79",38)]:
                records.append(dict(id=candidate+"_WT",candidate=candidate,mutation="WT",status="evaluated",
                    n_models=50,angle_deg=100.,harmonic_distance_A=2.,delta_angle_deg=0.,backend="legacy_CUDA_FP32"))
                controls.append(dict(candidate=candidate,angle_deg=101.,harmonic_distance_A=2.))
                for position in range(1,n_positions+1):
                    for index,aa in enumerate("CDEFGHIKLMNPQRSTVWY"):
                        aurora=index<7;delta=position/30+(index-9)/10
                        mutation=f"A{position}{aa}"
                        records.append(dict(id=candidate+"_"+mutation,candidate=candidate,mutation=mutation,status="evaluated",
                            n_models=50,angle_deg=100.+delta+aurora,harmonic_distance_A=2.+index/100,
                            delta_angle_deg=delta,backend="Aurora_XPU_FP32" if aurora else "Lambda_CUDA_FP32"))
            (campaign/"manifest.json").write_text(json.dumps(dict(n_variants=1770)))
            (campaign/"pose_geometry.csv").write_text("id,sample,distance_A,angle_deg\n")
            with (campaign/"variant_metrics.csv").open("w") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            (extension/"backend_controls.json").write_text(json.dumps(dict(controls=controls)))
            with patch.object(figures,"HERE",extension),patch.object(figures,"CAMPAIGN",campaign),patch.object(figures,"OUT",out),contextlib.redirect_stdout(io.StringIO()):
                figures.main()
            report=json.loads((out/"expanded_figure_audit.json").read_text())
            self.assertEqual(len(report["permutation"]["cells"]),186)
            self.assertEqual({r["n"] for r in report["permutation"]["position_substitution_counts"]},{19})
            self.assertEqual(sum(report["backend_counts"].values()),1767)
            self.assertEqual(len(report["artifacts_sha256"]),6)
            self.assertTrue(all((out/name).stat().st_size>1000 for name in report["artifacts_sha256"]))


if __name__=="__main__":unittest.main()
