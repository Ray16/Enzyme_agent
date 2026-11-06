"""Preserve distinct startup and generation measurements, including failed runs."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'run_logging', Path(__file__).parents[1] / 'runs/as9_motif_design_rfd4_proteina/run_logging.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_distinguishes_imports_model_warmup_and_batch_throughput():
    result = module.summarize_run(
        process_started=100, entrypoint_started=120, sampling_started=150,
        sampling_finished=440, finished=450, protein_lengths=[516] * 6,
        requested_samples=6, completed_samples=6, batch_size=6)
    assert result['startup_import_seconds'] == 20
    assert result['warmup_to_first_batch_seconds'] == 50
    assert result['generation_entrypoint_seconds'] == 330
    assert result['process_wall_seconds'] == 350
    assert result['entrypoint_seconds_per_completed_sample'] == 55
    assert result['protein_length_residues'] == 516


def test_failed_run_does_not_invent_latency_or_warmup():
    result = module.summarize_run(
        process_started=100, entrypoint_started=120, sampling_started=None,
        sampling_finished=None, finished=130, protein_lengths=[],
        requested_samples=6, completed_samples=0, batch_size=6)
    assert result['entrypoint_seconds_per_completed_sample'] is None
    assert result['warmup_to_first_batch_seconds'] is None
    assert result['protein_length_residues'] is None
