"""Consistent run-size and timing fields, with unmeasured stages left null."""
from __future__ import annotations


def summarize_run(*, process_started, entrypoint_started, sampling_started,
                  sampling_finished, finished, protein_lengths,
                  requested_samples, completed_samples, batch_size):
    def elapsed(start, end):
        if start is None or end is None:
            return None
        if end < start:
            raise ValueError('Run timestamps are out of order')
        return float(end - start)

    entrypoint = elapsed(entrypoint_started, finished)
    lengths = list(protein_lengths)
    return {
        'requested_samples': requested_samples,
        'completed_samples': completed_samples,
        'batch_size': batch_size,
        'protein_lengths_per_design': lengths,
        'protein_length_residues': lengths[0] if lengths and len(set(lengths)) == 1 else None,
        'startup_import_seconds': elapsed(process_started, entrypoint_started),
        'warmup_to_first_batch_seconds': elapsed(process_started, sampling_started),
        'generation_entrypoint_seconds': entrypoint,
        'sampling_and_export_seconds': elapsed(sampling_started, sampling_finished),
        'process_wall_seconds': elapsed(process_started, finished),
        'entrypoint_seconds_per_completed_sample': (
            entrypoint / completed_samples if entrypoint is not None and completed_samples else None
        ),
        'timing_definitions': {
            'startup_import_seconds': 'Process start through imports and config composition.',
            'warmup_to_first_batch_seconds': 'Process start through the first generation batch-start callback; includes model loading.',
            'generation_entrypoint_seconds': 'Entrypoint start through output checks, including model loading.',
            'sampling_and_export_seconds': 'First batch-start callback through return from run_generation, including export and metric overhead.',
            'entrypoint_seconds_per_completed_sample': 'Batch throughput average, not individual sample latency.',
            'queue_wait': 'Excluded; retain scheduler submission/start/end times separately.',
        },
    }
