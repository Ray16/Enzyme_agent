"""Guard sample-batch repair without initializing GPUs or changing an environment."""
import pytest
from structural_modeling_agent.cofold.boltz_serial_predict import serial_sampler_source


def test_known_batching_defect_is_repaired_and_fixed_source_is_idempotent():
    original='for batch in sample_ids.chunk(multiplicity % max_parallel_samples + 1):\n    pass'
    fixed,changed=serial_sampler_source(original)
    assert changed and 'sample_ids.split(max_parallel_samples)' in fixed
    assert serial_sampler_source(fixed)==(fixed,False)


def test_unknown_batching_source_is_not_patched_speculatively():
    with pytest.raises(RuntimeError,match='Unrecognized'):
        serial_sampler_source('for batch in undocumented_batcher(sample_ids): pass')
