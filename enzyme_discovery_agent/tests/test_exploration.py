"""Discovery scheduling and auditable sequence-panel regressions."""
import json

import pytest

from enzyme_discovery_agent.claude_subscription import Response, Choice, Message
from enzyme_discovery_agent.exploration import (
    add_record, initial_tranche, nominate, parse_nomination, select_panel, sequence_identity,
    search_from_source_groups,
    conservative_charge,
)
from enzyme_discovery_agent.llm_client import TokenUsage


def response(value):
    return Response([Choice(Message(None, [{'id': 'synthetic', 'type': 'function', 'function': {
        'name': 'propose_exploration', 'arguments': json.dumps(value)}}]), 'tool_calls')], None)


def proposal(group='g'):
    return {'nominations': [{'group_id': group, 'family': 'Synthetic catalyst family',
        'route': 'direct_candidate', 'hypothesis': 'Transfer at the target reaction center.',
        'structural_question': 'Can both partners occupy the catalytic site?'}],
        'searches': [], 'gaps': []}


def test_dataclass_transport_and_unknown_groups():
    assert parse_nomination(response(proposal()), {'g'})['nominations'][0]['group_id'] == 'g'
    with pytest.raises(ValueError, match='Unknown'):
        parse_nomination(response(proposal('unseen')), {'g'})


def test_search_must_reference_retrieved_evidence():
    value = proposal()
    value['searches'] = [{'query': 'synthetic query', 'basis_group_ids': ['unseen']}]
    with pytest.raises(ValueError, match='reference retrieved'):
        parse_nomination(response(value), {'g'})


def test_actual_record_does_not_inherit_a_source_family_label():
    groups = {}
    add_record(groups, {'primaryAccession': 'SYNTH', 'proteinDescription': {
        'recommendedName': {'fullName': {'value': 'Synthetic enzyme'}}},
        'uniProtKBCrossReferences': [{'database': 'InterPro', 'id': 'SOURCE_FOLD'}]}, {'path': 'synthetic'})
    assert next(iter(groups.values()))['family_ids'] == ['SOURCE_FOLD']


def test_bounded_tranche_preserves_family_breadth():
    groups = {key: {'id': key, 'family_ids': [family], 'members': {key: {'accession': key}}}
              for key, family in [('g1', 'f1'), ('g2', 'f1'), ('g3', 'f2')]}
    families = {f: {'hypotheses': ['synthetic']} for f in ('f1', 'f2')}
    result = initial_tranche(groups, families, limit=2)
    assert {g['family_ids'][0] for g in result.values()} == {'f1', 'f2'}
    assert len(groups) == 3


def test_duplicate_sequences_and_high_identity_do_not_fill_panel():
    def member(acc, seq, group):
        return {'accession': acc, 'sequence': seq, 'group_id': group,
                'annotation': {'native_activity_evidence': 'unknown'}, 'record': {}}
    sequence = 'ACDEFGHIKLMNPQRSTVWY'*3
    other = 'YWVTSRQPNMLKIHGFEDCA'*3
    pool = [member('SYNTH_A', sequence, 'g1'), member('SYNTH_B', sequence, 'g1'),
            member('SYNTH_C', sequence[:-1]+'A', 'g1'), member('SYNTH_D', other, 'g2')]
    nominations = {'g1': {'family': 'first'}, 'g2': {'family': 'second'}}
    selected, pairs = select_panel(pool, nominations, 4, .8)
    assert [m['accession'] for m in selected] == ['SYNTH_A', 'SYNTH_D']
    assert len(pairs) == 1 and pairs[0]['identity'] < .8
    assert sequence_identity(sequence, sequence) == 1


def test_nomination_preflight_blocks_call_before_allowance_overrun(tmp_path):
    class Client:
        requested_model = 'synthetic'
        _estimate_input_tokens = staticmethod(lambda *args: 200)
        def call(self, **kwargs):
            pytest.fail('Budget preflight must prevent this call')
    with pytest.raises(RuntimeError, match='Remaining allowance'):
        nominate(Client(), TokenUsage(), {}, {}, {}, tmp_path, 'synthetic', 100, float('inf'), 48)


def test_source_name_search_uses_supported_fields_and_retains_provenance():
    groups = {'g': {'id': 'g', 'name': 'Synthetic catalyst', 'members': {'SYNTH': {}}}}
    searches = search_from_source_groups(groups, {'g': {'route': 'direct_candidate'}})
    assert searches[0]['query'] == 'protein_name:"Synthetic catalyst"'
    assert searches[0]['basis_group_ids'] == ['g']


def test_budget_counts_provider_total_when_reasoning_is_not_in_completion():
    from types import SimpleNamespace
    usage = TokenUsage()
    usage.record('synthetic', SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=100, completion_tokens=20, total_tokens=170)))
    assert usage.total_tokens == 170
    assert usage.calls[0]['unattributed_tokens'] == 50
    assert TokenUsage.from_dict(usage.to_dict()).total_tokens == 170


def test_retry_admission_counts_repeated_input_despite_underreported_prompt():
    calls = [{'prompt_tokens': 2, 'completion_tokens': 30, 'total_tokens': 32},
             {'prompt_tokens': 90, 'completion_tokens': 20, 'total_tokens': 140}]
    assert conservative_charge(calls, 100) == 280
