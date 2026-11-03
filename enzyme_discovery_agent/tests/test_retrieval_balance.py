"""The campaign must put its hypotheses to the sequence archives, not only to papers.

A run that searches written reports exclusively accumulates enzyme names that no
lookup can resolve: the measured failure was 11 report searches against 1 archive
search, yielding 12 sequences where an earlier agent reached 85.
"""
import json

from enzyme_discovery_agent.models.event_log import append_event, retrieval_balance


def record(root, tool, n=1):
    for _ in range(n):
        append_event(root, 'tool_call', tool=tool, status='success')


def test_an_opening_report_search_is_not_yet_an_imbalance(tmp_path):
    """The signal must describe a pattern, not punish the first literature lookup."""
    record(tmp_path, 'search_pubmed', 2)
    assert not retrieval_balance(tmp_path)['reports_crowding_out_records']


def test_reports_without_any_archive_search_are_flagged(tmp_path):
    record(tmp_path, 'search_pubmed', 3)
    balance = retrieval_balance(tmp_path)
    assert balance['reports_crowding_out_records']
    assert balance['report_searches'] == 3 and balance['record_searches'] == 0


def test_a_balanced_campaign_is_never_flagged(tmp_path):
    record(tmp_path, 'search_pubmed', 4)
    record(tmp_path, 'search_uniprot', 4)
    assert not retrieval_balance(tmp_path)['reports_crowding_out_records']


def test_the_signal_clears_once_the_archives_are_actually_searched(tmp_path):
    """Withholding report sources has to be self-limiting, not a permanent ban."""
    record(tmp_path, 'search_pubmed', 6)
    assert retrieval_balance(tmp_path)['reports_crowding_out_records']
    record(tmp_path, 'search_interpro', 3)
    assert not retrieval_balance(tmp_path)['reports_crowding_out_records']


def test_only_recent_retrieval_counts(tmp_path):
    """An old report-only stretch must not keep firing after the run corrects."""
    record(tmp_path, 'search_pubmed', 8)
    record(tmp_path, 'search_uniprot', 8)
    balance = retrieval_balance(tmp_path)
    assert balance['record_searches'] == 8 and balance['report_searches'] == 0


def test_deliberation_and_reading_are_neither_reports_nor_records(tmp_path):
    """Only retrieval counts: reading a paper already fetched is not a new search."""
    record(tmp_path, 'inspect_discovery_source', 5)
    record(tmp_path, 'conclude_research_review', 5)
    balance = retrieval_balance(tmp_path)
    assert balance['retrievals_in_window'] == 0
    assert not balance['reports_crowding_out_records']


def test_the_split_is_read_from_the_registry_not_a_list_kept_here(tmp_path):
    """A newly registered source must classify itself, with no edit to this signal."""
    from enzyme_discovery_agent.bootstrap import register_builtin_tools
    from enzyme_discovery_agent.tool_registry import get_tools_by_category
    register_builtin_tools()
    categories = get_tools_by_category()
    assert 'search_interpro' in categories['database']
    assert 'search_uniprot' in categories['database']
    assert 'search_pubmed' in categories['literature']


def test_the_planner_still_offers_every_action_after_the_schema_deduplication(tmp_path):
    """Dropping the duplicated anyOf must not shrink what a plan may execute.

    Each action's schema is still carried once, as its own definition, so the
    contract the model writes arguments against is unchanged.
    """
    from enzyme_discovery_agent.bootstrap import register_builtin_tools
    from enzyme_discovery_agent.tool_registry import get_all_tools
    from enzyme_discovery_agent.context import stable_tool_definitions, strategic_tool_definitions
    register_builtin_tools()
    tools = get_all_tools()
    sent = stable_tool_definitions(tools)
    planner = next(t for t in strategic_tool_definitions(tools)
                   if t['function']['name'] == 'plan_research_step')
    properties = planner['function']['parameters']['properties']
    offered = set(properties['tool_name']['enum'])
    assert {'search_interpro', 'search_uniprot', 'get_protein_domains'} <= offered

    # Every action the planner may name must still carry a real schema somewhere in
    # the payload, otherwise the model is writing arguments blind.
    defined = {t['function']['name']: t['function'].get('parameters') or {} for t in sent}
    for name in offered:
        assert name in defined, f'{name} is offered with no schema sent'
    assert defined['search_interpro']['required'] == ['query']
    assert 'anyOf' not in properties['arguments'], 'the duplicated schema copy is back'
    assert len(json.dumps(sent)) < 45_000, 'the tool block regrew past its measured budget'
