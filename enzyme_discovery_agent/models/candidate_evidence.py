"""Canonical candidate evidence, separate from display summaries."""


def candidate_record(metadata):
    """Overlay current evidence on the evaluation snapshot without reviving stale fields.

    Current metadata owns later source, interpretation and modeling updates, including
    explicit empty values. Older evaluations also wrote a three-reaction display
    prefix under the scientific field name. Recover that exact legacy prefix from
    its complete snapshot; never combine conflicting reaction lists.
    """
    evaluation = metadata.get('evaluation_data') or {}
    record = {**evaluation, **metadata}
    current = metadata.get('catalytic_activities')
    complete = evaluation.get('catalytic_activities')
    if (isinstance(current, list) and isinstance(complete, list)
            and len(current) == 3 and len(complete) > 3 and current == complete[:3]):
        record['catalytic_activities'] = complete
    return record
