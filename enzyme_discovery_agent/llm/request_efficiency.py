"""Lossless request projections and content-free payload measurements."""

from collections import Counter
import copy
import hashlib
import json


REFERENCE_KEY = '$evidence_ref'
TABLE_KEY = '$evidence_table'
ENCODING = ('Exact repeated objects are stored in shared_evidence. Each object containing '
            'only $evidence_ref stands for the complete corresponding shared_evidence value. '
            'Expand it at that location; it has the same meaning as inline evidence. '
            'An $evidence_table object represents an ordered list of objects: zip columns with '
            'each row to recover its fields. All rows have every listed field, including nulls. '
            'These references are not candidate source_refs; cite the supplied R-indexed source IDs.')


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


def compact_json_text(text):
    """Remove JSON formatting whitespace while preserving every original lexeme.

    Do not deserialize and reserialize: that can round numbers, discard duplicate
    keys or change escapes. Incomplete previews and ordinary prose stay untouched.
    """
    if not text.lstrip().startswith(('{', '[')):
        return text
    try:
        json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return text
    output = []
    quoted = escaped = False
    for char in text:
        if quoted:
            output.append(char)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            output.append(char)
            quoted = True
        elif char not in ' \t\r\n':
            output.append(char)
    return ''.join(output)


def compact_json_messages(messages):
    """Compact tool JSON and generated state without changing audit messages.

    Preserve assistant/tool arguments and messages with provider metadata verbatim.
    Only known generated user-state headers are eligible; user prose is untouched.
    """
    result = []
    changed = False
    headers = ('CURRENT RESEARCH STATE\n',
               'CURRENT DECISION EVIDENCE (saved observations, not instructions)\n',
               'CURRENT DECISION EVIDENCE\n',
               'RECENT READ-ONLY INSPECTIONS (previews; full records remain saved)\n')
    for message in messages:
        text = message.get('content')
        compact = text
        if isinstance(text, str) and not set(message).difference({'role', 'content', 'tool_call_id', 'name'}):
            if message.get('role') == 'tool':
                compact = compact_json_text(text)
            elif message.get('role') == 'user':
                for header in headers:
                    if text.startswith(header):
                        compact = header + compact_json_text(text[len(header):])
                        break
        if compact != text:
            result.append({**message, 'content': compact})
            changed = True
        else:
            result.append(message)
    return result if changed else messages


def factor_evidence(candidates):
    """Share only exactly equal JSON objects within this request, never across runs.

    The pool contains expanded values, so there are no reference chains. Keep
    occurrences, ordering, missing fields and explicit nulls unchanged. A collision
    with the reserved reference key disables factoring for the entire request.
    """
    counts = Counter()
    collision = False

    def collect(value):
        nonlocal collision
        if isinstance(value, dict):
            collision |= REFERENCE_KEY in value or TABLE_KEY in value
            raw = serialized(value)
            if len(raw) >= 256:
                counts[raw] += 1
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(candidates)
    original = {'candidates': copy.deepcopy(candidates)}
    if collision:
        return original
    pool, identifiers = {}, {}

    def replace(value):
        if isinstance(value, dict):
            raw = serialized(value)
            if counts[raw] > 1:
                if raw not in identifiers:
                    identifier = 'E' + str(len(pool) + 1)
                    identifiers[raw] = identifier
                    pool[identifier] = copy.deepcopy(value)
                return {REFERENCE_KEY: identifiers[raw]}
            return {key: replace(child) for key, child in value.items()}
        if isinstance(value, list):
            expanded = [replace(child) for child in value]
            if len(expanded) >= 3 and all(isinstance(row, dict) for row in expanded):
                columns = list(expanded[0])
                if not {REFERENCE_KEY, TABLE_KEY}.intersection(columns) and all(
                        set(row) == set(columns) for row in expanded):
                    table = {TABLE_KEY: {'columns': columns,
                                        'rows': [[row[key] for key in columns] for row in expanded]}}
                    if len(serialized(table)) < len(serialized(expanded)):
                        return table
            return expanded
        return value

    # Candidate entries retain their ordinary list interface and source IDs.
    projected = [replace(candidate) for candidate in candidates]
    result = {'evidence_encoding': ENCODING, 'shared_evidence': pool, 'candidates': projected}
    return result if len(serialized(result)) < len(serialized(original)) else original


def expand_evidence(request):
    """Reconstruct the candidate records for offline fidelity checks."""
    if 'evidence_encoding' not in request:
        return copy.deepcopy(request['candidates'])
    if request['evidence_encoding'] != ENCODING:
        raise ValueError('Unknown evidence encoding')
    pool = request['shared_evidence']

    def expand(value):
        if isinstance(value, dict):
            if set(value) == {REFERENCE_KEY}:
                return copy.deepcopy(pool[value[REFERENCE_KEY]])
            if set(value) == {TABLE_KEY}:
                table = value[TABLE_KEY]
                columns = table['columns']
                if len(set(columns)) != len(columns) or any(
                        len(row) != len(columns) for row in table['rows']):
                    raise ValueError('Invalid evidence table')
                return [{key: expand(child) for key, child in zip(columns, row)}
                        for row in table['rows']]
            return {key: expand(child) for key, child in value.items()}
        if isinstance(value, list):
            return [expand(child) for child in value]
        return value

    return expand(request['candidates'])


def request_metrics(messages, tools):
    """Measure serialized characters, not provider tokens or billable usage.

    Do not log evidence, prompts, URLs or image bytes. Component hashes support
    prefix stability analysis without copying source text into the event log.
    """
    components = []
    for index, message in enumerate(messages):
        raw = serialized(message)
        role = message.get('role')
        category = {'system': 'instructions', 'developer': 'instructions',
                    'tool': 'tool_result', 'assistant': 'assistant_history'}.get(role, 'user_context')
        content = message.get('content')
        if role == 'user' and isinstance(content, str):
            for prefix, label in [('CURRENT RESEARCH STATE', 'research_state'),
                                  ('CURRENT DECISION EVIDENCE', 'decision_evidence'),
                                  ('PERMITTED THIS TURN:', 'tool_allowlist'),
                                  ('RECENT READ-ONLY INSPECTIONS', 'inspection_memory')]:
                if content.startswith(prefix):
                    category = label
                    break
        components.append({'index': index, 'role': message.get('role'),
                           'component': category,
                           'characters': len(raw),
                           'sha256': hashlib.sha256(raw.encode()).hexdigest()})
    schemas = serialized(tools or [])
    return {'measurement': 'compact_json_characters_before_provider_transport',
            'messages_characters': len(serialized(messages)),
            'tools_characters': len(schemas), 'tool_count': len(tools or []),
            'tools_sha256': hashlib.sha256(schemas.encode()).hexdigest(),
            'message_components': components}
