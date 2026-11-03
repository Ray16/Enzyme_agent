"""Publish one run's bounded read-only snapshots to a private Site.

Credentials are supplied in the first line of stdin and remain in process
memory. This helper cannot start, steer or modify discovery work.
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import requests
from enzyme_discovery_agent.output.live_progress import snapshot


_IPR = re.compile(r"(?:ipr:)?(IPR\d{6})", re.I)


def _fetch_family_name(accession):
    """Read the authoritative InterPro entry label; failures remain unresolved."""
    try:
        response = requests.get(f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{accession}/",
                                headers={"Accept": "application/json"}, timeout=(3, 6), allow_redirects=False)
        response.raise_for_status()
        metadata = response.json().get("metadata") or {}
        name = (metadata.get("name") or {}).get("name")
        if metadata.get("accession") == accession and isinstance(name, str) and name.strip():
            return name.strip()
    except (requests.RequestException, ValueError, AttributeError, TypeError):
        pass
    return None


def enrich_family_names(state, cache):
    """Enrich display labels without treating annotation groups as families."""
    families = (state.get("frontier") or {}).get("families") or []
    candidate_rows = state.get("rows") or []
    ids = set()
    for row in families:
        match = _IPR.fullmatch(str(row.get("family") or ""))
        if match:
            ids.add(match.group(1).upper())
    for row in candidate_rows:
        match = _IPR.fullmatch(str(row.get("family_id") or row.get("family") or ""))
        if match:
            ids.add(match.group(1).upper())
    now = time.monotonic()
    pending = [key for key in sorted(ids) if key not in cache or cache[key][1] <= now][:6]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for accession, name in zip(pending, pool.map(_fetch_family_name, pending)):
            cache[accession] = (name, now + (86400 if name else 300))
    for row in families:
        key = str(row.get("family") or "")
        match = _IPR.fullmatch(key)
        if match:
            accession = match.group(1).upper()
            row["family_name"] = (cache.get(accession) or (None,))[0] or "Name unavailable"
            row["interpro_id"] = accession
        else:
            label = key.split(":", 1)[-1] if ":" in key else key
            row["family_name"] = (label or "Unassigned") + " (annotation group; family unresolved)"
            row["interpro_id"] = ""
    for row in candidate_rows:
        match = _IPR.fullmatch(str(row.get("family_id") or row.get("family") or ""))
        if match:
            accession = match.group(1).upper()
            row["family_id"] = accession
            row["family"] = (cache.get(accession) or (None,))[0] or row.get("family") or "Name unavailable"
    return state


def bounded_snapshot(root, name_cache=None):
    state = snapshot(Path(root))
    state['rows'] = state['rows'][:300]
    state['transitions'] = []
    original = state.get('frontier') or {}
    frontier = {'overview': original.get('overview') or {}}
    for name, cap in [('families', 150), ('hypotheses', 100), ('questions', 100)]:
        frontier[name] = (original.get(name) or [])[:cap]
    state['frontier'] = frontier
    if name_cache is not None:
        enrich_family_names(state, name_cache)
    payload = json.dumps(state, ensure_ascii=False, allow_nan=False).encode()
    if len(payload) > 1400000:
        raise ValueError('bounded snapshot exceeds the private site limit')
    return state, payload


def site_endpoint(site_url):
    base = urlsplit(site_url)
    if (base.scheme != 'https' or not base.hostname or not base.hostname.endswith('.chatgpt.site')
            or base.username or base.password or base.port not in (None, 443)
            or base.path not in ('', '/') or base.query or base.fragment):
        raise ValueError('site_url must be a direct HTTPS ChatGPT Site origin')
    return site_url.rstrip('/') + '/api/state'


def main():
    print('READY', flush=True)
    config = json.loads(sys.stdin.readline())
    root = Path(config['run_dir'])
    url = site_endpoint(config['site_url'])
    headers = {'OAI-Sites-Authorization': 'Bearer ' + config['token'], 'Content-Type': 'application/json'}
    last, last_publish, failures, name_cache = None, 0.0, 0, {}
    while True:
        try:
            state, payload = bounded_snapshot(root, name_cache)
            identity = json.dumps({k: v for k, v in state.items() if k != 'generated_at'}, sort_keys=True, default=str)
            if identity != last or time.monotonic() - last_publish >= 60:
                response = requests.put(url, data=payload, headers=headers, timeout=20, allow_redirects=False)
                response.raise_for_status()
                if not 200 <= response.status_code < 300:
                    raise RuntimeError(f'private site returned HTTP {response.status_code}')
                print(json.dumps({'published_at': state['generated_at'], 'status': state['status'],
                                  'candidate_count': state['candidate_count'], 'bytes': len(payload)}), flush=True)
                last, last_publish = identity, time.monotonic()
            failures = 0
            if state['status'] != 'running' or state['worker_alive'] is False:
                print('Publisher stopped after final or stopped-run snapshot.', flush=True)
                return 0
        except Exception as exc:
            failures += 1
            print(f'Publish failure {failures}: {type(exc).__name__}: {str(exc)[:300]}', flush=True)
            if failures >= 10:
                print('Publisher stopped after repeated failures; local run remains unaffected.', flush=True)
                return 1
        time.sleep(15)


if __name__ == '__main__':
    raise SystemExit(main())
