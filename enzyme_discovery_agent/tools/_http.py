"""Shared HTTP helper with retry, exponential backoff, and connection pooling."""

import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from enzyme_discovery_agent import config

logger = logging.getLogger("enzyme_agent")

_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

# ---------------------------------------------------------------------------
# Proactive per-host rate limiting
# ---------------------------------------------------------------------------
# Tools run concurrently in a thread pool, so several requests can target the
# same host at once.  We serialize per-host requests through a lock and enforce
# a minimum interval between them (config.HTTP_RATE_LIMITS), which prevents the
# bursts that trip 429s -- complementing the reactive 429 retry below.  Hosts
# with interval 0 skip the lock entirely, so unthrottled traffic is unaffected.
_RATE_STATE: dict[str, float] = {}       # host -> last request monotonic time
_RATE_LOCKS: dict[str, threading.Lock] = {}
_RATE_LOCKS_GUARD = threading.Lock()


def _host_min_interval(host: str) -> float:
    """Return the configured minimum inter-request interval for a host."""
    for fragment, interval in config.HTTP_RATE_LIMITS.items():
        if fragment in host:
            return interval
    return config.HTTP_MIN_INTERVAL_DEFAULT


def _get_host_lock(host: str) -> threading.Lock:
    lock = _RATE_LOCKS.get(host)
    if lock is None:
        with _RATE_LOCKS_GUARD:
            lock = _RATE_LOCKS.get(host)
            if lock is None:
                lock = threading.Lock()
                _RATE_LOCKS[host] = lock
    return lock


def _respect_rate_limit(host: str) -> None:
    """Block until at least the per-host minimum interval has elapsed.

    Holds the per-host lock across the wait so concurrent callers to the same
    host are spaced sequentially, not merely each delayed by a fixed amount.
    """
    interval = _host_min_interval(host)
    if interval <= 0:
        return
    lock = _get_host_lock(host)
    with lock:
        last = _RATE_STATE.get(host)
        now = time.monotonic()
        if last is not None:
            wait = interval - (now - last)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
        _RATE_STATE[host] = now

# Thread-safe lazy singleton requests.Session with connection pooling
_session: requests.Session | None = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    """Return a pooled requests.Session (created once, reused across threads)."""
    global _session
    if _session is not None:
        return _session
    with _session_lock:
        if _session is not None:
            return _session
        s = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=list(_RETRYABLE_HTTP_CODES),
            allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry,
        )
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _session = s
        return _session


def request_with_retry(
    req: urllib.request.Request,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> bytes:
    """Execute an HTTP request with exponential backoff on transient failures.

    Args:
        req: urllib.request.Request object
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        backoff_base: Base delay in seconds (doubles each retry)

    Returns:
        Response body as bytes

    Raises:
        urllib.error.URLError: After all retries are exhausted
        TimeoutError: After all retries are exhausted
    """
    session = _get_session()

    url = req.full_url
    headers = dict(req.headers)
    headers.update(req.unredirected_hdrs)
    # Some hosts (bioRxiv/Cloudflare, several publishers) reject the default
    # library User-Agent.  Send a browser-like one unless the caller set its own.
    if not any(k.lower() == "user-agent" for k in headers):
        headers["User-Agent"] = config.HTTP_USER_AGENT
    data = req.data
    method = req.get_method()

    _respect_rate_limit(urllib.parse.urlparse(url).netloc)

    try:
        resp = session.request(
            method=method,
            url=url,
            headers=headers,
            data=data,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.HTTPError as e:
        raise urllib.error.HTTPError(
            url, e.response.status_code, str(e),
            e.response.headers, None,
        ) from e
    except requests.exceptions.ConnectionError as e:
        raise urllib.error.URLError(str(e)) from e
    except requests.exceptions.Timeout as e:
        raise TimeoutError(str(e)) from e
    except requests.exceptions.RequestException as e:
        raise urllib.error.URLError(str(e)) from e
