"""Shared HTTP helper with retry, exponential backoff, and connection pooling."""

import logging
import threading
import urllib.request

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("enzyme_agent")

_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

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
    data = req.data
    method = req.get_method()

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
        import urllib.error
        raise urllib.error.HTTPError(
            url, e.response.status_code, str(e),
            e.response.headers, None,
        ) from e
    except requests.exceptions.ConnectionError as e:
        import urllib.error
        raise urllib.error.URLError(str(e)) from e
    except requests.exceptions.Timeout as e:
        raise TimeoutError(str(e)) from e
    except requests.exceptions.RequestException as e:
        import urllib.error
        raise urllib.error.URLError(str(e)) from e
