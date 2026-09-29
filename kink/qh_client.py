"""QH Fairvalue API client.

Runs in the Streamlit (Python) process, NOT in the browser, so browser CORS
rules do not apply here - the CORS-specific banner from the browser design is
therefore intentionally absent. Everything else from the spec is implemented:
  - header is always built as "Bearer " + access (never doubled)
  - ~8 s timeout -> "unreachable" message
  - 401/403 -> token message
  - non-JSON body -> first ~300 chars of the raw text
  - 429 -> exponential backoff 2s/4s/8s, max 3 retries, on_retry() callback
`access` (raw token) is only ever held in memory by the caller.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

BASE_URL = "https://qh-api.corp.hertshtengroup.com/apis/fairvalue/"
TIMEOUT_S = 8
BACKOFF_S = (2, 4, 8)


class QHError(Exception):
    kind = "error"


class QHUnreachable(QHError):
    kind = "unreachable"


class QHTlsError(QHError):
    kind = "tls"


class QHAuthError(QHError):
    kind = "auth"


class QHNonJson(QHError):
    kind = "non_json"


class QHRateLimited(QHError):
    kind = "rate_limited"


class QHHttpError(QHError):
    kind = "http"


@dataclass
class Quote:
    price: float
    ts: int   # epoch ms


def auth_header(access: str) -> dict:
    raw = (access or "").strip().strip("\"'").strip()
    value = raw if raw.lower().startswith("bearer ") else "Bearer " + raw
    return {"Authorization": value}


def _get(access: str, products: str, on_retry=None) -> dict:
    url = f"{BASE_URL}?products={products}"
    for attempt in range(len(BACKOFF_S) + 1):
        try:
            r = requests.get(url, headers=auth_header(access), timeout=TIMEOUT_S)
        except requests.exceptions.SSLError as e:
            raise QHTlsError(
                "TLS certificate error talking to the QH API. If your company uses a custom CA, "
                f"set the REQUESTS_CA_BUNDLE environment variable to its .pem file. ({e.__class__.__name__})")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            raise QHUnreachable("QH API unreachable — check VPN/network connection.")

        if r.status_code == 429:
            if attempt < len(BACKOFF_S):
                wait = BACKOFF_S[attempt]
                if on_retry:
                    on_retry(attempt + 1, wait)
                time.sleep(wait)
                continue
            raise QHRateLimited("Rate limited (429) — gave up after 3 retries. Staging limit is 7 requests/min.")
        if r.status_code in (401, 403):
            raise QHAuthError("Token invalid or expired — please paste a fresh token.")
        if not r.ok:
            raise QHHttpError(f"HTTP {r.status_code} from QH API: {r.text[:300]}")
        try:
            return r.json()
        except ValueError:
            raise QHNonJson(f"QH API returned a non-JSON response (login redirect?). First 300 chars: {r.text[:300]}")
    raise QHRateLimited("Rate limited (429).")


def _parse(payload) -> dict[str, Quote]:
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise QHNonJson(f"Unexpected JSON shape (no 'data' list): {str(payload)[:300]}")
    out = {}
    for row in rows:
        try:
            out[str(row["Contract"]).upper()] = Quote(float(row["Price"]), int(row["Timestamp"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fetch(access: str, products: list[str], on_retry=None) -> dict[str, Quote]:
    """ONE batched request for every distinct leg code. Keys are upper-case."""
    codes = sorted({p.strip().upper() for p in products if p.strip()})
    return _parse(_get(access, ",".join(codes), on_retry))


def fetch_all_codes(access: str, on_retry=None) -> set[str]:
    """'Test Connection': products=* once; returns the set of valid codes (upper-case)."""
    return set(_parse(_get(access, "*", on_retry)))
