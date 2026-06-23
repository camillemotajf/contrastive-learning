"""High-precision external heuristics — anchors, NOT ground truth.

These flag obvious bots (crawler/script User-Agents, unsubstituted templates).
They are precise but not complete, so we use them to CROSS-CHECK and interpret
suspicion rankings, never as the definition of noise.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Dict, List, Sequence

import numpy as np

CRAWLER_UA = re.compile(
    r"facebookexternalhit|googlebot|bingbot|bot\b|crawler|spider|slurp|"
    r"bingpreview|headless",
    re.IGNORECASE,
)
SCRIPT_UA = re.compile(
    r"python-requests|curl|wget|scrapy|selenium|playwright|puppeteer|"
    r"httpclient|okhttp|go-http-client|java/|libwww|aiohttp",
    re.IGNORECASE,
)
TEMPLATE_RE = re.compile(r"\{\{.*?\}\}")


def _parse_headers(h: str) -> dict:
    try:
        d = json.loads(h)
        return d if isinstance(d, dict) else {}
    except Exception:
        try:
            d = ast.literal_eval(h)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}


def _ua(headers_raw: str) -> str:
    hd = _parse_headers(headers_raw)
    for key in ("User-Agent", "user-agent", "USER-AGENT"):
        if key in hd:
            return str(hd[key])
    # case-insensitive fallback
    for k, v in hd.items():
        if k.lower() == "user-agent":
            return str(v)
    return ""


def heuristic_flags(headers: Sequence[str], requests: Sequence[str]) -> Dict[str, np.ndarray]:
    """Per-sample boolean columns used for interpretation of suspects."""
    has_template = np.array([bool(TEMPLATE_RE.search(str(r))) for r in requests])
    uas = [_ua(h) for h in headers]
    has_crawler_ua = np.array([bool(CRAWLER_UA.search(ua)) for ua in uas])
    has_script_ua = np.array([bool(SCRIPT_UA.search(ua)) for ua in uas])
    bot_flag = has_template | has_crawler_ua | has_script_ua
    return {
        "has_template": has_template,
        "has_crawler_ua": has_crawler_ua,
        "has_script_ua": has_script_ua,
        "heuristic_bot_flag": bot_flag,
        "user_agent": np.array(uas, dtype=object),
    }


def heuristic_bot_flag(headers: Sequence[str], requests: Sequence[str]) -> np.ndarray:
    return heuristic_flags(headers, requests)["heuristic_bot_flag"]
