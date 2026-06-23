"""HTTP-level augmentations — two plausible *views* of the same request.

Motivation
----------
The original Contrastive Clustering applied masking + Gaussian noise directly to
the already-compressed SVD vector. Those perturbations have no clear meaning in
HTTP space, so the two views were not realistic variants of the same request and
the instance-contrastive signal was weak.

Here we augment at the TEXT / JSON level — BEFORE TF-IDF/SVD — so the two views
are genuinely different serialisations of the same underlying HTTP request:
reordered keys, dropped optional headers, masked values, case changes, etc. The
fitted vectoriser then maps both views into feature space.

All randomness flows through a passed-in ``numpy.random.Generator`` (``rng``) so
augmentation is reproducible and leakage-free.

Honesty note: an augmentation defines an *invariance* we ask the model to learn.
``ua_masking`` / ``template_masking`` deliberately destroy the strongest bot
signals — they exist for optional shortcut-robustness checks (does CC collapse
without the shortcut?), not because masking the User-Agent is realistic for
production traffic.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Headers that genuinely vary request-to-request and carry little class signal —
# safe to drop/mask as an invariance.
UNSTABLE_HEADERS = {
    "date", "x-request-id", "x-amzn-trace-id", "cf-ray", "x-correlation-id",
    "cookie", "set-cookie", "x-forwarded-for", "via", "x-real-ip",
}
# Headers a real client may or may not send — dropping one is plausible.
OPTIONAL_HEADERS = {
    "accept-encoding", "accept-language", "cache-control", "dnt", "pragma",
    "upgrade-insecure-requests", "sec-ch-ua", "sec-ch-ua-mobile",
    "sec-ch-ua-platform", "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site",
    "sec-fetch-user", "referer", "te",
}
# Headers that carry the class signal — preserved under *_preserving configs.
SEMANTIC_HEADERS = {"user-agent", "accept", "from"}

_TEMPLATE_RE = re.compile(r"\{\{.*?\}\}")
_VERSION_RE = re.compile(r"\d+(?:\.\d+)+")


@dataclass
class AugConfig:
    reorder_keys: bool = True
    drop_optional_prob: float = 0.0
    mask_value_prob: float = 0.0
    change_case_prob: float = 0.0
    drop_unstable_prob: float = 0.0
    whitespace_jitter: bool = True
    ua_mask_prob: float = 0.0          # partial UA version masking
    preserve_ua: bool = True           # never fully drop/replace the UA
    request_mask_value_prob: float = 0.0
    preserve_template: bool = True     # keep {{...}} request templates intact
    # internal list of "mask" tokens
    mask_token: str = "<MASK>"

    def copy(self) -> "AugConfig":
        return AugConfig(**self.__dict__)


# --------------------------------------------------------------------------- #
# Named configurations
# --------------------------------------------------------------------------- #
AUG_CONFIGS: Dict[str, AugConfig] = {
    "light_http_aug": AugConfig(
        reorder_keys=True, drop_optional_prob=0.05, mask_value_prob=0.05,
        change_case_prob=0.10, drop_unstable_prob=0.50, whitespace_jitter=True,
        ua_mask_prob=0.0, preserve_ua=True, request_mask_value_prob=0.05,
        preserve_template=True,
    ),
    "medium_http_aug": AugConfig(
        reorder_keys=True, drop_optional_prob=0.15, mask_value_prob=0.15,
        change_case_prob=0.25, drop_unstable_prob=0.80, whitespace_jitter=True,
        ua_mask_prob=0.15, preserve_ua=True, request_mask_value_prob=0.15,
        preserve_template=True,
    ),
    "strong_http_aug": AugConfig(
        reorder_keys=True, drop_optional_prob=0.30, mask_value_prob=0.30,
        change_case_prob=0.40, drop_unstable_prob=1.0, whitespace_jitter=True,
        ua_mask_prob=0.30, preserve_ua=True, request_mask_value_prob=0.30,
        preserve_template=True,
    ),
    # Ablation pairs: keep vs destroy a specific signal.
    "ua_preserving_aug": AugConfig(
        reorder_keys=True, drop_optional_prob=0.15, mask_value_prob=0.15,
        change_case_prob=0.25, drop_unstable_prob=0.80, ua_mask_prob=0.0,
        preserve_ua=True, request_mask_value_prob=0.15, preserve_template=True,
    ),
    "ua_masking_aug": AugConfig(
        reorder_keys=True, drop_optional_prob=0.15, mask_value_prob=0.15,
        change_case_prob=0.25, drop_unstable_prob=0.80, ua_mask_prob=1.0,
        preserve_ua=False, request_mask_value_prob=0.15, preserve_template=True,
    ),
    "template_preserving_aug": AugConfig(
        reorder_keys=True, drop_optional_prob=0.15, mask_value_prob=0.15,
        change_case_prob=0.25, drop_unstable_prob=0.80, ua_mask_prob=0.15,
        preserve_ua=True, request_mask_value_prob=0.15, preserve_template=True,
    ),
    "template_masking_aug": AugConfig(
        reorder_keys=True, drop_optional_prob=0.15, mask_value_prob=0.15,
        change_case_prob=0.25, drop_unstable_prob=0.80, ua_mask_prob=0.15,
        preserve_ua=True, request_mask_value_prob=1.0, preserve_template=False,
    ),
}


def get_config(name_or_cfg) -> AugConfig:
    if isinstance(name_or_cfg, AugConfig):
        return name_or_cfg
    if name_or_cfg in AUG_CONFIGS:
        return AUG_CONFIGS[name_or_cfg]
    raise KeyError(f"Unknown augmentation config: {name_or_cfg}. "
                   f"Available: {sorted(AUG_CONFIGS)}")


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _parse(raw: str):
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _serialise(obj: dict, rng: np.random.Generator, jitter: bool) -> str:
    if jitter and rng.random() < 0.5:
        return json.dumps(obj, separators=(", ", ": "))   # spaced
    return json.dumps(obj, separators=(",", ":"))          # compact


def _mask_ua(value: str, rng: np.random.Generator, mask_prob: float) -> str:
    """Partially mask a User-Agent: blur version numbers but keep brand tokens
    (Mozilla/Chrome/bot/crawler/...) so the class signal is not fully destroyed
    unless mask_prob is high."""
    if rng.random() >= mask_prob:
        return value
    return _VERSION_RE.sub(lambda m: ".".join("X" * len(p) for p in m.group().split(".")),
                           value)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def augment_headers_text(headers_raw: str, rng: np.random.Generator, config) -> str:
    """Return one augmented serialisation of a headers JSON string."""
    cfg = get_config(config)
    obj = _parse(headers_raw)
    if obj is None:
        return headers_raw  # unparseable: leave as-is

    items: List[Tuple[str, object]] = list(obj.items())
    if cfg.reorder_keys:
        rng.shuffle(items)

    out: Dict[str, object] = {}
    for key, value in items:
        low = key.lower()
        is_ua = low == "user-agent"

        # drop unstable headers
        if low in UNSTABLE_HEADERS and rng.random() < cfg.drop_unstable_prob:
            continue
        # drop optional headers (never the semantic ones)
        if (low in OPTIONAL_HEADERS and low not in SEMANTIC_HEADERS
                and rng.random() < cfg.drop_optional_prob):
            continue

        new_key = key
        if rng.random() < cfg.change_case_prob:
            new_key = _vary_case(key, rng)

        new_val = value
        if is_ua:
            if not cfg.preserve_ua and cfg.ua_mask_prob >= 1.0:
                new_val = cfg.mask_token
            else:
                new_val = _mask_ua(str(value), rng, cfg.ua_mask_prob)
        elif low not in SEMANTIC_HEADERS and rng.random() < cfg.mask_value_prob:
            new_val = cfg.mask_token  # mask value, keep the key

        out[new_key] = new_val

    return _serialise(out, rng, cfg.whitespace_jitter)


def augment_request_text(request_raw: str, rng: np.random.Generator, config) -> str:
    """Return one augmented serialisation of a request JSON string. Structure is
    preserved; values may be masked. ``{{...}}`` templates are kept intact under
    ``preserve_template`` and masked otherwise."""
    cfg = get_config(config)
    obj = _parse(request_raw)
    if obj is None:
        # non-JSON request: only optionally mask templates
        if not cfg.preserve_template:
            return _TEMPLATE_RE.sub(cfg.mask_token, request_raw)
        return request_raw

    items = list(obj.items())
    if cfg.reorder_keys:
        rng.shuffle(items)

    out: Dict[str, object] = {}
    for key, value in items:
        sval = str(value)
        has_template = bool(_TEMPLATE_RE.search(sval))
        if has_template and not cfg.preserve_template:
            value = _TEMPLATE_RE.sub(cfg.mask_token, sval)
        elif not has_template and rng.random() < cfg.request_mask_value_prob:
            value = cfg.mask_token  # mask value, preserve structure/key
        new_key = _vary_case(key, rng) if rng.random() < cfg.change_case_prob else key
        out[new_key] = value

    return _serialise(out, rng, cfg.whitespace_jitter)


def make_two_http_views(headers_raw: str, request_raw: str,
                        rng: np.random.Generator, config) -> Tuple[str, str]:
    """Two independently-augmented views of the SAME request, each a
    ``headers || request`` concatenation (the string the vectoriser sees)."""
    cfg = get_config(config)
    h1 = augment_headers_text(headers_raw, rng, cfg)
    h2 = augment_headers_text(headers_raw, rng, cfg)
    r1 = augment_request_text(request_raw, rng, cfg)
    r2 = augment_request_text(request_raw, rng, cfg)
    return f"{h1} {r1}", f"{h2} {r2}"


def _vary_case(key: str, rng: np.random.Generator) -> str:
    """Plausible header-key case change (keys are case-insensitive in HTTP)."""
    choice = rng.integers(0, 3)
    if choice == 0:
        return key.lower()
    if choice == 1:
        return key.upper()
    return "-".join(p.capitalize() for p in key.split("-"))
