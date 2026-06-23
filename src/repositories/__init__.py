"""Data-access layer (repositories).

Pulls live traffic data for the contrastive study straight from the production
stores, with no ORM:

  - ``TrafficSourceRepository`` (MySQL)  -> the id <-> name map of traffic
    sources, so we know which id each source represents.
  - ``RequestsRepository`` (ClickHouse)  -> the raw HTTP requests for each
    traffic source, shaped into the ``{headers, request, decision}`` records the
    feature pipeline (:mod:`src.pipeline`) already understands.

Connection settings are read from a ``.env`` file at the project root — see
``.env.example``.
"""
from .config import Settings, get_settings
from .traffic_source_repo import TrafficSource, TrafficSourceRepository
from .requests_repository import RequestRecord, RequestsRepository

__all__ = [
    "Settings",
    "get_settings",
    "TrafficSource",
    "TrafficSourceRepository",
    "RequestRecord",
    "RequestsRepository",
]
