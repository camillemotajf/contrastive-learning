"""Data-access layer (repositories).

Pulls live traffic data for the contrastive study straight from the production
stores, with no ORM:

  - ``TrafficSourceRepository`` (MySQL) resolves traffic-source ids and names.
  - ``RequestsRepository`` (ClickHouse) reads all corresponding raw HTTP rows,
    preserving the campaign id of each request.

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
