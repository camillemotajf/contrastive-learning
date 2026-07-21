"""Traffic-source repository (MySQL, no ORM).

Resolves the relation *which id represents which traffic source*, used to label
and group the ClickHouse requests pulled by :class:`RequestsRepository`.

Schema (TWR `local_TWR`):
    traffic_sources(traffic_sources_pk, traffic_source, active)
    traffic_sources_child(id, name, traffic_sources_fk)   -- optional sub-sources

Connection comes from ``MYSQL_URI`` (or discrete ``MYSQL_*`` vars) via
:func:`src.repositories.config.get_settings`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config import MySQLSettings, get_settings


@dataclass(frozen=True)
class TrafficSource:
    id: int
    name: str
    active: bool
    parent_id: Optional[int] = None  # set for sub-sources (traffic_sources_child)


def _connect(settings: MySQLSettings):
    """Open a raw MySQL connection. Tries PyMySQL, then mysql-connector."""
    try:
        import pymysql  # type: ignore

        return pymysql.connect(
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
    except ImportError:
        pass

    try:
        import mysql.connector  # type: ignore

        return mysql.connector.connect(
            host=settings.host,
            port=settings.port,
            user=settings.user,
            password=settings.password,
            database=settings.database,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "No MySQL driver installed. Run: uv pip install pymysql"
        ) from exc


class TrafficSourceRepository:
    """Reads the traffic-source id <-> name relation from MySQL."""

    def __init__(self, settings: Optional[MySQLSettings] = None):
        self._settings = settings or get_settings().mysql

    def _query(self, sql: str, params: tuple = ()) -> List[dict]:
        conn = _connect(self._settings)
        try:
            cur = conn.cursor(dictionary=True) if _is_connector(conn) else conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
            return list(rows)
        finally:
            conn.close()

    def find_all(
        self,
        active_only: bool = True,
        *,
        traffic_source_ids: Optional[Sequence[int]] = None,
        traffic_source_names: Optional[Sequence[str]] = None,
    ) -> List[TrafficSource]:
        """Top-level traffic sources, optionally filtered by id or name."""
        conditions = []
        params: list[object] = []
        if active_only:
            conditions.append("active = 1")
        if traffic_source_ids:
            ids = [int(value) for value in traffic_source_ids]
            conditions.append(f"traffic_sources_pk IN ({', '.join(['%s'] * len(ids))})")
            params.extend(ids)
        if traffic_source_names:
            names = [str(value).strip().lower() for value in traffic_source_names]
            conditions.append(
                f"LOWER(TRIM(traffic_source)) IN ({', '.join(['%s'] * len(names))})"
            )
            params.extend(names)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._query(
            f"""
            SELECT traffic_sources_pk AS id,
                   traffic_source     AS name,
                   active             AS active
            FROM traffic_sources
            {where}
            ORDER BY traffic_sources_pk
            """,
            tuple(params),
        )
        return [
            TrafficSource(id=int(r["id"]), name=str(r["name"]), active=bool(r["active"]))
            for r in rows
        ]

    def find_children(self) -> List[TrafficSource]:
        """Sub-sources from traffic_sources_child (id, name, parent fk)."""
        rows = self._query(
            """
            SELECT id                 AS id,
                   name               AS name,
                   traffic_sources_fk AS parent_id
            FROM traffic_sources_child
            ORDER BY id
            """
        )
        return [
            TrafficSource(
                id=int(r["id"]),
                name=str(r["name"]),
                active=True,
                parent_id=int(r["parent_id"]) if r["parent_id"] is not None else None,
            )
            for r in rows
        ]

    def id_to_name(self, active_only: bool = True) -> Dict[int, str]:
        """Convenience map ``{traffic_source_id: name}``."""
        return {ts.id: ts.name for ts in self.find_all(active_only=active_only)}

    def name_to_id(self, active_only: bool = True) -> Dict[str, int]:
        return {ts.name: ts.id for ts in self.find_all(active_only=active_only)}


def _is_connector(conn) -> bool:
    """True when the connection is a mysql-connector connection (needs the
    ``dictionary=True`` cursor kwarg instead of PyMySQL's DictCursor)."""
    return type(conn).__module__.startswith("mysql.connector")
