"""Requests repository (ClickHouse, no ORM).

Pulls raw HTTP requests for each traffic source from the `requests` table and
shapes them into the ``{headers, request, decision}`` records that the feature
pipeline (:mod:`src.pipeline`) already consumes.

Relevant `requests` columns (see TWR core `clickhouseRequestsRow`):
    datetime, traffic_source_id, decision (safe|offer|bot),
    user_agent, headers Map(String,String), params Map(String,String),
    body Map(String,String)

We reconstruct each record as:
    headers  -> JSON object of the header map (user_agent re-injected)
    request  -> JSON object of the request params (the closest analogue to the
                original ``request`` field used in the offline JSON dumps)
    decision -> the system label (kept as-is so callers map it to 0/1)

Connection comes from ``CLICKHOUSE_URL`` (or discrete ``CLICKHOUSE_*`` vars) via
:func:`src.repositories.config.get_settings`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .config import ClickHouseSettings, get_settings


@dataclass(frozen=True)
class RequestRecord:
    traffic_source_id: int
    decision: str
    headers: str  # JSON string (matches src.pipeline expectations)
    request: str  # JSON string
    datetime: str

    def to_pipeline_item(self) -> dict:
        """Shape used by :func:`src.pipeline.load_raw` / the JSON dumps."""
        return {
            "headers": self.headers,
            "request": self.request,
            "decision": self.decision,
        }


def _client(settings: ClickHouseSettings):
    """Open a ClickHouse client over the HTTP interface."""
    try:
        import clickhouse_connect  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "clickhouse-connect not installed. Run: uv pip install clickhouse-connect"
        ) from exc

    return clickhouse_connect.get_client(
        host=settings.host,
        port=settings.port,
        username=settings.username,
        password=settings.password,
        database=settings.database,
        secure=settings.secure,
    )


class RequestsRepository:
    """Reads raw requests per traffic source from ClickHouse."""

    def __init__(self, settings: Optional[ClickHouseSettings] = None):
        self._settings = settings or get_settings().clickhouse

    # -- core query ------------------------------------------------------- #
    def fetch_by_traffic_source(
        self,
        traffic_source_id: int,
        decisions: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        start: Optional[str] = None,  # "YYYY-MM-DD HH:MM:SS" (UTC)
        end: Optional[str] = None,
    ) -> List[RequestRecord]:
        """All requests for one traffic source, newest first.

        ``decisions`` filters by the system label (e.g. ``["bot"]`` or
        ``["safe", "offer"]``); ``None`` returns every decision.
        """
        where = ["traffic_source_id = {tsid:UInt32}"]
        params: Dict[str, object] = {"tsid": int(traffic_source_id)}

        if decisions:
            where.append("decision IN {decisions:Array(String)}")
            params["decisions"] = list(decisions)
        if start:
            where.append("datetime >= {start:DateTime}")
            params["start"] = start
        if end:
            where.append("datetime < {end:DateTime}")
            params["end"] = end

        sql = f"""
            SELECT traffic_source_id,
                   decision,
                   user_agent,
                   headers,
                   params,
                   toString(datetime) AS datetime
            FROM requests
            WHERE {' AND '.join(where)}
            ORDER BY datetime DESC
            {f'LIMIT {int(limit)}' if limit else ''}
        """

        client = _client(self._settings)
        try:
            result = client.query(sql, parameters=params)
            columns = result.column_names
            return [self._to_record(dict(zip(columns, row))) for row in result.result_rows]
        finally:
            client.close()

    def fetch_for_sources(
        self,
        traffic_source_ids: Iterable[int],
        decisions: Optional[Sequence[str]] = None,
        limit_per_source: Optional[int] = None,
    ) -> Dict[int, List[RequestRecord]]:
        """Convenience: a ``{traffic_source_id: [records]}`` map over many ids."""
        return {
            tsid: self.fetch_by_traffic_source(
                tsid, decisions=decisions, limit=limit_per_source
            )
            for tsid in traffic_source_ids
        }

    def counts_by_decision(self, traffic_source_id: int) -> Dict[str, int]:
        """How many requests of each decision this source has (sanity check)."""
        sql = """
            SELECT decision, count() AS n
            FROM requests
            WHERE traffic_source_id = {tsid:UInt32}
            GROUP BY decision
        """
        client = _client(self._settings)
        try:
            result = client.query(sql, parameters={"tsid": int(traffic_source_id)})
            return {row[0]: int(row[1]) for row in result.result_rows}
        finally:
            client.close()

    # -- shaping ---------------------------------------------------------- #
    @staticmethod
    def _to_record(row: dict) -> RequestRecord:
        headers = dict(row.get("headers") or {})
        ua = row.get("user_agent")
        if ua and "User-Agent" not in headers and "user-agent" not in headers:
            headers["User-Agent"] = ua  # re-inject the column extracted on write
        params = dict(row.get("params") or {})
        return RequestRecord(
            traffic_source_id=int(row["traffic_source_id"]),
            decision=str(row["decision"]),
            headers=json.dumps(headers, ensure_ascii=False),
            request=json.dumps(params, ensure_ascii=False),
            datetime=str(row.get("datetime", "")),
        )


def dump_records_json(records: Sequence[RequestRecord], path: str) -> None:
    """Persist records to a JSON file in the same shape as the offline dumps
    (``[{headers, request, decision}, ...]``), so the existing pipeline and
    notebooks can read them unchanged."""
    items = [r.to_pipeline_item() for r in records]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)
