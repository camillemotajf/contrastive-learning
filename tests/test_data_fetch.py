from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data_fetch import TrafficDataFetcher
from src.repositories.config import ClickHouseSettings, MySQLSettings
from src.repositories.requests_repository import RequestRecord, RequestsRepository
from src.repositories.traffic_source_repo import TrafficSource, TrafficSourceRepository


SOURCE = TrafficSource(id=7, name="Outbrain", active=True)


class CapturingTrafficSourceRepository(TrafficSourceRepository):
    def __init__(self):
        super().__init__(MySQLSettings("db", 3306, "user", "pass", "twr"))
        self.sql = ""
        self.params = ()

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        self.sql, self.params = sql, params
        return [{"id": 7, "name": "Outbrain", "active": 1}]


class FakeQueryResult:
    column_names = [
        "traffic_source_id", "campaign_id", "decision", "user_agent",
        "headers", "params", "body", "datetime",
    ]
    result_rows = [[
        7, 123, "bot", "Agent/1.0", {"Accept": "*/*"}, {"x": "1"},
        {"event": "click"}, "2026-01-01 00:00:00",
    ]]


class FakeClickHouseClient:
    def __init__(self):
        self.sql = ""
        self.parameters = {}
        self.closed = False

    def query(self, sql: str, parameters: dict):
        self.sql, self.parameters = sql, parameters
        return FakeQueryResult()

    def close(self):
        self.closed = True


class FakeTrafficSources:
    def find_all(self, **_kwargs):
        return [SOURCE]


class FakeRequests:
    def fetch_by_traffic_source(self, traffic_source_id: int, **kwargs):
        decision = kwargs["decisions"][0]
        return [RequestRecord(
            campaign_id=123,
            traffic_source_id=traffic_source_id,
            decision=decision,
            headers=json.dumps({"User-Agent": "Agent/1.0"}),
            request=json.dumps({"click_id": "abc"}),
            body=json.dumps({}),
            datetime="2026-01-01 00:00:00",
        )]


class DataFetchTests(unittest.TestCase):
    def test_source_filters_use_bound_mysql_values(self):
        repo = CapturingTrafficSourceRepository()
        sources = repo.find_all(
            traffic_source_ids=[7],
            traffic_source_names=["Outbrain"],
        )
        self.assertEqual([7], [source.id for source in sources])
        self.assertIn("traffic_sources_pk", repo.sql)
        self.assertEqual((7, "outbrain"), repo.params)

    def test_clickhouse_query_is_scoped_only_by_source(self):
        settings = ClickHouseSettings("ch", 8123, "default", "", "shield", False)
        repo = RequestsRepository(settings)
        client = FakeClickHouseClient()
        with patch("src.repositories.requests_repository._client", return_value=client):
            records = repo.fetch_by_traffic_source(
                7,
                decisions=["bot"],
                start="2026-01-01 00:00:00",
                end="2026-01-02 00:00:00",
            )
        self.assertIn("traffic_source_id = {tsid:UInt16}", client.sql)
        self.assertNotIn("campaign_id =", client.sql)
        self.assertIn("duplicated = false", client.sql)
        self.assertEqual(123, records[0].campaign_id)
        self.assertTrue(client.closed)

    def test_export_creates_source_layout_and_keeps_campaign_id(self):
        fetcher = TrafficDataFetcher(FakeTrafficSources(), FakeRequests())
        with tempfile.TemporaryDirectory() as tmp:
            exported = fetcher.export([SOURCE], output_dir=tmp)
            self.assertEqual(3, len(exported))
            paths = [Path(item.path) for item in exported]
            self.assertTrue(all(path.is_file() for path in paths))
            manifest = json.loads(
                (Path(tmp) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual("traffic_source", manifest["scope"])
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(123, payload[0]["campaign_id"])
            self.assertEqual(7, payload[0]["traffic_source_id"])


if __name__ == "__main__":
    unittest.main()
