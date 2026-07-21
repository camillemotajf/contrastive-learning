"""Application service for exporting request datasets by traffic source."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from src.repositories import RequestsRepository, TrafficSource, TrafficSourceRepository
from src.repositories.requests_repository import dump_records_json


# The core dashboard maps ``offer`` to unsafeClicks. Keep ``safe`` separate so
# the fetch layer does not silently invent a binary label.
DEFAULT_LABEL_GROUPS: Mapping[str, tuple[str, ...]] = {
    "bot": ("bot",),
    "unsafe": ("offer",),
    "safe": ("safe",),
}


@dataclass(frozen=True)
class ExportedDataset:
    traffic_source_id: int
    traffic_source_name: str
    label_group: str
    decisions: tuple[str, ...]
    rows: int
    path: str


class TrafficDataFetcher:
    """Resolve sources in MySQL and export all their ClickHouse request rows."""

    def __init__(
        self,
        traffic_source_repository: Optional[TrafficSourceRepository] = None,
        requests_repository: Optional[RequestsRepository] = None,
    ):
        self.traffic_sources = traffic_source_repository or TrafficSourceRepository()
        self.requests = requests_repository or RequestsRepository()

    def resolve_sources(
        self,
        *,
        active_only: bool = True,
        traffic_source_ids: Optional[Sequence[int]] = None,
        traffic_source_names: Optional[Sequence[str]] = None,
    ) -> list[TrafficSource]:
        return self.traffic_sources.find_all(
            active_only=active_only,
            traffic_source_ids=traffic_source_ids,
            traffic_source_names=traffic_source_names,
        )

    def export(
        self,
        traffic_sources: Sequence[TrafficSource],
        *,
        output_dir: str | Path,
        label_groups: Mapping[str, Sequence[str]] = DEFAULT_LABEL_GROUPS,
        limit_per_group: Optional[int] = 10_000,
        start: Optional[str] = None,
        end: Optional[str] = None,
        include_duplicates: bool = False,
        write_empty: bool = False,
    ) -> list[ExportedDataset]:
        """Export one pipeline-compatible JSON file per source/label group.

        Requests from every campaign belonging to the source are included.
        ``campaign_id`` remains present in every exported row.
        """
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        exported: list[ExportedDataset] = []

        for traffic_source in traffic_sources:
            source_slug = _slug(traffic_source.name)
            source_dir = root / source_slug

            for label_group, decisions in label_groups.items():
                decision_tuple = tuple(decisions)
                records = self.requests.fetch_by_traffic_source(
                    traffic_source.id,
                    decisions=decision_tuple,
                    limit=limit_per_group,
                    start=start,
                    end=end,
                    include_duplicates=include_duplicates,
                )
                if not records and not write_empty:
                    continue

                source_dir.mkdir(parents=True, exist_ok=True)
                path = source_dir / f"{source_slug}-{_slug(label_group)}.json"
                dump_records_json(records, str(path))
                exported.append(
                    ExportedDataset(
                        traffic_source_id=traffic_source.id,
                        traffic_source_name=traffic_source.name,
                        label_group=label_group,
                        decisions=decision_tuple,
                        rows=len(records),
                        path=str(path),
                    )
                )

        _write_manifest(root, exported, start=start, end=end)
        return exported


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "unknown"


def _write_manifest(
    root: Path,
    exported: Sequence[ExportedDataset],
    *,
    start: Optional[str],
    end: Optional[str],
) -> None:
    payload = {
        "scope": "traffic_source",
        "interval": {"start_inclusive": start, "end_exclusive": end},
        "datasets": [asdict(item) for item in exported],
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
