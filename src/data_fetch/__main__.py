"""Command line entrypoint for traffic-source-level exports."""
from __future__ import annotations

import argparse
from pathlib import Path

from .service import TrafficDataFetcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve traffic sources in MySQL and export all raw "
            "requests from ClickHouse."
        )
    )
    parser.add_argument("--source-ids", nargs="*", type=int)
    parser.add_argument("--source-names", nargs="*")
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--start", help="UTC inclusive: YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--end", help="UTC exclusive: YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--limit-per-group", type=int, default=10_000)
    parser.add_argument("--include-duplicates", action="store_true")
    parser.add_argument("--write-empty", action="store_true")
    parser.add_argument("--list-only", action="store_true",
                        help="resolve and print sources without querying ClickHouse")
    parser.add_argument("--out", default=str(Path("data") / "raw"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fetcher = TrafficDataFetcher()
    traffic_sources = fetcher.resolve_sources(
        active_only=not args.include_inactive,
        traffic_source_ids=args.source_ids,
        traffic_source_names=args.source_names,
    )

    print(f"Traffic sources matched: {len(traffic_sources)}")
    for traffic_source in traffic_sources:
        print(f"  [{traffic_source.id}] {traffic_source.name}")

    if args.list_only:
        return

    exported = fetcher.export(
        traffic_sources,
        output_dir=args.out,
        limit_per_group=args.limit_per_group,
        start=args.start,
        end=args.end,
        include_duplicates=args.include_duplicates,
        write_empty=args.write_empty,
    )
    print(f"Datasets exported: {len(exported)}")
    for item in exported:
        print(
            f"  source={item.traffic_source_id:<8} group={item.label_group:<7} "
            f"rows={item.rows:<7} {item.path}"
        )


if __name__ == "__main__":
    main()
