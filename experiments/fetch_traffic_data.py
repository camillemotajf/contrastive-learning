"""Pull live request data per traffic source into local JSON dumps.

Ties the two repositories together:
  1. TrafficSourceRepository (MySQL)  -> the id <-> name relation.
  2. RequestsRepository (ClickHouse)  -> the raw requests for each source,
     shaped into {headers, request, decision} records.

Output: one JSON file per (traffic source, decision group) under data/raw/,
ready for src.pipeline.load_raw.

Configure connections in a project-root .env (see .env.example), then:
    python experiments/fetch_traffic_data.py --limit 10000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.repositories import RequestsRepository, TrafficSourceRepository
from src.repositories.requests_repository import dump_records_json


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=10000,
                    help="max requests per (source, decision group)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    ap.add_argument("--decisions", nargs="*", default=["bot", "safe", "offer"],
                    help="decisions to pull (each saved to its own file)")
    ap.add_argument("--source-ids", nargs="*", type=int, default=None,
                    help="restrict to these traffic_source_ids (default: all active)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    ts_repo = TrafficSourceRepository()
    req_repo = RequestsRepository()

    id_to_name = ts_repo.id_to_name(active_only=True)
    if args.source_ids:
        id_to_name = {i: id_to_name.get(i, f"source-{i}") for i in args.source_ids}

    print(f"Traffic sources: {len(id_to_name)}")
    for tsid, name in id_to_name.items():
        for decision in args.decisions:
            records = req_repo.fetch_by_traffic_source(
                tsid, decisions=[decision], limit=args.limit
            )
            if not records:
                continue
            fname = f"{_slug(name)}-{tsid}-{decision}.json"
            path = os.path.abspath(os.path.join(args.out, fname))
            dump_records_json(records, path)
            print(f"  [{tsid}] {name:<24} {decision:<6} -> {len(records):>6} rows  {fname}")


if __name__ == "__main__":
    main()
