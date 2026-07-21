"""Run the full per-source study for every traffic source and SAVE artifacts.

Heavy compute lives here, not in the notebook — so a long run can't be lost to a
notebook cell timeout, and the saved files under ``results/cross_source/<source>/``
are the single source of truth the report notebook plots from.

    python experiments/run_cross_source.py                  # default sources
    python experiments/run_cross_source.py --sources facebook tiktok
    python experiments/run_cross_source.py --subsample 0    # full 20k corpus
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.source_study import run_source_study


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="*",
                    default=["outbrain", "facebook", "tiktok", "taboola"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 7])
    ap.add_argument("--subsample", type=int, default=4000,
                    help="per-source cap; 0 means the full corpus")
    ap.add_argument("--cc-epochs", type=int, default=15)
    ap.add_argument("--cc-clusters", type=int, default=4)
    ap.add_argument("--cc-aug", default="medium_http_aug")
    ap.add_argument("--noise-rates", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--knn-k", type=int, default=20)
    args = ap.parse_args()

    subsample = None if args.subsample in (0, None) else args.subsample
    t0 = time.time()
    for src in args.sources:
        print(f"\n=== {src} ===", flush=True)
        run_source_study(
            src, seeds=args.seeds, subsample=subsample,
            cc_clusters=args.cc_clusters, cc_epochs=args.cc_epochs, cc_aug=args.cc_aug,
            noise_rates=args.noise_rates, knn_k=args.knn_k, verbose=True)
    print(f"\nAll {len(args.sources)} sources done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
