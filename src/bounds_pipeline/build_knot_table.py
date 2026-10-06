"""
Builds the knot table knot_identification.py reads: the unknot,
every prime knot with at most --max-crossings crossings in both
chiralities, and the connected sums of those whose crossing numbers
add up to at most the same bound, each with the invariants the
identification compares and its Rasmussen s-invariant.

Usage:
    sage -python src/bounds_pipeline/build_knot_table.py [--max-crossings 10]
        [--source-csv datasets/rolfsen.csv]
        [--out src/bounds_pipeline/knots_le10_invariants.json]
        [--no-knotjob]

The input is a CSV with a Name column in KnotInfo's style (3_1,
10_165) and a PD Notation column as in datasets/rolfsen.csv, the
default; it must list every prime knot up to the crossing bound.
The output defaults to knots_le<max-crossings>_invariants.json beside
knot_identification.py, which is where it reads the table from.
With --no-knotjob, s comes from the closed forms on
positive and alternating diagrams and stays empty elsewhere.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Build the knot table of knot_identification.py.")
    parser.add_argument("--max-crossings", type=int, default=10,
                        help="largest crossing number tabulated (default 10)")
    parser.add_argument("--source-csv", default=None,
                        help="CSV of prime knots with Name and PD Notation "
                             "columns (default datasets/rolfsen.csv)")
    parser.add_argument("--out", default=None,
                        help="output JSON (default "
                             "knots_le<max-crossings>_invariants.json "
                             "beside knot_identification.py)")
    parser.add_argument("--no-knotjob", action="store_true",
                        help="do not call KnotJob; s comes from the closed "
                             "forms where they apply")
    parser.add_argument("--knotjob-timeout", type=int, default=60,
                        help="per-knot KnotJob timeout in seconds (default 60)")
    parser.add_argument("--knotjob-workers", type=int, default=1,
                        help="parallel KnotJob processes (default 1)")
    args = parser.parse_args()

    from bounds_pipeline.knot_identification import DEFAULT_SOURCE_CSV, build_knot_table

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"knots_le{args.max_crossings}_invariants.json")
    build_knot_table(max_crossings=args.max_crossings,
                     source_csv=args.source_csv or DEFAULT_SOURCE_CSV,
                     out_path=out,
                     use_knotjob=not args.no_knotjob,
                     verbose=True,
                     knotjob_timeout=args.knotjob_timeout,
                     knotjob_workers=args.knotjob_workers)
    print(f"wrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
