"""
Compute slice genus, strong slice genus, and unknotting number
bounds for a dataset and save the updated results.

Usage:
    PYTHONPATH=src sage -python \
        src/bounds_pipeline/compute_invariants_bounds.py <dataset.csv> \
        [--mt-primes 2 3] [--strong-mt-primes 2 3]

Outputs:
    - Prints bound statistics and improvements (slice genus, strong slice
      genus, and unknotting number)
    - Saves updated dataset, computed bounds and provenance to
      <dataset>_newBounds.csv, preserving original columns under input_...
"""
import sys
import os
import argparse

import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Compute slice genus, strong slice genus, and "
                    "unknotting number bounds for a dataset.")
    parser.add_argument("csv_path", help="Path to the dataset CSV file")
    parser.add_argument("--mt-primes", type=int, nargs="+", default=[2, 3],
                        help="Orders for the weak Murasugi-Tristram bound, each a "
                             "prime power greater than 1, evaluated at "
                             "omega = zeta_a (default: 2 3)")
    # default=None: the library's _STRONG_MT_PRIMES_DEFAULT applies.
    parser.add_argument("--strong-mt-primes", type=int, nargs="+", default=None,
                        help="Primes for the strong-slice Murasugi-Tristram "
                             "bound, evaluated at Tristram's "
                             "omega_p = zeta_p^((p-1)/2) (default: 2 3)")
    parser.add_argument("--simplify", action="store_true",
                        help="Simplify dataset link diagrams before computing any bound")
    parser.add_argument("--knotjob", action="store_true",
                        help="Compute the Rasmussen s-invariant via KnotJob for links "
                             "not covered by the positive closed form or the "
                             "alternating derivation (s = -sigma)")
    parser.add_argument("--knotjob-timeout", type=int, default=15,
                        help="Per-link KnotJob timeout in seconds (default 15)")
    parser.add_argument("--no-positive-unknotting", action="store_true",
                        help="Skip the positive-link unknotting pass (Collari "
                             "Thm. 1.9, u(L) = lk(L) + sum u(K_i)). On by "
                             "default; needs no external solver.")
    parser.add_argument("--no-component-bounds", action="store_true",
                        help="Skip the per-component bound pass on the "
                             "sub-diagrams of every multi-component link. It "
                             "powers g_4* >= sum_i g_4(K_i), "
                             "u >= |lk|(L) + sum_i u(K_i), the splitting-number "
                             "bound and the simply-linked upper bound on u. On "
                             "by default; it never calls KnotJob.")
    parser.add_argument("--no-component-identification", action="store_true",
                        help="Skip merging each component's best known g_4/u "
                             "(knot_identification.py) into the component "
                             "bound pass; the pipeline-only bounds are kept. "
                             "On by default; has no effect with "
                             "--no-component-bounds.")
    args = parser.parse_args()
    if not os.path.isfile(args.csv_path):
        print(f"Error: file not found: {args.csv_path}")
        sys.exit(1)

    from bounds_pipeline.bounds_report import (print_improved_bounds,
                               print_improved_unknotting_bounds)
    from dataset import Dataset
    from bounds_pipeline.slice_bounds_py import (
        _STRONG_MT_PRIMES_DEFAULT,
        _tristram_omega,
        _validate_mt_order,
        compute_bounds_for_dataset,
        update_dataset_with_bounds,
        update_dataset_with_unknotting_bounds,
    )

    # Validate the orders before the first link: the weak bound holds only at
    # roots of unity of prime-power order [kauffman1978signature, Theorem 4.1].
    try:
        for order in args.mt_primes:
            _validate_mt_order(order)
    except ValueError as exc:
        print(f"Error: --mt-primes: {exc}")
        sys.exit(1)

    # The strong bound needs primes [tristram1969cobordism, Theorem 2.27];
    # validate them up front too.
    if args.strong_mt_primes is None:
        args.strong_mt_primes = list(_STRONG_MT_PRIMES_DEFAULT)
    try:
        for order in args.strong_mt_primes:
            _tristram_omega(order)
    except ValueError as exc:
        print(f"Error: --strong-mt-primes: {exc}")
        sys.exit(1)

    input_df = pd.read_csv(args.csv_path)
    dataset = Dataset.from_dataframe(input_df)
    print(f"Loaded {len(dataset)} links from {args.csv_path}")
    print(f"Murasugi-Tristram primes (weak):   {args.mt_primes}")
    print(f"Murasugi-Tristram primes (strong): {args.strong_mt_primes}\n")

    results_df, summary = compute_bounds_for_dataset(
        dataset, verbose=True, mt_primes=args.mt_primes,
        strong_mt_primes=args.strong_mt_primes,
        simplify_dataset=args.simplify,
        use_knotjob=args.knotjob,
        knotjob_timeout=args.knotjob_timeout,
        use_positive_unknotting=not args.no_positive_unknotting,
        use_component_bounds=not args.no_component_bounds,
        use_component_identification=not args.no_component_identification)

    print_improved_bounds(results_df, dataset)
    print_improved_unknotting_bounds(results_df, dataset)

    update_dataset_with_bounds(dataset, results_df)
    update_dataset_with_unknotting_bounds(dataset, results_df)

    out_path = args.csv_path.removesuffix(".csv") + "_newBounds.csv"
    # Keep the original provenance even when a rerun reports only known_*.
    # Prefix input columns and omit duplicate result identity fields so the
    # canonical dataset columns remain unambiguous when the CSV is reloaded.
    bounds_df = results_df.set_index("index").drop(
        columns=["name", "PD_code"], errors="ignore")
    output_df = (dataset.to_dataframe()
                 .join(input_df.add_prefix("input_"))
                 .join(bounds_df))
    output_df.to_csv(out_path, index=False)
    print(f"\nSaved updated dataset to {out_path}")

if __name__ == "__main__":
    main()
