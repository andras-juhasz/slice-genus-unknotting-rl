"""
Text reports on the output of slice_bounds_py: one BoundsResult,
the DatasetBoundsSummary of a run, and the links whose known
invariants a results frame narrows.

Usage:
    from bounds_pipeline.bounds_report import (format_bounds,
                                               print_improved_bounds)
    print(format_bounds(compute_bounds(link)))
"""

import math
from typing import List


def _missing(v) -> bool:
    """True for None and NaN, the missing values of a results frame."""
    return v is None or (isinstance(v, float) and math.isnan(v))


def _fmt_known(v) -> str:
    """
    Compact string for a list of admissible values, as
    dataset.invariant_to_str writes it: a single value as itself,
    a contiguous run as [lo;hi], any other set as {a;b;c}.
    """
    if v is None:
        return ''
    v = list(v)
    if len(v) == 1:
        return str(v[0])
    if all(isinstance(x, int) for x in v) and \
            all(v[i + 1] == v[i] + 1 for i in range(len(v) - 1)):
        return f'[{v[0]};{v[-1]}]'
    return '{' + ';'.join(str(n) for n in v) + '}'


def _component_names(bounds, field: str) -> str:
    """
    One 'name (source)' entry per component, for the given
    ComponentBounds field ('genus_lower' or 'u_lower'); '?' where
    identification found nothing.
    """
    names = bounds.component_names or []
    sources = (bounds.component_genus_lower_source if field == 'genus_lower'
               else bounds.component_u_lower_source) or []
    return ', '.join(f"{name or '?'} ({src})" for name, src in
                     zip(names or [None] * len(sources), sources))


def format_bounds(bounds) -> str:
    """
    Rendering of a BoundsResult.

    Args:
        bounds: a BoundsResult, as returned by compute_bounds().
    """
    lines = []

    lines.append("=== Link Properties ===")
    lines.append(f"  Components: {bounds.num_components}")
    lines.append(f"  Crossings: {bounds.num_crossings}")
    lines.append(f"  Seifert circles: {bounds.num_seifert_circles}")
    lines.append(f"  Positive diagram: {bounds.is_positive}")
    lines.append(f"  Alternating diagram: {bounds.is_alternating}")
    lines.append(f"  Split: {bounds.is_split}")
    lines.append(f"  Algebraically split: {bounds.is_algebraically_split}")

    lines.append("\n=== Invariants ===")
    lines.append(f"  Writhe: {bounds.writhe}")
    lines.append(f"  Signature: {bounds.signature}")
    lines.append(f"  Nullity: {bounds.nullity}")

    lines.append("\n=== Rasmussen s-Invariant ===")
    if bounds.s_invariant is not None:
        lines.append(f"  s: {bounds.s_invariant}")
        if bounds.s_invariant_from_positive:
            lines.append("  Source: positive diagram formula (s = c(D) - O(D) + 1)")
        elif bounds.s_invariant_from_alternating_signature:
            lines.append("  Source: alternating non-split diagram (s = -σ)")
        elif bounds.s_invariant_from_knotjob:
            lines.append("  Source: KnotJob (Khovanov homology)")
        else:
            lines.append("  Source: unknown")
        if bounds.nu_s is not None:
            lines.append(f"  v_s = (s + ℓ - 1)/2: {bounds.nu_s:.1f}")
    else:
        lines.append("  s: NOT COMPUTED")
        if not bounds.is_positive and not bounds.is_negative:
            lines.append("  Reason: diagram is neither positive, negative, nor "
                         "alternating non-split, requires KnotJob")
        else:
            lines.append("  Reason: computation failed or not requested")

    lines.append("\n=== Slice Genus Bounds ===")
    lines.append(f"  Upper (Seifert): {bounds.slice_upper_seifert}")
    lines.append(f"  Lower (signature): {bounds.slice_lower_signature}")
    lines.append(f"  Lower (Murasugi-Tristram): {bounds.slice_lower_mt}")
    if bounds.slice_lower_nu_s is not None:
        lines.append(f"  Lower (nu_s): {bounds.slice_lower_nu_s}")
        if bounds.nu_s_bound1 is not None and bounds.nu_s_bound2 is not None:
            lines.append(f"    bound1 (g₄ ≥ ceil(-nu_s)): {bounds.nu_s_bound1}")
            lines.append(f"    bound2 (g₄ ≥ ceil(nu_s - ℓ + 1)): {bounds.nu_s_bound2}")
    if bounds.slice_lower_tau is not None:
        lines.append(f"  Lower (τ, HFK): {bounds.slice_lower_tau}")
    if bounds.slice_lower_nu_hfk is not None:
        lines.append(f"  Lower (ν, HFK): {bounds.slice_lower_nu_hfk}"
                     f"  [ν(K) = {bounds.hfk_nu}, ν(mK) = {bounds.hfk_nu_mirror}]")
    if bounds.slice_lower_fox_milnor_det is not None:
        square = ('a perfect square, no obstruction' if bounds.determinant_is_square
                  else 'not a perfect square, K is not slice')
        lines.append(f"  Lower (Fox-Milnor, determinant): "
                     f"{bounds.slice_lower_fox_milnor_det}")
        lines.append(f"    det(K) = {bounds.determinant} ({square})")
    if bounds.slice_lower_fox_milnor_full is not None:
        lines.append(f"  Lower (Fox-Milnor, full): "
                     f"{bounds.slice_lower_fox_milnor_full}")
        lines.append(f"    Δ_K = f(t)·f(1/t) admissible: "
                     f"{bounds.fox_milnor_satisfied}"
                     f"{'' if bounds.fox_milnor_satisfied else ', K is not slice'}")
    if bounds.slice_exact is not None:
        sign_str = "positive" if bounds.is_positive else "negative"
        lines.append(f"  EXACT ({sign_str} non-split): {bounds.slice_exact}")
    lines.append(f"  => Range: [{bounds.best_slice_lower}, {bounds.slice_upper_seifert}]")

    if not bounds.is_knot and bounds.is_algebraically_split:
        lines.append("\n=== Strong Slice Genus Bounds ===")
        lines.append(f"  Lower (from slice bounds): {bounds.strong_slice_lower_from_slice}")
        if bounds.strong_slice_lower_signature is not None:
            lines.append(f"  Lower (signature, alt. alg. split): {bounds.strong_slice_lower_signature}")
        if bounds.strong_slice_lower_pseudo_thin is not None:
            lines.append(f"  Lower (pseudo-thin, (|s|+ℓ-1)/2): {bounds.strong_slice_lower_pseudo_thin}")
        if bounds.strong_slice_lower_mt is not None:
            lines.append(f"  Lower (Murasugi-Tristram, m = ℓ): {bounds.strong_slice_lower_mt}")
        if bounds.strong_slice_lower_nu_s is not None:
            lines.append(f"  Lower (slice-torus |v_s| = |s+ℓ-1|/2): "
                         f"{bounds.strong_slice_lower_nu_s}")
        if bounds.strong_slice_lower_components is not None:
            lines.append(f"  Lower (components, Σ g₄(K_i)): "
                         f"{bounds.strong_slice_lower_components}"
                         f"  {bounds.component_genus_lower}")
            if bounds.component_names is not None:
                lines.append(f"    {_component_names(bounds, 'genus_lower')}")
        if bounds.strong_slice_lower_obstruction > 0:
            lines.append(f"  Lower (obstruction): {bounds.strong_slice_lower_obstruction}")
            lines.append(f"    Reason: {bounds.strong_slice_obstruction_reason}")
        lines.append(f"  Best lower: {bounds.best_strong_slice_lower}")

    lines.append("\n=== Strong Slice Genus (this link) ===")
    if bounds.strong_slice_genus_is_infinite:
        lines.append("  g₄* = ∞ (some pairwise linking number is non-zero)")
    elif bounds.num_crossings == 0 and bounds.num_components > 0:
        lines.append("  g₄* = 0 (crossingless unlink)")
    elif bounds.is_knot:
        lines.append(f"  g₄* = g₄ ∈ [{bounds.strong_slice_genus_lower:g}, "
                     f"{bounds.strong_slice_genus_upper:g}] (knot)")
    else:
        lines.append(f"  g₄* ≥ {bounds.strong_slice_genus_lower:g}; "
                     f"no finite upper bound is proved")

    lines.append("\n=== Unknotting Number Bounds (not through g₄) ===")
    lines.append(f"  Simply linked: {bounds.is_simply_linked}"
                 f"   |lk|(L) = {bounds.abs_linking_number}")
    if bounds.unknotting_lower_lt is not None:
        lines.append(f"  Lower (Levine-Tristram, Conway Thm 1): {bounds.unknotting_lower_lt}")
    if bounds.unknotting_lower_slice_torus is not None:
        lines.append(f"  Lower (slice-torus |v_s|): {bounds.unknotting_lower_slice_torus}")
    if bounds.unknotting_lower_components is not None:
        lines.append(f"  Lower (|lk| + Σ u(K_i)): {bounds.unknotting_lower_components}"
                     f"  {bounds.component_u_lower}")
        if bounds.component_names is not None:
            lines.append(f"    {_component_names(bounds, 'u_lower')}")
    if bounds.unknotting_lower_splitting is not None:
        lines.append(f"  Lower (splitting number): {bounds.unknotting_lower_splitting}")
    if bounds.unknotting_upper_simply_linked is not None:
        lines.append(f"  Upper (simply-linked, |lk| + Σ ū(K_i)): "
                     f"{bounds.unknotting_upper_simply_linked}")
    lines.append(f"  => best direct lower: {bounds.best_unknotting_lower_direct}")

    lines.append("\n=== Obstructions ===")
    lines.append(f"  Slice obstructed (g₄ ≥ 1): {bounds.slice_obstructed}")
    lines.append(f"  Strong slice obstructed (g₄* ≥ 1): {bounds.strong_slice_obstructed}")
    if bounds.strong_slice_obstructed and bounds.strong_slice_obstruction_reason:
        lines.append(f"    Reason: {bounds.strong_slice_obstruction_reason}")

    if bounds.hfk_computed:
        lines.append("\n=== HFK Invariants (knot) ===")
        if bounds.hfk_tau is not None:
            lines.append(f"  τ: {bounds.hfk_tau}")
        if bounds.hfk_nu is not None:
            lines.append(f"  ν: {bounds.hfk_nu}   ν(mirror): {bounds.hfk_nu_mirror}")
        if bounds.hfk_seifert_genus is not None:
            lines.append(f"  Seifert genus: {bounds.hfk_seifert_genus}")
        if bounds.hfk_epsilon is not None:
            lines.append(f"  ε: {bounds.hfk_epsilon}")
        if bounds.hfk_fibered is not None:
            lines.append(f"  Fibered: {bounds.hfk_fibered}")

    return "\n".join(lines)


def format_summary(summary) -> str:
    """
    Rendering of a DatasetBoundsSummary, with the
    inconsistency report appended when there is one.
    """
    lines = [
        "=" * 50,
        "SLICE GENUS BOUNDS - DATASET SUMMARY",
        "=" * 50,
        "",
        f"Total links: {summary.total_links}",
        f"  Knots: {summary.knot_count}",
        f"  Links (multi-component): {summary.total_links - summary.knot_count}",
        "",
        "--- Link Types ---",
        f"  Positive diagrams: {summary.positive_count}",
        f"  Alternating diagrams: {summary.alternating_count}",
        f"  Split links: {summary.split_count}",
        f"  Algebraically split: {summary.alg_split_count}",
        "",
        "--- Rasmussen s-Invariant Statistics ---",
        f"  s-invariant computed: {summary.s_invariant_computed_count}/{summary.total_links}",
        f"    From positive formula: {summary.s_invariant_from_positive_count}",
        f"    From alternating signature (s = -σ): {summary.s_invariant_from_alternating_signature_count}",
        f"    From KnotJob: {summary.s_invariant_from_knotjob_count}",
        "",
        "--- Fox-Milnor Statistics (knots only) ---",
        f"  Determinant test run: {summary.fox_milnor_det_checked_count}/{summary.total_links}",
        f"    det(K) not a perfect square => g4 >= 1: {summary.fox_milnor_det_obstruction_count}",
        f"  Full test run (last resort): {summary.fox_milnor_full_checked_count}",
        f"    Fox-Milnor violated => g4 >= 1: {summary.fox_milnor_full_obstruction_count}",
        "",
        "--- HFK Statistics (knots only) ---",
        f"  HFK computed: {summary.hfk_computed_count}/{summary.knot_count}",
        f"  Fibered knots: {summary.hfk_fibered_count}",
        f"  ν beat |τ| as a slice lower bound: {summary.nu_hfk_improved_count}",
        "",
        "--- Component / Orientation-Class Bounds ---",
        f"  Component pass ran: {summary.component_bounds_count} links",
        f"  Simply-linked diagrams: {summary.simply_linked_count}",
        f"  Σ g₄(K_i) is the best strong-slice bound: "
        f"{summary.strong_components_best_count}",
        f"  |ν_s| is the best strong-slice bound: {summary.strong_nu_s_best_count}",
        f"  Strong-slice lower raised by another orientation: "
        f"{summary.strong_slice_orientation_improved_count}",
        f"  A direct (non-g₄) bound is the best unknotting lower bound: "
        f"{summary.unknotting_direct_best_count}",
        f"  Components identified: {summary.component_identified_count}"
        f"  ambiguous: {summary.component_ambiguous_count}",
        f"  Links where a known component value beat the pipeline's own: "
        f"{summary.component_known_improved_count}",
        "",
    ]

    if summary.bounds_inconsistent_count > 0:
        lines.append("--- INTERNAL BOUNDS INCONSISTENCIES (BUG DETECTED) ---")
        lines.append(f"  Links with best_lower > best_upper: {summary.bounds_inconsistent_count}")
        lines.append(f"    Indices: {summary.bounds_inconsistent_indices[:20]}{'...' if len(summary.bounds_inconsistent_indices) > 20 else ''}")
        lines.append("")

    lines.append("--- Consistency with Known Values ---")

    if summary.slice_known_count > 0:
        pct = 100.0 * summary.slice_consistent_count / summary.slice_known_count
        lines.append(f"  Slice genus consistent: {summary.slice_consistent_count}/{summary.slice_known_count} ({pct:.1f}%)")
        if summary.slice_inconsistent_indices:
            lines.append(f"    Inconsistent indices: {summary.slice_inconsistent_indices[:10]}{'...' if len(summary.slice_inconsistent_indices) > 10 else ''}")
    else:
        lines.append("  No links with known slice genus")

    if summary.strong_slice_known_count > 0:
        pct = 100.0 * summary.strong_slice_consistent_count / summary.strong_slice_known_count
        lines.append(f"  Strong slice genus consistent: {summary.strong_slice_consistent_count}/{summary.strong_slice_known_count} ({pct:.1f}%)")

    if summary.unknotting_known_count > 0:
        pct = 100.0 * summary.unknotting_consistent_count / summary.unknotting_known_count
        lines.append(f"  Unknotting number consistent:   {summary.unknotting_consistent_count}/{summary.unknotting_known_count} ({pct:.1f}%)")
        if summary.unknotting_inconsistent_indices:
            lines.append(f"    Inconsistent indices: {summary.unknotting_inconsistent_indices[:10]}{'...' if len(summary.unknotting_inconsistent_indices) > 10 else ''}")

    if summary.positive_unknotting_certified_count > 0:
        lines.append(f"  Positive links certified:      "
                     f"{summary.positive_unknotting_certified_count} "
                     f"({summary.positive_unknotting_exact_count} exact via "
                     f"Collari Thm. 1.9, {summary.positive_unknotting_improved_count} "
                     f"rows where it is the best bound)")

    lines.append("")
    lines.append("--- Bound Tightness ---")
    if summary.slice_known_count > 0:
        pct = 100.0 * summary.slice_tight_count / summary.slice_known_count
        lines.append(f"  Bound equals known: {summary.slice_tight_count}/{summary.slice_known_count} ({pct:.1f}%)")

    lines.append("")
    lines.append("--- Best Bound Distribution ---")
    for combo, count in sorted(summary.best_bound_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / summary.total_links if summary.total_links > 0 else 0
        lines.append(f"  {combo}: {count} ({pct:.1f}%)")

    lines.append("=" * 50)

    inconsistency_report = format_inconsistency_report(summary)
    if inconsistency_report:
        lines.append("")
        lines.append(inconsistency_report)

    return "\n".join(lines)


def format_inconsistency_report(summary) -> str:
    """
    A report of every link whose computed bounds miss its known
    values: the known set, the computed bounds, and which
    individual bounds agree with the known set and which do not.
    Empty when there is no such link.
    """
    sections = (('SLICE GENUS', summary.slice_inconsistent_details),
                ('STRONG SLICE GENUS', summary.strong_slice_inconsistent_details),
                ('UNKNOTTING NUMBER', summary.unknotting_inconsistent_details))
    if not any(details for _, details in sections):
        return ""
    lines = ["", "=" * 60, "DETAILED INCONSISTENCY REPORT", "=" * 60]
    for title, details in sections:
        if not details:
            continue
        lines += ["", "-" * 60,
                  f"{title} INCONSISTENCIES ({len(details)} links)", "-" * 60]
        for d in details:
            lines += ["", f"Link: {d['name']} (index {d['index']})",
                      f"  Known: {_fmt_known(d['known'])}"]
            if d['upper'] is None:
                lines.append(f"  Best lower bound: {d['lower']}")
            else:
                lines.append(f"  Computed bounds: [{d['lower']}, {d['upper']}]")
            for heading, rows in (
                    ("INCONSISTENT bounds (a lower bound above every known "
                     "value, or an upper bound below all of them):",
                     d['inconsistent_bounds']),
                    ("Consistent bounds:", d['consistent_bounds'])):
                if rows:
                    lines.append(f"  {heading}")
                    lines += [f"    - {label}: {value}" for label, value in rows]
    lines += ["", "=" * 60]
    return "\n".join(lines)


def _print_interval_report(title: str, symbol: str, rows: List[dict]) -> None:
    """
    Prints which links a computed interval [lower, upper] improves
    on.  Each row is a dict with name, known (the admissible
    values, or None), lower, upper and, optionally, best (the
    names of the bounds, shown in brackets).
    """
    narrowed, exact, new_exact, new_nontrivial, new_trivial = [], [], [], [], []
    for r in rows:
        tag = f"  ({r['best']})" if r.get('best') else ''
        known = r['known']
        if known:
            lo, hi = max(min(known), r['lower']), min(max(known), r['upper'])
            if lo > min(known) or hi < max(known):
                head = f"  {r['name']}: {_fmt_known(known)} -> "
                if lo == hi:
                    exact.append(f"{head}{symbol} = {lo}{tag}")
                else:
                    narrowed.append(f"{head}[{lo}, {hi}]")
        elif r['lower'] == r['upper']:
            new_exact.append(f"  {r['name']}: {symbol} = {r['lower']}{tag}")
        elif r['lower'] > 0:
            new_nontrivial.append(f"  {r['name']}: {r['lower']} <= {symbol} "
                                  f"<= {r['upper']}{tag}")
        else:
            new_trivial.append(f"  {r['name']}: 0 <= {symbol} <= {r['upper']}")

    print("=" * 70)
    print(title)
    print("=" * 70)
    for heading, lines in (
            ("*** NEWLY EXACT (known value narrowed)", exact),
            ("--- Range narrowed", narrowed),
            ("*** PREVIOUSLY UNKNOWN, NOW EXACT", new_exact),
            ("--- Previously unknown, non-trivial lower bound", new_nontrivial),
            ("--- Previously unknown, trivial lower bound", new_trivial)):
        if lines:
            print(f"\n{heading}: {len(lines)} links\n")
            print("\n".join(lines))
    print(f"\nTotal: {len(narrowed) + len(exact)} improved, {len(exact)} exact | "
          f"{len(new_exact)} new exact, {len(new_nontrivial)} new non-trivial, "
          f"{len(new_trivial)} trivial")


def _print_strong_report(rows: List[dict]) -> None:
    """
    Prints which algebraically split links a strong-slice lower
    bound improves on.  Each row is a dict with name, nc (the
    number of components), known and lower.
    """
    narrowed, exact, new_nontrivial = [], [], []
    for r in rows:
        known, lower = r['known'], r['lower']
        head = f"  {r['name']} ({r['nc']} comp): "
        if known:
            if lower > min(known):
                new = [v for v in known if v >= lower]
                if len(new) == 1:
                    exact.append(f"{head}{_fmt_known(known)} -> g_4* = {new[0]}")
                else:
                    narrowed.append(f"{head}{_fmt_known(known)} -> "
                                    f"{_fmt_known(new)}  (lower bound: {lower})")
        elif lower > 0:
            new_nontrivial.append(f"{head}g_4* >= {lower}")

    print(f"\n{'=' * 70}")
    print("STRONG SLICE GENUS (algebraically split)")
    print("=" * 70)
    print(f"\nAlgebraically split links: {len(rows)}")
    for heading, lines in (
            ("*** NEWLY EXACT (lower bound matches known)", exact),
            ("--- Lower bound improved", narrowed),
            ("--- Previously unknown, non-trivial lower bound", new_nontrivial)):
        if lines:
            print(f"\n{heading}: {len(lines)} links\n")
            print("\n".join(lines))


def print_improved_bounds(results_df, dataset) -> None:
    """
    Prints the links whose slice or strong slice genus the results
    frame improves on or newly determines.

    Args:
        results_df: the frame of compute_bounds_for_dataset.
        dataset: the Dataset it was computed on.
    """
    from bounds_pipeline.slice_bounds_py import _strong_lower_of

    df = results_df[results_df['error'].isna()]
    rows = []
    for _, row in df.iterrows():
        best = row.get('best_bound_names', ['?'])
        rows.append({'name': row['name'],
                     'known': dataset.slice_genera[int(row['index'])],
                     'lower': int(row['best_slice_lower']),
                     'upper': int(row['slice_upper_seifert']),
                     'best': " + ".join(best) if isinstance(best, list) else best})
    _print_interval_report("SLICE GENUS: BOUNDS ANALYSIS", "g_4", rows)

    df_as = df[(~df['is_knot']) & (df['is_algebraically_split'])]
    _print_strong_report([
        {'name': row['name'], 'nc': row.get('num_components', '?'),
         'known': dataset.strong_slice_genera[int(row['index'])],
         'lower': _strong_lower_of(row)}
        for _, row in df_as.iterrows()])


def print_improved_unknotting_bounds(results_df, dataset) -> None:
    """
    Prints the links whose unknotting number the results frame
    improves on or newly determines.

    Args:
        results_df: the frame of compute_bounds_for_dataset.
        dataset: the Dataset it was computed on.
    """
    if 'unknotting_lower' not in results_df.columns:
        return
    df = results_df[results_df['error'].isna()]
    rows = []
    for _, row in df.iterrows():
        lower, upper = row.get('unknotting_lower'), row.get('unknotting_upper')
        if _missing(lower) or _missing(upper):
            continue
        rows.append({'name': row['name'],
                     'known': dataset.unknotting_nums[int(row['index'])],
                     'lower': int(lower), 'upper': int(upper)})
    _print_interval_report("UNKNOTTING NUMBER: BOUNDS ANALYSIS", "u", rows)
