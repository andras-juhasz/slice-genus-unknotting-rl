r"""
Exact unknotting numbers for positive links.

For a link L = K_1 u ... u K_\ell given by a diagram D that is
positive for some orientation of its components,
    u(L) = lk(L) + \sum_i u(K_i)
with lk(L) = \sum_{i<j} lk(K_i, K_j).  Write D_i for the
sub-diagram of D drawn by K_i.
The pipeline first bounds each u(K_i) from D_i.  For the lower
bound, we know that g_4 <= u, and for positive knots we have
    g_i = g_4(K_i) = (c(D_i) - O(D_i) + 1)/2;
since every component of a positive link is a positive knot,
we obtain this lower bound for every component.  For the upper
bound, u(K_i) <= floor(c(D_i)/2), where D_i may be replaced by
a simplified copy with fewer crossings.
It then identifies each K_i via knot_identification, with g_i
as a hint, and reads off the known value of u(K_i), or its best
known lower and upper bound when an exact value is not known.
Keeping the best of both gives a lower bound \underline u_i and
an upper bound \bar u_i on u(K_i), and the above formula
carries them over to the link:
    lk(L) + \sum_i \underline u_i <= u(L) <= lk(L) + \sum_i \bar u_i.
The lower bound is raised to the slice-torus bound
(c(D) - O(D) + \ell)/2 of the whole diagram when that is
larger, and u(L) is exact when the two bounds meet.

The same equality holds on the wider class of simply-linked
links, where slice_bounds_py.py applies it to every link it
certifies.  This module stays separate for what positivity adds
and simple linkedness does not: the orientation making the
diagram positive is searched for rather than assumed, c(D) and
O(D) are read off a diagram that must not be simplified, and g_i
is the exact genus rather than a bound on it.

Reference: [collari2021slicetoruspositive, Theorem 1.9] (the formula),
           [collari2021slicetoruspositive, Proposition 4.1] (simply linked),
           [nakamura2000positive, Theorem 1.1] (the genus)
"""

from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field, asdict

import pandas as pd
from spherogram import Link, Crossing

from bounds_pipeline.slice_bounds_py import _is_positive_diagram, _to_sage_link, signature
import bounds_pipeline.knot_identification as ki


# =============================================================================
# ORIENTATION-PARAMETERISED DIAGRAM PRIMITIVES
# =============================================================================
#
# A relative orientation is a vector eps in {+1,-1}^\ell indexed by link
# component, with eps[0] == +1.  The primitives read the unmodified
# spherogram diagram and apply eps on the fly: no Reidemeister move is ever
# performed, since one may destroy positivity.  spherogram conventions:
# positions 0,2 carry the under-strand and 1,3 the over-strand; the
# over-strand enters at 3 on a positive crossing and at 1 on a negative
# one; strand_components[0] and [1] are the components of the under- and
# over-strand.


def _entry_exit(crossing, eps: Sequence[int]) -> Tuple[Tuple[int, int],
                                                       Tuple[int, int]]:
    """((under_entry, under_exit), (over_entry, over_exit)) under eps."""
    under = (0, 2) if eps[crossing.strand_components[0]] == 1 else (2, 0)
    over_in = 3 if crossing.sign == 1 else 1
    over_out = (over_in + 2) % 4
    over = ((over_in, over_out) if eps[crossing.strand_components[1]] == 1
            else (over_out, over_in))
    return under, over


def crossing_sign_under(crossing, eps: Sequence[int]) -> int:
    """
    Sign of crossing under the relative orientation eps: the
    stored sign scaled by eps_a * eps_b, a and b the components
    of its two strands.
    """
    return (crossing.sign
            * eps[crossing.strand_components[0]]
            * eps[crossing.strand_components[1]])


def self_crossing_count(link: Link, i: int) -> int:
    """c(D_i): the crossings of link with both strands on component i."""
    return sum(1 for c in link.crossings
               if c.strand_components[0] == i == c.strand_components[1])


def _sage_seifert_circles(diagram: Link) -> int:
    """Sage's count of the Seifert circles of a diagram."""
    return len(_to_sage_link(diagram).seifert_circles())


def reoriented_link(link: Link, eps: Sequence[int], mirror: bool = False) -> Link:
    """
    Rebuilds link as an oriented diagram carrying the orientation
    eps, mirrored when mirror is set.  The 4-valent graph is copied
    as is and only the strand directions change.  Component
    indices are not preserved: spherogram re-derives
    link_components on rebuild.
    """
    old = list(link.crossings)
    index = {id(c): n for n, c in enumerate(old)}
    new = [Crossing(f'c{n}') for n in range(len(old))]
    for n, c in enumerate(old):
        for p in range(4):
            other, q = c.adjacent[p]
            new[n].adjacent[p] = (new[index[id(other)]], q)
    for n, c in enumerate(old):
        (u_in, _), (o_in, _) = _entry_exit(c, eps)
        new[n].make_tail(u_in)
        new[n].make_tail(o_in)
        new[n].orient()
    out = Link(new, check_planarity=False)
    out.unlinked_unknot_components = link.unlinked_unknot_components
    return out.mirror() if mirror else out


# =============================================================================
# POSITIVITY CERTIFICATION
# =============================================================================


@dataclass
class PositivityCertificate:
    """
    The relative orientation under which the diagram is positive;
    mirrored records that it was all-negative under orientation
    and is read mirrored.  Mirroring leaves the oriented
    resolution, hence c(D), O(D) and the sub-diagram counts,
    unchanged.
    """
    orientation: List[int]
    mirrored: bool


def certify_positive(link: Link) -> Optional[PositivityCertificate]:
    r"""
    Searches the 2^(\ell-1) relative orientations for one making
    link positive, or all-negative with a mirror.

    Returns:
        The PositivityCertificate, or None when the diagram
        certifies nothing; a positive link presented by a
        non-positive diagram is missed.
    """
    import itertools
    ell = len(link.link_components)
    if ell == 0 or not link.crossings:
        return None
    for tail in itertools.product((1, -1), repeat=ell - 1):
        eps = [1] + list(tail)
        signs = {crossing_sign_under(c, eps) for c in link.crossings}
        if signs == {1}:
            return PositivityCertificate(eps, mirrored=False)
        if signs == {-1}:
            return PositivityCertificate(eps, mirrored=True)
    return None


# =============================================================================
# DIAGRAM QUANTITIES
# =============================================================================


@dataclass
class DiagramData:
    """Everything the pipeline reads off the certified positive diagram.

    c_D and o_D are c(D) and O(D), the crossings and Seifert circles of
    the diagram; c_Di and o_Di the same for each sub-diagram D_i.
    """
    n_components: int
    c_D: int
    o_D: int
    c_Di: List[int]
    o_Di: List[int]
    lk: int
    genera: List[int]
    nu: int

    @property
    def lower_from_components(self) -> int:
        r"""lk(L) + \sum_i g_i, i.e. (c(D) - \sum_i O(D_i) + \ell)/2."""
        return self.lk + sum(self.genera)


def diagram_data(link: Link, cert: PositivityCertificate) -> DiagramData:
    """
    Computes c(D), O(D), lk and the component genera on the
    certified diagram, with the structural checks a parser bug
    would break: lk_parity, lk_nonneg, genus_integral.

    Args:
        link: the diagram.
        cert: its positivity certificate.

    Raises:
        ValueError: when a check fails.

    Reference: [nakamura2000positive, Theorem 1.1] (the component genera),
               [collari2021slicetoruspositive, Theorem 1.9] (lk)
    """
    eps = cert.orientation
    # Every component counts, including the crossingless ones spherogram keeps
    # in unlinked_unknot_components: they carry no crossing and no genus, but
    # o_D includes their circles.
    drawn = len(link.link_components)
    ell = drawn + link.unlinked_unknot_components
    c_D = len(link.crossings)
    o_D = (_sage_seifert_circles(reoriented_link(link, eps))
           + link.unlinked_unknot_components)

    c_Di, o_Di, genera = [], [], []
    for i in range(drawn):
        ci = self_crossing_count(link, i)
        # O(D_i) does not depend on the orientation, K_i reversing as a
        # whole; a component with no self-crossing is one round circle.
        sub = link.sublink(i)
        oi = _sage_seifert_circles(sub) if sub.crossings else 1
        if (ci - oi + 1) % 2 or ci - oi + 1 < 0:
            raise ValueError(f'genus_integral failed on component {i}: '
                             f'c(D_i)={ci}, O(D_i)={oi}, parser bug')
        c_Di.append(ci)
        o_Di.append(oi)
        genera.append((ci - oi + 1) // 2)
    # The crossingless components: one round circle, no crossing, no genus.
    for _ in range(link.unlinked_unknot_components):
        c_Di.append(0)
        o_Di.append(1)
        genera.append(0)

    numer = c_D - sum(c_Di)
    if numer % 2:
        raise ValueError(f'lk_parity failed: c(D)={c_D}, Σc(D_i)={sum(c_Di)}, '
                         'parser bug')
    lk = numer // 2
    if lk < 0:
        raise ValueError(f'lk_nonneg failed: lk={lk}, orientation bug')
    for i in range(drawn):
        for j in range(i + 1, drawn):
            pair = sum(1 for c in link.crossings
                       if {c.strand_components[0],
                           c.strand_components[1]} == {i, j})
            if pair % 2 or pair < 0:
                raise ValueError(f'lk_nonneg failed on the pair ({i},{j}): '
                                 f'{pair} crossings, orientation bug')

    if (c_D - o_D + ell) % 2:
        raise ValueError(f'nu is not integral: c(D)={c_D}, O(D)={o_D}, '
                         f'ell={ell}, parser bug')
    nu = (c_D - o_D + ell) // 2

    data = DiagramData(ell, c_D, o_D, c_Di, o_Di, lk, genera, nu)
    # Closed form of the component lower bound:
    #     lk + sum_i g_i == (c(D) - sum_i O(D_i) + ell)/2.
    closed = (c_D - sum(o_Di) + ell)
    if closed % 2 or closed // 2 != data.lower_from_components:
        raise ValueError('component lower bound disagrees with its closed '
                         f'form ({closed / 2} vs {data.lower_from_components})')
    return data


# =============================================================================
# THE PIPELINE
# =============================================================================


@dataclass
class PositiveUnknottingRecord:
    """One output row, exact or bounded."""
    link_id: Optional[str]
    orientation: List[int]
    mirrored: bool
    n_components: int
    c_D: int
    o_D: int
    c_Di: List[int]
    o_Di: List[int]
    components: List[Optional[str]]
    u_components: List[Optional[int]]
    lk: int
    u: Optional[int]
    u_lower: int
    u_upper: Optional[int]
    exact: bool
    nu_lower_bound: int
    lower_bound_source: str
    provenance: str
    checks_passed: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    # One entry per component: how the identification protocol went.  Keys:
    # index, crossings (of the simplified sub-diagram), domain_max, in_domain,
    # status, name, candidates, keys_used, u.  See format_identification_report.
    component_protocol: List[dict] = field(default_factory=list)
    pd_code: Optional[str] = None
    crossing_signs: Optional[str] = None


def _resolve_component(sub: Link, g: int, table: ki.KnotTable,
                       known: Optional[ki.KnownValueTable]
                       ) -> Tuple[Optional[str], int, int, dict]:
    r"""
    Resolves one component of a positive link against the knot
    table, with g, the exact genus of its positive sub-diagram,
    as identify()'s genus hint and as the floor of u(K_i).
    The ceiling is floor(c/2), c the crossings of the sub-diagram
    or of its simplified copy, lowered to the known upper bound.

    Args:
        sub: the component's sub-diagram.
        g: its genus, (c(D_i) - O(D_i) + 1)/2.
        table: the knot table.
        known: the known values, or None.

    Returns:
        (name, u_lower, u_upper, protocol_entry).

    Raises:
        ValueError: if g disagrees with the genus identify() finds.

    Reference: [collari2021slicetoruspositive, Theorem 1.9]
    """
    identification = ki.identify(sub, table, genus_hint=g)
    kv = ki.known_values(identification, table, known)
    u_lower = g if kv.u_lower is None else max(g, kv.u_lower)
    c = len(sub.crossings)
    if identification.crossings is not None:
        c = min(c, identification.crossings)
    u_upper = c // 2 if kv.u_upper is None else min(c // 2, kv.u_upper)
    candidates = (list(identification.names)
                 if identification.status == ki.Status.AMBIGUOUS else [])
    protocol = {
        'index': -1, 'genus': g, 'crossings': identification.crossings,
        'domain_max': identification.domain_max,
        'in_domain': (identification.crossings is not None
                      and identification.domain_max is not None
                      and identification.crossings <= identification.domain_max),
        'status': identification.status, 'name': identification.label,
        'candidates': candidates, 'keys_used': list(identification.keys_used),
        'u': u_lower if u_upper == u_lower else None}
    return identification.label, u_lower, u_upper, protocol


def positive_unknotting(link: Link,
                        table: Optional[ki.KnotTable] = None,
                        known: Optional[ki.KnownValueTable] = None,
                        link_id: Optional[str] = None,
                        record_diagram: bool = True,
                        check_sigma: bool = False
                        ) -> Optional[PositiveUnknottingRecord]:
    """
    Runs the whole pipeline on one diagram.

    Args:
        link: the diagram, with at least two components.
        table: the shared knot table; None loads knot_identification's
            default_table().
        known: the best known values; None loads
            knot_identification.load_known_values(table=table).
        link_id: name stored on the record.
        record_diagram: also store the certified PD code and crossing signs.
        check_sigma: also check sigma < 0 on the certified diagram.

    Returns:
        None when the diagram is not certified positive, or when the table
        is unavailable; otherwise a record, exact when every component
        resolved and bounded otherwise.

    Raises:
        ValueError: a structural check failed (a parser or orientation bug).

    Reference: [collari2021slicetoruspositive, Proposition 4.1 and Theorem 1.9]
    """
    if len(link.link_components) + link.unlinked_unknot_components < 2:
        # A knot goes through a different pipeline.  A knot together with a
        # crossingless unknot is a two-component link and does belong here.
        return None
    if table is None:
        table = ki.default_table()
    if table is None:
        return None
    if known is None:
        known = ki.load_known_values(table=table)
    cert = certify_positive(link)
    if cert is None:
        return None

    data = diagram_data(link, cert)
    checks = ['all_positive', 'lk_parity', 'lk_nonneg', 'genus_integral']

    names: List[Optional[str]] = []
    u_components: List[Optional[int]] = []
    lower_components: List[int] = []
    upper_components: List[int] = []
    unresolved: List[str] = []
    protocol: List[dict] = []
    for i, g in enumerate(data.genera):
        # A positive knot has g_3 = g_4 = g, so g = 0 means it bounds a
        # disk in S^3: the component is the unknot.
        if g == 0:
            names.append('0_1')
            u_components.append(0)
            lower_components.append(0)
            upper_components.append(0)
            protocol.append({'index': i, 'genus': 0, 'crossings': 0,
                             'domain_max': table.max_crossings,
                             'in_domain': True, 'status': ki.Status.IDENTIFIED,
                             'name': '0_1', 'candidates': [],
                             'keys_used': ['g'], 'u': 0})
            continue
        # Only the drawn components have a sub-diagram; a crossingless one has
        # g = 0 and was answered above.
        sub = link.sublink(i)
        if len(sub.crossings) != data.c_Di[i]:
            raise ValueError(f'sublink({i}) has {len(sub.crossings)} crossings '
                             f'but c(D_i) = {data.c_Di[i]}, parser bug')
        # A self-crossing keeps its sign under any reorientation, so the
        # sub-diagram is already positive unless the certificate mirrored
        # the whole diagram, which it must pick up too.
        if cert.mirrored:
            sub = sub.mirror()
        name, u_lower_i, u_upper_i, entry = _resolve_component(sub, g, table, known)
        entry['index'] = i
        names.append(name)
        exact_i = u_upper_i == u_lower_i
        u_components.append(u_lower_i if exact_i else None)
        lower_components.append(u_lower_i)
        upper_components.append(u_upper_i)
        protocol.append(entry)
        if not exact_i:
            unresolved.append(f'component {i}: {entry["status"]}'
                              + (f' {entry["candidates"]}' if entry['candidates']
                                 else ''))

    # Two lower bounds: take the max, and record which one dominates.  Each
    # \underline u_i is at least the g_i of data.lower_from_components, so the
    # sum below is that closed form or better.
    from_components = data.lk + sum(lower_components)
    u_lower = max(from_components, data.nu)
    if from_components > data.nu:
        source = 'components'
    elif data.nu > from_components:
        source = 'nu'
    else:
        source = 'tie'

    u_upper = data.lk + sum(upper_components)
    if u_upper < u_lower:
        raise ValueError(f'u_ge_nu failed: upper bound {u_upper} < lower bound '
                         f'{u_lower}, the component identification is wrong')

    # Exact whenever the bounds meet, which includes every component being
    # exact but also the slice-torus bound reaching the upper bound.
    exact = u_upper == u_lower
    u = u_upper if exact else None
    # Both provenances are the same theorem; the suffix records that the row
    # is an interval, some component having no exact u.
    if exact:
        checks.append('u_ge_nu')
        provenance = 'collari_positive'
    else:
        provenance = 'collari_positive_bounded'

    record = PositiveUnknottingRecord(
        link_id=link_id, orientation=list(cert.orientation),
        mirrored=cert.mirrored, n_components=data.n_components,
        c_D=data.c_D, o_D=data.o_D, c_Di=data.c_Di, o_Di=data.o_Di,
        components=names, u_components=u_components, lk=data.lk,
        u=u, u_lower=u_lower, u_upper=u_upper, exact=exact,
        nu_lower_bound=data.nu, lower_bound_source=source,
        provenance=provenance, checks_passed=checks, unresolved=unresolved,
        component_protocol=protocol)

    if record_diagram:
        certified = reoriented_link(link, cert.orientation, mirror=cert.mirrored)
        if not _is_positive_diagram(certified):
            raise ValueError('the certified diagram is not positive after '
                             'reorientation, orientation bug')
        record.pd_code = str([list(c) for c in certified.PD_code()])
        record.crossing_signs = str([c.sign for c in certified.crossings])
        # sigma < 0 for a non-trivial positive link (Przytycki): a warn-level
        # check, opt-in for its cost of one Seifert matrix per link.
        if check_sigma:
            import warnings
            sigma = int(signature(certified))
            if sigma < 0:
                checks.append('sigma_neg')
            else:
                warnings.warn(f'sigma_neg: σ = {sigma} ≥ 0 on the certified '
                              f'positive diagram of {link_id}')
    return record


def format_identification_report(record: 'PositiveUnknottingRecord') -> str:
    """
    One line per component of a record: the domain gate, what the
    key ladder concluded, and the knot identified; 'ok' when the
    component was identified, else the step that stopped it.
    """
    lines = [f"{record.link_id or 'link'}: {record.n_components} components, "
             f"lk = {record.lk}, "
             f"u = {record.u if record.exact else f'[{record.u_lower}, {record.u_upper}]'}"]
    for entry in record.component_protocol:
        head = (f"  component {entry['index']}: g = {entry['genus']}, "
                f"c(simplified) = {entry['crossings']}")
        if not entry['in_domain']:
            lines.append(f"{head} > {entry['domain_max']} = table domain "
                         f"-> OUT OF DOMAIN, no identification claimed")
            continue
        status = entry['status']
        if status == ki.Status.IDENTIFIED:
            via = (" via a diagrammatic connected sum" if '#' in
                  (entry['name'] or '') else '')
            lines.append(f"{head} <= {entry['domain_max']} -> ok{via}, "
                         f"identified as {entry['name']} (u = {entry['u']})")
        elif status == ki.Status.AMBIGUOUS:
            lines.append(f"{head} <= {entry['domain_max']} -> AMBIGUOUS, the "
                         f"key ladder leaves {entry['candidates']}")
        else:
            lines.append(f"{head} <= {entry['domain_max']} -> MISS, no entry "
                         f"matches (keys tried: {entry['keys_used']})")
    return "\n".join(lines)


def print_identification_report(dataset, table: Optional[ki.KnotTable] = None,
                                only_problems: bool = False,
                                limit: Optional[int] = None) -> dict:
    """
    Runs the identification protocol over a whole dataset and prints one block
    per certified link, then a summary.

    Args:
        dataset: a dataset.Dataset.
        table: the shared knot table; loaded from the default path when omitted.
        only_problems: print only the links where the protocol did not settle
            every component, i.e. where it broke and how.
        limit: stop printing after this many blocks; the summary still counts
            every link.

    Returns:
        The summary counters, so a caller can assert on them.
    """
    table = table or ki.default_table()
    known = ki.load_known_values(table=table) if table is not None else None
    counts = {'certified': 0, 'not_certified': 0, 'error': 0,
              'exact': 0, 'bounded': 0}
    per_component: Dict[str, int] = {}
    printed = 0
    for i in range(len(dataset)):
        link, name = dataset.links[i], dataset.names[i]
        try:
            record = positive_unknotting(link, table, known, link_id=name,
                                         record_diagram=False)
        except Exception as exc:                     # noqa: BLE001
            counts['error'] += 1
            print(f'{name}: identification failed: {type(exc).__name__}: {exc}')
            continue
        if record is None:
            counts['not_certified'] += 1
            continue
        counts['certified'] += 1
        counts['exact' if record.exact else 'bounded'] += 1
        settled = True
        for entry in record.component_protocol:
            key = entry['status'] if entry['in_domain'] else 'out_of_domain'
            per_component[key] = per_component.get(key, 0) + 1
            if key != ki.Status.IDENTIFIED:
                settled = False
        if (not only_problems or not settled) and (limit is None or printed < limit):
            print(format_identification_report(record))
            printed += 1
    print()
    print(f'links: {counts}')
    print(f'components: {per_component}')
    return {'links': counts, 'components': per_component}


# =============================================================================
# DATASET DRIVER
# =============================================================================


def positive_unknotting_for_dataset(dataset,
                                    table: Optional[ki.KnotTable] = None,
                                    strict: bool = True,
                                    verbose: bool = True) -> pd.DataFrame:
    """
    Runs the pipeline over a dataset.Dataset and cross-checks every
    exact value against the dataset's known unknotting number.

    Args:
        dataset: a dataset.Dataset.
        table: the shared knot table; loaded from the default path when omitted.
        strict: raise on a disagreement instead of warning.
        verbose: print a one-line summary.

    Returns:
        One row per certified link, with the PositiveUnknottingRecord fields.
    """
    import warnings
    if table is None:
        table = ki.default_table()
    known = ki.load_known_values(table=table) if table is not None else None

    records, disagreements = [], []
    n_certified = 0
    for bundle in dataset:
        link, known_u, name = bundle[0], bundle[1], bundle[-1]
        record = positive_unknotting(link, table, known, link_id=name)
        if record is None:
            continue
        n_certified += 1
        if record.exact and known_u is not None:
            if record.u not in known_u:
                disagreements.append((name, record.u, known_u))
            else:
                record.checks_passed.append('linkinfo_agree')
        records.append(record)

    if disagreements:
        message = ('linkinfo_agree failed on '
                   f'{len(disagreements)} link(s): ' + ', '.join(
                       f'{n}: computed {u}, known {k}'
                       for n, u, k in disagreements[:10]))
        if strict:
            raise ValueError(message)
        warnings.warn(message)

    df = pd.DataFrame([asdict(r) for r in records])
    if verbose:
        n_exact = int(df['exact'].sum()) if len(df) else 0
        print(f'{len(dataset)} links -> {n_certified} certified positive '
              f'-> {n_exact} exact, {n_certified - n_exact} bounded')
    return df


# =============================================================================
# FIXTURES
# =============================================================================


def torus_link(m: int, n: int) -> Link:
    r"""The (m, n) torus link, closure of (\sigma_1 ... \sigma_{m-1})^n."""
    word = [s for _ in range(n) for s in range(1, m)]
    return Link(braid_closure=word)


def torus_unknotting_number(m: int, n: int) -> int:
    """
    u(T(m,n)) = ((m-1)(n-1) + gcd(m,n) - 1)/2.

    Reference: [kawamura2002unknotting, Example 4.1]
    """
    from math import gcd
    return ((m - 1) * (n - 1) + gcd(m, n) - 1) // 2
