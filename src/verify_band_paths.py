"""
Replays saved band paths and checks that each one is a
valid certificate for the value it claims.

The moves are applied with the state classes of
representable_py, since the action labels name crossings
of the diagrams these classes produce, and the Nim random
state saved before each end action makes the
simplification after it reproducible. Everything else is
recomputed here from the diagrams: the last diagram must
have no crossings; an unknotting path certifies its number
of crossing changes; a genus path certifies the genus
computed from the Euler characteristic of each connected
component of the surface that its bands and unknot births
build. The one thing read from the search code is the
connected component label (CCL) attached to each link
component, which tells the surface components apart across
simplifications. The answer of the search code is reported
alongside, for comparison.

A search run with simplification_switch=False keeps the
unsimplified diagram as its state, so its action labels
name crossings of those diagrams. Replay its band paths
with --no-simplification: the last diagram then keeps its
crossings, and it must simplify to a diagram with none.
That simplification is read from the search code, which
computes it after every end action and decides a win on
it; the saved random state makes it the same one.

Usage:
    python verify_band_paths.py FILE.csv [--framework slice]
        [--no-simplification]
"""
import argparse
import ast
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from representable import (bandmovestate_link,
                           bandmovestate_strand_ccls,
                           crosschangestate_link,
                           nim_set_random_state)
from representable_py import (BandMoveState, CrossChangeState,
                              DimensionExceededError)
from utils import link_from_crossings

FRAMEWORKS = ('unknotting', 'ribbon', 'slice',
              'strong ribbon', 'strong slice')
SLICE_FRAMEWORKS = ('slice', 'strong slice')
STRONG_FRAMEWORKS = ('strong ribbon', 'strong slice')

# Stem of a framework's columns in the band-path files:
# best_<stem>_band_path, <stem>_random_state, ...
COLUMN_STEM = {'unknotting': 'unknotting_num',
               'ribbon': 'ribbon_genus',
               'slice': 'slice_genus',
               'strong ribbon': 'strong_ribbon_genus',
               'strong slice': 'strong_slice_genus'}

# Value column of a framework in the final dataset files
DATASET_VALUE_COLUMN = {'unknotting': 'Unknotting Number',
                        'ribbon': 'Ribbon Genus-4D',
                        'slice': 'Genus-4D',
                        'strong ribbon': 'Strong Ribbon Genus-4D',
                        'strong slice': 'Strong Slice Genus-4D'}

NAME_COLUMNS = ('name', 'Name')
PD_COLUMNS = ('pd_code', 'PD Notation')
SIGNS_COLUMNS = ('crossing_signs', 'Crossing Signs')

# Largest diagram a replay accepts (crossings, components);
# a link beyond it is reported as skipped. Set with
# VERIFY_MAX_CROSSINGS and VERIFY_MAX_COMPONENTS; raising
# them costs memory in every replayed state.
MAX_CROSSINGS = int(os.environ.get('VERIFY_MAX_CROSSINGS', '500'))
MAX_COMPONENTS = int(os.environ.get('VERIFY_MAX_COMPONENTS', '50'))

# The twist cap limits the search, and a band with more
# twists is still a band, so the replay sets it out of reach
MAX_TWISTS = 10**4

_STRAND_ACTION = re.compile(r'^C(\d+)S([0-3])_(start|over|under|end[012])$')


class CertificateError(ValueError):
    """
    Raised when a band path does not certify its value: an
    illegal move, a last diagram with crossings, or a
    surface that breaks the rules of the framework.
    """


class MissingInputError(ValueError):
    """
    Raised when a band path cannot be replayed exactly, for
    instance because its random states are missing.
    """


@dataclass
class Verdict:
    """
    The outcome of one replay.

    Attributes:
        answer: the value certified by the band path, as
            recomputed here; None if the replay failed
        search_answer: the value computed by the search
            code on the replayed state
        history: whether the replayed diagrams equal the
            saved link history: "match", "differs at step
            k", "differs in length", or "" when no history
            is given
        start: the diagram the replay started from:
            "dataset", or "reoriented" for the saved
            diagram of an unknotting path that differs from
            the dataset one only in orientation
        reason: why the replay failed; "" if it did not
    """
    answer: Optional[int] = None
    search_answer: Optional[int] = None
    history: str = ''
    start: str = 'dataset'
    reason: str = ''


@dataclass
class _Piece:
    # one connected component of the surface: bands
    # attached, starting circles (original components
    # and births), original components, current circles
    bands: int
    start: int
    original: int
    circles: int


class _Surface:
    """
    The surface built by a band path, followed one
    connected component at a time. Components are keyed by
    CCL, and a band joining two components merges their
    labels.
    """

    def __init__(self, circles_by_ccl: Dict[int, int]):
        self.parent: Dict[int, int] = {}
        self.pieces: Dict[int, _Piece] = {}
        for ccl, n in circles_by_ccl.items():
            self.parent[ccl] = ccl
            self.pieces[ccl] = _Piece(0, n, n, n)

    def find(self, ccl: int) -> int:
        if ccl not in self.parent:
            raise CertificateError(f'unknown CCL {ccl}')
        while self.parent[ccl] != ccl:
            self.parent[ccl] = self.parent[self.parent[ccl]]
            ccl = self.parent[ccl]
        return ccl

    def add_band(self, ccl1: int, ccl2: int, fusion: bool,
                 strong: bool) -> None:
        """
        Attaches a band between circles with CCLs ccl1 and
        ccl2.

        Args:
            ccl1, ccl2: the CCLs at the two ends of the band
            fusion: whether the band joins two different
                circles (otherwise it splits one circle)
            strong: whether two original link components
                must stay on different surface components

        Raises:
            CertificateError: if the band breaks the strong
                condition, or splits a circle whose ends
                carry different CCLs
        """
        root1, root2 = self.find(ccl1), self.find(ccl2)
        if root1 != root2:
            if not fusion:
                raise CertificateError(
                    'a band splits a circle carrying two CCLs')
            piece1, piece2 = self.pieces[root1], self.pieces[root2]
            if strong and piece1.original and piece2.original:
                raise CertificateError(
                    'a band joins two components of the link')
            self.parent[root1] = root2
            del self.pieces[root1]
            piece2.bands += piece1.bands
            piece2.start += piece1.start
            piece2.original += piece1.original
            piece2.circles += piece1.circles
        piece = self.pieces[root2]
        piece.bands += 1
        piece.circles += -1 if fusion else 1

    def add_birth(self, ccl: int) -> None:
        if ccl in self.parent:
            raise CertificateError(f'a birth reuses CCL {ccl}')
        self.parent[ccl] = ccl
        self.pieces[ccl] = _Piece(0, 1, 0, 1)

    def check(self, circles_by_ccl: Dict[int, int]) -> None:
        """
        Compares the circles counted here with the CCLs of
        the replayed diagram.

        Raises:
            CertificateError: if the counts differ on some
                surface component
        """
        counted = Counter()
        for ccl, n in circles_by_ccl.items():
            counted[self.find(ccl)] += n
        for root, piece in self.pieces.items():
            if counted[root] != piece.circles:
                raise CertificateError(
                    f'surface component {root} has {piece.circles} '
                    f'circles here and {counted[root]} in the diagram')

    def genus(self, discard_closed: bool) -> int:
        """
        Returns the total genus, capping every circle with a
        disc. A component with o original link components,
        i births, k bands and m circles left has Euler
        characteristic i - k + m and o boundary circles,
        hence genus (2 + k - o - i - m) / 2.

        Args:
            discard_closed: whether to leave out components
                with no original link component, which are
                closed surfaces made of births only

        Raises:
            CertificateError: if some genus is negative or
                not an integer
        """
        total = 0
        for root, piece in self.pieces.items():
            twice = 2 + piece.bands - piece.start - piece.circles
            if twice < 0 or twice % 2:
                raise CertificateError(
                    f'surface component {root} has genus {twice}/2')
            if piece.original or not discard_closed:
                total += twice // 2
        return total


def _literal(value):
    """
    Parses a saved value: a Python literal, or a list in the
    dataset style with ";" between its entries.
    """
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value in ('', 'nan', 'None'):
        return None
    if value.startswith('[') and ';' in value and "'" not in value:
        value = value.replace(';', ',')
    return ast.literal_eval(value)


def _normalise_pd(pd: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    # relabel the edges 0..2n-1 in order of appearance
    labels: Dict[int, int] = {}
    return [tuple(labels.setdefault(e, len(labels)) for e in crossing)
            for crossing in pd]


def _same_diagram(pd1: Sequence[Sequence[int]],
                  pd2: Sequence[Sequence[int]],
                  turns: Sequence[int] = (0,)) -> bool:
    """
    Tells whether two PD codes with the same crossing order
    describe the same diagram, up to a relabelling of the
    edges and a cyclic turn of each crossing by one of the
    given amounts.

    Args:
        pd1, pd2: the PD codes
        turns: the allowed turns, in positions; (0,) asks
            for the same oriented diagram, and (0, 2) for
            the same diagram up to the orientation of its
            components, since reversing a component reads
            the crossings where it passes under from the
            other end of their under-strand

    Returns:
        Whether such a relabelling exists.
    """
    if len(pd1) != len(pd2):
        return False
    # visit the crossings so that each one, except the first
    # of each split piece, shares an edge with an earlier one
    crossings_at: Dict[int, List[int]] = {}
    for c, crossing in enumerate(pd1):
        for edge in crossing:
            crossings_at.setdefault(edge, []).append(c)
    order, seen = [], set()
    for root in range(len(pd1)):
        if root in seen:
            continue
        seen.add(root)
        stack = [root]
        while stack:
            c = stack.pop()
            order.append(c)
            for edge in pd1[c]:
                for d in crossings_at[edge]:
                    if d not in seen:
                        seen.add(d)
                        stack.append(d)

    def extend(k: int, to2: Dict[int, int], to1: Dict[int, int]) -> bool:
        if k == len(order):
            return True
        c = order[k]
        for turn in turns:
            new2, new1 = dict(to2), dict(to1)
            if all(new2.setdefault(pd1[c][i], pd2[c][(i + turn) % 4])
                   == pd2[c][(i + turn) % 4]
                   and new1.setdefault(pd2[c][(i + turn) % 4], pd1[c][i])
                   == pd1[c][i] for i in range(4)):
                if extend(k + 1, new2, new1):
                    return True
        return False

    return extend(0, {}, {})


def _parse_action(action: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    Returns the type of an action (start, over, under,
    twist, end or birth) and its crossing strand, if any.

    Raises:
        CertificateError: if the action is none of these
    """
    if action in ('cw_twist', 'ccw_twist'):
        return 'twist', None
    if action == 'unknot_birth':
        return 'birth', None
    match = _STRAND_ACTION.match(action)
    if match is None:
        raise CertificateError(f'unknown action {action!r}')
    kind = 'end' if match.group(3).startswith('end') else match.group(3)
    return kind, (int(match.group(1)), int(match.group(2)))


def _segments(actions: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """
    Splits a band path into bands, each closed by its end
    action, and unknot births.

    Raises:
        CertificateError: if a band does not open with a
            start action, or the path stops inside a band
    """
    segments, band = [], []
    for action in actions:
        kind, _ = _parse_action(action)
        if kind == 'birth':
            if band:
                raise CertificateError('an unknot birth inside a band')
            segments.append(('birth', [action]))
            continue
        if (kind == 'start') != (not band):
            raise CertificateError(f'{action} cannot be at position '
                                   f'{len(band)} of a band')
        band.append(action)
        if kind == 'end':
            segments.append(('band', band))
            band = []
    if band:
        raise CertificateError('the path stops inside a band')
    return segments


def _diagram(state) -> Tuple[List[List[int]], int]:
    # the PD code of the state's link, as the Nim code
    # numbers it, and its number of unlinked unknots
    if isinstance(state, CrossChangeState):
        link = json.loads(crosschangestate_link(state.state))
    else:
        link = json.loads(bandmovestate_link(state.state))
    pd = [list(crossing) for crossing in link['PD_code']]
    return pd, link['unlinked_unknot_components']


def _strand_circles(pd: Sequence[Sequence[int]]) -> List[List[int]]:
    """
    Returns, for each crossing strand, the index of the
    circle (link component) it lies on. Strands 0 and 2 of
    a crossing lie on one circle, strands 1 and 3 on
    another, and the two ends of an edge on the same one.
    """
    parent = list(range(4 * len(pd)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_end: Dict[int, int] = {}
    for c, crossing in enumerate(pd):
        parent[find(4 * c)] = find(4 * c + 2)
        parent[find(4 * c + 1)] = find(4 * c + 3)
        for s, edge in enumerate(crossing):
            if edge in first_end:
                parent[find(first_end[edge])] = find(4 * c + s)
            else:
                first_end[edge] = 4 * c + s
    return [[find(4 * c + s) for s in range(4)] for c in range(len(pd))]


def _circles_by_ccl(state: BandMoveState) -> Dict[int, int]:
    """
    Counts the circles of the state's link by CCL.

    Raises:
        CertificateError: if one circle carries two CCLs
    """
    pd, unlinked = _diagram(state)
    strand_ccls, unlinked_ccls = bandmovestate_strand_ccls(state.state)
    ccl_of: Dict[int, int] = {}
    for c, row in enumerate(_strand_circles(pd)):
        for s, circle in enumerate(row):
            if ccl_of.setdefault(circle, strand_ccls[c][s]) != strand_ccls[c][s]:
                raise CertificateError('one circle carries two CCLs')
    if len(unlinked_ccls) != unlinked:
        raise CertificateError(f'{unlinked} unlinked unknots with '
                               f'{len(unlinked_ccls)} CCLs')
    return Counter(list(ccl_of.values()) + list(unlinked_ccls))


def _apply(state, action: str):
    try:
        return state.next_state(action)
    except DimensionExceededError as e:
        raise MissingInputError(f'diagram too large: {e}') from e
    except Exception as e:
        raise CertificateError(f'{action} is illegal: '
                               f'{str(e).splitlines()[0]}') from e


def verify_band_path(pd_code, crossing_signs, framework: str,
                     band_path, random_states,
                     link_history=None,
                     simplification_switch: bool = True) -> Verdict:
    """
    Replays a band path from a diagram and recomputes the
    value it certifies.

    Args:
        pd_code: the PD code of the starting diagram, the
            one the action labels refer to
        crossing_signs: the signs of its crossings, in the
            order of pd_code
        framework: one of FRAMEWORKS
        band_path: the saved actions, as (action, reward)
            pairs or action names; the rewards are ignored
        random_states: the Nim random states saved before
            each end action, one per band
        link_history: the saved diagrams after each band or
            birth, compared with the replay if given
        simplification_switch: the simplification_switch of
            the search that found the band path; with False
            the replayed diagrams stay unsimplified and the
            last one must simplify to no crossings

    Returns:
        A Verdict; its reason is empty if the band path is a
        valid certificate for its answer.

    Raises:
        ValueError: if framework is not one of FRAMEWORKS
    """
    if framework not in FRAMEWORKS:
        raise ValueError(f'unknown framework {framework!r}')
    verdict = Verdict()
    try:
        actions = [a if isinstance(a, str) else a[0]
                   for a in _literal(band_path) or []]
        if not actions:
            raise MissingInputError('no band path')
        states = _literal(random_states) or []
        history = _literal(link_history) if link_history else None
        segments = _segments(actions)
        num_bands = sum(kind == 'band' for kind, _ in segments)
        if len(states) != num_bands:
            raise MissingInputError(f'{len(states)} random states for '
                                    f'{num_bands} bands')
        pd, signs = _literal(pd_code), _literal(crossing_signs)
        if framework == 'unknotting' and history:
            # the unknotting number does not depend on the
            # orientation, and a search may have run on another
            # orientation of the dataset diagram
            saved_pd, saved_signs = history[0][0], history[0][2]
            if (not _same_diagram(pd, saved_pd)
                    and _same_diagram(pd, saved_pd, (0, 2))):
                pd, signs = saved_pd, saved_signs
                verdict.start = 'reoriented'
        link = link_from_crossings(_normalise_pd(pd), signs)
        cls = (CrossChangeState if framework == 'unknotting'
               else BandMoveState)
        try:
            state = cls.from_link(link, max_twists=MAX_TWISTS,
                                  framework=framework,
                                  shape=(MAX_CROSSINGS, MAX_COMPONENTS),
                                  simplification_switch=simplification_switch)
        except DimensionExceededError as e:
            raise MissingInputError(f'diagram too large: {e}') from e
        surface = (None if framework == 'unknotting'
                   else _Surface(_circles_by_ccl(state)))
        if history is not None:
            verdict.history = ('match' if len(history) == len(segments) + 1
                               else 'differs in length')
        bands_done = 0
        for step, (kind, segment) in enumerate(segments, 1):
            if kind == 'birth':
                if framework not in SLICE_FRAMEWORKS:
                    raise CertificateError(f'unknot birth in {framework}')
                known = set(surface.parent)
                state = _apply(state, segment[0])
                circles = _circles_by_ccl(state)
                new = [ccl for ccl in circles if ccl not in known]
                if len(new) != 1 or circles[new[0]] != 1:
                    raise CertificateError('a birth did not add one circle')
                surface.add_birth(new[0])
                surface.check(circles)
            else:
                for action in segment[:-1]:
                    state = _apply(state, action)
                if surface is not None:
                    _, (c1, s1) = _parse_action(segment[0])
                    _, (c2, s2) = _parse_action(segment[-1])
                    pd, _ = _diagram(state)
                    if not (c1 < len(pd) and c2 < len(pd)):
                        raise CertificateError('a band leaves the diagram')
                    circle = _strand_circles(pd)
                    strand_ccls, _ = bandmovestate_strand_ccls(state.state)
                    surface.add_band(strand_ccls[c1][s1], strand_ccls[c2][s2],
                                     circle[c1][s1] != circle[c2][s2],
                                     framework in STRONG_FRAMEWORKS)
                nim_set_random_state(states[bands_done])
                bands_done += 1
                state = _apply(state, segment[-1])
                if surface is not None:
                    surface.check(_circles_by_ccl(state))
            if verdict.history == 'match':
                # the history is saved with the edge numbering
                # of the Python Link, so compare in that form
                replayed = state.link()
                saved_pd, saved_unlinked = history[step][0], history[step][1]
                if ([list(c) for c in saved_pd]
                        != [list(c) for c in replayed.PD_code()]
                        or saved_unlinked != replayed.unlinked_unknot_components):
                    verdict.history = f'differs at step {step}'
        pd, unlinked = _diagram(state)
        if simplification_switch:
            if pd:
                raise CertificateError(f'the last diagram has {len(pd)} crossings')
        else:
            # the simplification the search computed after the
            # last end action, reproduced from its random state
            left = state.simplified_link_num_crossings()
            if left:
                raise CertificateError(f'the last diagram simplifies to '
                                       f'{left} crossings')
        if surface is None:
            # each band of an unknotting path is one crossing change
            verdict.answer = num_bands
        else:
            # every circle of the last diagram, with or without
            # crossings, is a boundary circle of the surface
            circles = len({circle for row in _strand_circles(pd)
                           for circle in row}) + unlinked
            if sum(p.circles for p in surface.pieces.values()) != circles:
                raise CertificateError('the circle count disagrees with the '
                                       'last diagram')
            verdict.answer = surface.genus(framework in SLICE_FRAMEWORKS)
        try:
            verdict.search_answer = state.compute_current_answer(
                link, len(segments))
        except Exception:
            verdict.search_answer = None
    except (CertificateError, MissingInputError) as e:
        verdict.answer = None
        verdict.reason = f'{type(e).__name__}: {e}'
    except (SyntaxError, ValueError, TypeError, IndexError) as e:
        verdict.answer = None
        verdict.reason = f'MissingInputError: unreadable input ({e})'
    return verdict


def claimed_upper(value) -> Optional[int]:
    """
    Returns the upper end of a saved value such as "1",
    "[1;2]" or "[1, 2]", or None if it holds no integer.
    """
    numbers = [int(n) for n in re.findall(r'\d+', str(value or ''))]
    return max(numbers) if numbers else None


def status(verdict: Verdict, claimed: Optional[int]) -> str:
    """
    Classifies a verdict against the claimed upper bound.

    Returns:
        "invalid" if the band path certifies nothing,
        "skipped" if it could not be replayed, otherwise
        "match", "below" (it certifies a value below the
        claim) or "above" (a value above the claim, which
        then comes from elsewhere); "unclaimed" if there is
        no claim.
    """
    if verdict.reason.startswith('CertificateError'):
        return 'invalid'
    if verdict.reason:
        return 'skipped'
    if claimed is None:
        return 'unclaimed'
    if verdict.answer == claimed:
        return 'match'
    return 'below' if verdict.answer < claimed else 'above'


def _pick(columns: Sequence[str], candidates: Sequence[str],
          what: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise SystemExit(f'no {what} column: tried {", ".join(candidates)}')


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Command-line entry point.

    Returns:
        The exit status: 1 if some band path is invalid,
        0 otherwise.
    """
    parser = argparse.ArgumentParser(
        description='Replay saved band paths and check the values '
                    'they certify.')
    parser.add_argument('csv', help='a dataset or band-path CSV file')
    parser.add_argument('--framework', nargs='+',
                        type=lambda s: s.replace('_', ' '),
                        choices=FRAMEWORKS,
                        help='frameworks to check (default: every '
                             'framework with a band-path column)')
    parser.add_argument('--path-column', help='band-path column')
    parser.add_argument('--states-column', help='random-state column')
    parser.add_argument('--history-column', help='link-history column')
    parser.add_argument('--value-column', help='claimed-value column')
    parser.add_argument('--name', action='append',
                        help='check only this link (repeatable)')
    parser.add_argument('--out', help='write one report row per path here')
    parser.add_argument('--no-simplification', action='store_true',
                        help='the band paths come from a search run with '
                             'simplification_switch=False')
    args = parser.parse_args(argv)

    csv.field_size_limit(sys.maxsize)
    with open(args.csv, newline='') as f:
        columns = csv.DictReader(f).fieldnames or []
    frameworks = args.framework or [
        fw for fw in FRAMEWORKS
        if f'best_{COLUMN_STEM[fw]}_band_path' in columns]
    overrides = (args.path_column, args.states_column,
                 args.history_column, args.value_column)
    if len(frameworks) != 1 and any(overrides):
        parser.error('column options need exactly one --framework')
    name_col = _pick(columns, NAME_COLUMNS, 'name')
    pd_col = _pick(columns, PD_COLUMNS, 'PD code')
    signs_col = _pick(columns, SIGNS_COLUMNS, 'crossing signs')

    plan = []
    for fw in frameworks:
        stem = COLUMN_STEM[fw]
        path_col = args.path_column or f'best_{stem}_band_path'
        states_col = args.states_column or f'{stem}_random_state'
        history_col = args.history_column or f'{stem}_link_history'
        value_col = args.value_column or next(
            (c for c in (stem, DATASET_VALUE_COLUMN[fw]) if c in columns), None)
        if path_col not in columns:
            raise SystemExit(f'no column {path_col!r} for {fw}')
        plan.append((fw, path_col, states_col,
                     history_col if history_col in columns else None,
                     value_col))

    report = []
    with open(args.csv, newline='') as f:
        for row in csv.DictReader(f):
            name = row[name_col]
            if args.name and name not in args.name:
                continue
            for fw, path_col, states_col, history_col, value_col in plan:
                if (row.get(path_col) or '').strip() in ('', 'nan', 'None'):
                    continue
                verdict = verify_band_path(
                    row[pd_col], row[signs_col], fw, row[path_col],
                    row.get(states_col),
                    row.get(history_col) if history_col else None,
                    simplification_switch=not args.no_simplification)
                claimed = claimed_upper(row.get(value_col)) if value_col else None
                report.append({'name': name, 'framework': fw,
                               'status': status(verdict, claimed),
                               'answer': verdict.answer,
                               'claimed': claimed,
                               'search_answer': verdict.search_answer,
                               'history': verdict.history,
                               'start': verdict.start,
                               'reason': verdict.reason})

    if args.out:
        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(report[0]) if report
                                    else ['name'])
            writer.writeheader()
            writer.writerows(report)
    for fw in frameworks:
        rows = [r for r in report if r['framework'] == fw]
        counts = Counter(r['status'] for r in rows)
        disagree = sum(r['answer'] is not None
                       and r['answer'] != r['search_answer'] for r in rows)
        print(f'{fw}: {len(rows)} band paths, '
              + ', '.join(f'{k} {v}' for k, v in sorted(counts.items()))
              + f'; answer differs from the search code on {disagree}')
        for r in rows:
            if r['status'] in ('invalid', 'skipped'):
                print(f'  {r["status"]} {r["name"]}: {r["reason"]}')
    return 1 if any(r['status'] == 'invalid' for r in report) else 0


if __name__ == '__main__':
    sys.exit(main())
