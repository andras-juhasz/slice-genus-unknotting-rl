r"""
Identification of knots with at most ten crossings, and the best
known values of the knots it identifies.

A knot diagram is simplified, and invariants computed on the
simplified copy are compared with a table of every knot with at
most max_crossings crossings: the unknot, the prime knots in both
chiralities, and the connected sums of prime knots whose crossing
numbers add up to at most max_crossings.  The table is written by
src/bounds_pipeline/build_knot_table.py (see build_knot_table).

A match is a proof under three conditions.  A diagram with n'
crossings shows c(K) <= n', and the prime knots up to ten crossings
are tabulated completely, so a prime knot with n' <= 10 is in the
table. 

Completeness for composite knots follows from the finite classification
of Cantarella, Chapman, and Mastin, "Knot Probabilities in Random
Diagrams" (2016), arXiv:1512.05749: every composite knot admitting
an n'-crossing diagram, n' <= 10, has prime summands whose crossing
numbers total at most n'. Thus the candidate cutoff retains the
actual knot type. Last, uniqueness is 
checked on every query and never assumed of the table: a surviving 
entry is verified against every key, and several survivors are 
reported as such.

The keys, in the order they are computed:
    g, tau, hfk_rank   knot Floer homology, one computation
    sig                the signature of a Seifert matrix V
    det, alex          the Alexander polynomial det(V - tV^T)
    jones              the Jones polynomial, as (exponent,
                       coefficient) pairs
sig and tau change sign under mirroring and the exponents of jones
are negated; the other keys are mirror-invariant.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sage.all import QQ
from spherogram import Link

from bounds_pipeline.slice_bounds_py import (_alexander_polynomial_from_seifert, _compute_hfk,
                             _is_alternating_diagram, _is_positive_diagram,
                             _is_split, _s_invariant_positive,
                             _sage_seifert_matrix_from_sage_link,
                             _sage_signature_and_nullity_from_matrix,
                             _to_sage_link)


# This directory, which holds the two files the identification
# reads, and the repository above it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, os.pardir, os.pardir)

# Default input of build_knot_table: the prime knots up to ten
# crossings, with KnotInfo's names and PD codes.
DEFAULT_SOURCE_CSV = os.path.join(_ROOT, 'datasets', 'rolfsen.csv')

# The table default_table() reads, beside this module.  The
# environment variable KNOT_TABLE_JSON points it at another file.
DEFAULT_TABLE_JSON = os.path.join(_HERE, 'knots_le10_invariants.json')

# The best known values load_known_values() reads, beside this
# module.  The environment variable KNOWN_KNOT_VALUES_CSV points it
# at another file; without the file, identification runs without
# known values.
DEFAULT_KNOWN_VALUES_CSV = os.path.join(_HERE, 'KnotsTo13-bestKnown.csv')

# Largest key diagram the Jones polynomial is computed on, its cost
# growing quickly with the crossings; a larger diagram gives MISSING.
# A ten-crossing table never reaches it.  Raise it with the
# environment variable KNOT_ID_JONES_MAX_CROSSINGS only for a larger
# table, at the price of time.
JONES_MAX_CROSSINGS = int(os.environ.get('KNOT_ID_JONES_MAX_CROSSINGS', '20'))

# The key ladder, in the order the keys are computed and compared.
LADDER: Tuple[str, ...] = ('g', 'tau', 'hfk_rank', 'sig', 'det', 'alex',
                           'jones')

# The computation each key comes from; the keys of one rung are
# computed together.
_RUNG = {'g': 'hfk', 'tau': 'hfk', 'hfk_rank': 'hfk',
         'sig': 'seifert',
         'det': 'alexander', 'alex': 'alexander',
         'jones': 'jones'}

# Name of the unknot entry.
UNKNOT = '0_1'

# Provenance prefixes of the values known_values() returns: a value
# read for an identified knot, a sub-additive upper bound on a
# composite knot, and an s-invariant read from the table.  The
# unknot's values carry the bare UNKNOT_SOURCE.
KNOWN_SOURCE = 'known'
SUBADDITIVE_SOURCE = 'subadditive'
TABLE_SOURCE = 'table'
UNKNOT_SOURCE = 'unknot'

_PRIME_NAME = re.compile(r'(\d+)_(\d+)')
_SUMMAND_NAME = re.compile(r'(m\()?(\d+)_(\d+)(\))?')


class Status:
    """The outcomes of identify()."""
    IDENTIFIED = 'identified'
    AMBIGUOUS = 'ambiguous'
    MISSING = 'missing'
    OUT_OF_DOMAIN = 'out_of_domain'


# =============================================================================
# NAMES
# =============================================================================
#
# A prime entry is named K, as in KnotInfo, or m(K) for its mirror.
# A composite entry joins its summands with '#', sorted by crossing
# number, then KnotInfo index, then mirror flag: 3_1#m(3_1), 3_1#4_1.

def _summand_key(name: str) -> Tuple[int, int, bool]:
    """
    Sort key of one prime name: (crossing number, index,
    mirrored).

    Raises:
        ValueError: if name is not K or m(K) for a knot name K.
    """
    m = _SUMMAND_NAME.fullmatch(name)
    if m is None or bool(m.group(1)) != bool(m.group(4)):
        raise ValueError(f'not a prime knot name: {name!r}')
    return int(m.group(2)), int(m.group(3)), bool(m.group(1))


def name_order(name: str) -> tuple:
    """Sort key of any entry name; the unknot comes first."""
    if name == UNKNOT:
        return ()
    return tuple(_summand_key(part) for part in name.split('#'))


def composite_name(summands: Sequence[str]) -> str:
    """The canonical name of the connected sum of summands."""
    return '#'.join(sorted(summands, key=_summand_key))


def base_name(summand: str) -> str:
    """The KnotInfo name of a prime entry: m(3_1) gives 3_1."""
    c, index, _ = _summand_key(summand)
    return f'{c}_{index}'


def mirror_name(name: str) -> str:
    """The name of the mirror image; a composite mirrors every summand."""
    if name == UNKNOT:
        return name
    parts = []
    for part in name.split('#'):
        c, index, mirrored = _summand_key(part)
        parts.append(f'{c}_{index}' if mirrored else f'm({c}_{index})')
    return composite_name(parts)


def mirror_class(name: str) -> str:
    """
    The representative of {name, mirror_name(name)}, i.e. of the
    knot up to chirality: the one that sorts first.
    """
    return min(name, mirror_name(name), key=name_order)


# =============================================================================
# THE TABLE
# =============================================================================

@dataclass
class TableEntry:
    """
    One knot of the table.  summands lists the prime factors with
    their mirror flags (a prime entry lists itself, the unknot
    nothing); pd_code and crossing_signs are the spherogram diagram
    the keys were computed on; s_source says where s comes from
    ('knotjob', 'positive_formula', 'alternating_signature',
    'additivity' for a composite, 'unknot'), None when s is unknown.
    """
    name: str
    summands: Tuple[str, ...]
    prime: bool
    crossing_number: int
    pd_code: List[List[int]]
    crossing_signs: List[int]
    g: int
    tau: int
    hfk_rank: int
    det: int
    alex: Tuple[int, ...]
    sig: int
    jones: Tuple[Tuple[int, int], ...]
    s: Optional[int] = None
    s_source: Optional[str] = None

    def keys(self) -> dict:
        """The values of the LADDER keys."""
        return {key: getattr(self, key) for key in LADDER}

    def to_json(self) -> dict:
        """The entry as plain JSON types."""
        return {'name': self.name, 'summands': list(self.summands),
                'prime': self.prime,
                'crossing_number': self.crossing_number,
                'pd_code': [list(row) for row in self.pd_code],
                'crossing_signs': list(self.crossing_signs),
                'g': self.g, 'tau': self.tau, 'hfk_rank': self.hfk_rank,
                'det': self.det, 'alex': list(self.alex), 'sig': self.sig,
                'jones': [list(pair) for pair in self.jones],
                's': self.s, 's_source': self.s_source}

    @classmethod
    def from_json(cls, data: dict) -> 'TableEntry':
        """Inverse of to_json."""
        return cls(name=data['name'], summands=tuple(data['summands']),
                   prime=bool(data['prime']),
                   crossing_number=int(data['crossing_number']),
                   pd_code=[list(row) for row in data['pd_code']],
                   crossing_signs=list(data['crossing_signs']),
                   g=int(data['g']), tau=int(data['tau']),
                   hfk_rank=int(data['hfk_rank']), det=int(data['det']),
                   alex=tuple(data['alex']), sig=int(data['sig']),
                   jones=tuple(tuple(pair) for pair in data['jones']),
                   s=data.get('s'), s_source=data.get('s_source'))


class KnotTable:
    """
    The knots identify() compares a diagram with.

    Args:
        entries: the TableEntry records.
        report: the build report.
        max_crossings: the crossing number the table is complete
            up to; defaults to report['max_crossings'].
        identity: (path, modification time) of the file the table
            was read from, used in cache keys; None in memory.
    """

    def __init__(self, entries: Sequence[TableEntry],
                 report: Optional[dict] = None,
                 max_crossings: Optional[int] = None,
                 identity: Optional[tuple] = None):
        self.entries: List[TableEntry] = sorted(
            entries, key=lambda e: (e.crossing_number, name_order(e.name)))
        self.report: dict = dict(report or {})
        if max_crossings is None:
            max_crossings = self.report.get(
                'max_crossings',
                max((e.crossing_number for e in self.entries), default=0))
        self.max_crossings = int(max_crossings)
        self.identity = identity
        self._by_name = {e.name: e for e in self.entries}
        self.prime_names = frozenset(base_name(e.name) for e in self.entries
                                     if e.prime)

    def by_name(self, name: str) -> TableEntry:
        """
        The entry called name.

        Raises:
            KeyError: if there is none.
        """
        return self._by_name[name]

    @property
    def cache_key(self) -> tuple:
        """What identifies this table in a cache key."""
        return self.identity if self.identity is not None else ('memory', id(self))

    def to_json(self, path: str) -> None:
        """
        Writes the table.  Entries keep their sorted order and keys
        are sorted, so equal tables give identical files.
        """
        payload = {'max_crossings': self.max_crossings,
                   'report': self.report,
                   'entries': [e.to_json() for e in self.entries]}
        with open(path, 'w') as fh:
            fh.write(json.dumps(payload, indent=1, sort_keys=True))
            fh.write('\n')

    @classmethod
    def from_json(cls, path: str) -> 'KnotTable':
        """Reads a table written by to_json."""
        with open(path) as fh:
            payload = json.load(fh)
        return cls([TableEntry.from_json(e) for e in payload['entries']],
                   report=payload.get('report', {}),
                   max_crossings=payload.get('max_crossings'),
                   identity=_file_identity(path))


def _file_identity(path: str) -> Optional[tuple]:
    """(real path, modification time) of a file, None if it is missing."""
    real = os.path.realpath(path)
    if not os.path.isfile(real):
        return None
    return real, os.path.getmtime(real)


# =============================================================================
# KEYS
# =============================================================================

def key_diagram(knot: Link) -> Link:
    """
    The diagram the keys are computed on: a copy of knot after
    spherogram's global simplification, whose crossing count n'
    bounds the crossing number of the knot.  Simplifying is
    needed: Sage's braid conversion can recurse without bound on
    the raw sub-diagrams Link.sublink produces.
    """
    diagram = knot.copy()
    diagram.simplify('global')
    return diagram


def canonical_alexander(coeffs: Sequence[int]) -> Tuple[int, ...]:
    """
    Alexander coefficients up to the units of Z[t, 1/t] and the
    substitution t -> 1/t: zeros stripped from both ends, the
    leading coefficient made positive, and the smaller of the list
    and its reversal kept.
    """
    c = [int(x) for x in coeffs]
    while c and c[0] == 0:
        c.pop(0)
    while c and c[-1] == 0:
        c.pop()
    if not c:
        return (0,)
    forms = [tuple(seq) if seq[-1] > 0 else tuple(-x for x in seq)
             for seq in (c, c[::-1])]
    return min(forms)


def laurent_pairs(poly) -> Tuple[Tuple[int, int], ...]:
    """
    A Laurent polynomial in one variable as sorted (exponent,
    coefficient) pairs, zero coefficients dropped.

    Raises:
        ValueError: if an exponent is not an integer.
    """
    try:
        items = poly.coefficients()
    except AttributeError:
        items = [[poly, 0]]
    terms: Dict[int, int] = {}
    for coeff, exponent in items:
        q = QQ(exponent)
        if q.denominator() != 1:
            raise ValueError(f'non-integral exponent {exponent} in {poly}')
        terms[int(q)] = terms.get(int(q), 0) + int(coeff)
    return tuple(sorted((e, c) for e, c in terms.items() if c != 0))


def jones_key(diagram: Link) -> Tuple[Tuple[int, int], ...]:
    """
    The Jones polynomial of a knot diagram, as (exponent,
    coefficient) pairs in Sage's variable t.  V_{m(K)}(t) =
    V_K(1/t), so mirroring negates the exponents.

    Raises:
        ValueError: above JONES_MAX_CROSSINGS crossings.
    """
    if len(diagram.crossings) > JONES_MAX_CROSSINGS:
        raise ValueError(f'Jones refused: {len(diagram.crossings)} crossings '
                         f'exceed JONES_MAX_CROSSINGS = {JONES_MAX_CROSSINGS}')
    return laurent_pairs(_to_sage_link(diagram).jones_polynomial())


class _KeyError(Exception):
    """A key could not be computed on a diagram."""


# Computed keys, cached on the PD code of the key diagram and the
# rung: a census presents the same component diagrams many times.
_KEY_CACHE: Dict[Tuple[str, str], dict] = {}


class _KeyReader:
    """
    The keys of one key diagram, computed a rung at a time on
    demand.

    Args:
        diagram: the key diagram.
        seifert_matrix: a Seifert matrix of the same knot, if the
            caller already has one.
        hfk: spherogram's knot Floer homology of the same knot, if
            the caller already has it.
    """

    def __init__(self, diagram: Link, seifert_matrix=None,
                 hfk: Optional[dict] = None):
        self.diagram = diagram
        self._pd = str(diagram.PD_code())
        self._matrix = seifert_matrix
        self._hfk = hfk

    def value(self, key: str):
        """
        The value of one key of LADDER.

        Raises:
            _KeyError: if its rung cannot be computed.
        """
        rung = _RUNG[key]
        cache_key = (self._pd, rung)
        if cache_key not in _KEY_CACHE:
            try:
                _KEY_CACHE[cache_key] = self._compute(rung)
            except Exception as exc:
                raise _KeyError(f'{rung}: {type(exc).__name__}: {exc}') from exc
        return _KEY_CACHE[cache_key][key]

    def all(self) -> dict:
        """Every key of LADDER."""
        return {key: self.value(key) for key in LADDER}

    def _seifert_matrix(self):
        if self._matrix is None:
            self._matrix = _sage_seifert_matrix_from_sage_link(
                _to_sage_link(self.diagram))
        if self._matrix is None:
            raise ValueError('the diagram has no Seifert matrix')
        return self._matrix

    def _compute(self, rung: str) -> dict:
        if rung == 'hfk':
            hfk = (self._hfk if self._hfk is not None
                   else _compute_hfk(self.diagram))
            if hfk is None:
                raise ValueError('knot Floer homology is unavailable')
            return {'g': int(hfk['seifert_genus']), 'tau': int(hfk['tau']),
                    'hfk_rank': int(hfk['total_rank'])}
        if rung == 'seifert':
            V = self._seifert_matrix()
            sig, _ = _sage_signature_and_nullity_from_matrix(V)
            return {'sig': int(sig)}
        if rung == 'alexander':
            delta = _alexander_polynomial_from_seifert(self._seifert_matrix())
            if delta is None:
                raise ValueError('the Alexander polynomial vanishes')
            coeffs = [int(c) for c in delta.list()]
            det = abs(sum(c * (-1) ** i for i, c in enumerate(coeffs)))
            return {'det': det, 'alex': canonical_alexander(coeffs)}
        return {'jones': jones_key(self.diagram)}


def compute_keys(knot: Link) -> dict:
    """
    Every key of LADDER for a knot diagram, computed on
    key_diagram(knot) exactly as identify() computes them.

    Raises:
        _KeyError: if a key cannot be computed.
    """
    return _KeyReader(key_diagram(knot)).all()


def mirror_keys(keys: dict) -> dict:
    """
    The keys of the mirror image: sig and tau negated, the Jones
    exponents negated.
    """
    out = dict(keys)
    out['sig'] = -keys['sig']
    out['tau'] = -keys['tau']
    out['jones'] = tuple(sorted((-e, c) for e, c in keys['jones']))
    return out


def _polynomial_product(a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
    """Product of two dense coefficient lists, lowest term first."""
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return tuple(out)


def _laurent_product(a, b) -> Tuple[Tuple[int, int], ...]:
    """
    Product of two Laurent polynomials given as 
    (exponent, coefficient) pairs.
    """
    terms: Dict[int, int] = {}
    for ea, ca in a:
        for eb, cb in b:
            terms[ea + eb] = terms.get(ea + eb, 0) + ca * cb
    return tuple(sorted((e, c) for e, c in terms.items() if c != 0))


def connected_sum_keys(parts: Sequence[dict]) -> dict:
    """
    The keys of a connected sum from those of its summands: g, tau
    and sig add; det and the Alexander and Jones polynomials
    multiply; so does hfk_rank, knot Floer homology of a connected
    sum being the tensor product.
    """
    det = hfk_rank = 1
    alex: Tuple[int, ...] = (1,)
    jones: Tuple[Tuple[int, int], ...] = ((0, 1),)
    for part in parts:
        det *= part['det']
        hfk_rank *= part['hfk_rank']
        alex = _polynomial_product(alex, part['alex'])
        jones = _laurent_product(jones, part['jones'])
    return {'g': sum(part['g'] for part in parts),
            'tau': sum(part['tau'] for part in parts),
            'hfk_rank': hfk_rank,
            'sig': sum(part['sig'] for part in parts),
            'det': det, 'alex': canonical_alexander(alex),
            'jones': jones}


# =============================================================================
# BUILDING THE TABLE
# =============================================================================

# Number of prime knots up to mirror image, by crossing number.  A
# source must list exactly these many for the table to be complete,
# which the identification relies on.
_PRIME_KNOT_COUNTS = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165,
                      11: 552, 12: 2176, 13: 9988, 14: 46972, 15: 253293,
                      16: 1388705}


def _read_source(source_csv: str, max_crossings: int) -> List[dict]:
    """
    The prime knots of source_csv with at most max_crossings
    crossings, sorted by crossing number and index.

    Raises:
        ValueError: on a missing column, a repeated name, or a
            crossing number whose knots are not all listed.
    """
    frame = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    for column in ('Name', 'PD Notation'):
        if column not in frame.columns:
            raise ValueError(f'{source_csv} has no {column!r} column')
    u_column = next((c for c in ('Unknotting Number', 'Unlinking Number')
                     if c in frame.columns), None)
    rows, seen = [], set()
    for record in frame.to_dict('records'):
        name = record['Name'].strip()
        match = _PRIME_NAME.fullmatch(name)
        if match is None or not 3 <= int(match.group(1)) <= max_crossings:
            continue
        if name in seen:
            raise ValueError(f'{source_csv}: {name} appears twice')
        seen.add(name)
        rows.append({'name': name, 'c': int(match.group(1)),
                     'pd': json.loads(record['PD Notation'].replace(';', ',')),
                     'u': record.get(u_column, '') if u_column else '',
                     'g4': record.get('Genus-4D', '')})
    rows.sort(key=lambda row: _summand_key(row['name']))
    listed = Counter(row['c'] for row in rows)
    for c in range(3, max_crossings + 1):
        if c not in _PRIME_KNOT_COUNTS:
            raise ValueError(f'no census count of the prime knots with {c} '
                             'crossings to check the source against')
        if listed[c] != _PRIME_KNOT_COUNTS[c]:
            raise ValueError(f'{source_csv} lists {listed[c]} prime knots with '
                             f'{c} crossings, not {_PRIME_KNOT_COUNTS[c]}: '
                             'the table would not be complete')
    return rows


def _keys_of(name: str, knot: Link) -> dict:
    """compute_keys, naming the knot in the error."""
    try:
        return compute_keys(knot)
    except _KeyError as exc:
        raise ValueError(f'{name}: a key cannot be computed: {exc}') from exc


def _check_keys(name: str, derived: dict, computed: dict, how: str) -> None:
    """
    Raises:
        ValueError: if keys derived for a diagram built by how
            disagree with the keys computed on it.
    """
    wrong = [key for key in LADDER if derived[key] != computed[key]]
    if wrong:
        raise ValueError(
            f'{name}: the keys derived for {how} disagree with the ones '
            'computed on its diagram: '
            + ', '.join(f'{k} {derived[k]} != {computed[k]}' for k in wrong))


def _closed_form_s(knot: Link, sig: int) -> Optional[Tuple[int, str]]:
    r"""
    The Rasmussen invariant from a closed form, when one applies to
    the diagram: s = c(D) - O(D) + 1 on a positive diagram, and
    s = -\sigma on a connected alternating one.

    Args:
        knot: a knot diagram.
        sig: its signature, in Sage's convention.

    Returns:
        (s, s_source), or None when neither form applies.

    Reference: [cavallo2015links, Proposition 3.3] (positive),
               [rasmussen2010slice, Theorem 3] (alternating)
    """
    if _is_positive_diagram(knot):
        o = len(_to_sage_link(knot).seifert_circles())
        return (_s_invariant_positive(len(knot.crossings), o),
                'positive_formula')
    if _is_alternating_diagram(knot) and not _is_split(knot):
        return -sig, 'alternating_signature'
    return None


def _knotjob_s(diagrams: Dict[str, Link], per_link_timeout: int,
               workers: int) -> Tuple[Dict[str, Optional[int]], Optional[str]]:
    """
    s of every diagram, from one KnotJob batch.

    Returns:
        ({name: s or None}, None), or ({}, the error) when KnotJob
        cannot run.
    """
    names = list(diagrams)
    try:
        from bounds_pipeline.knotjob_wrapper import compute_s_invariants_batch
        values = compute_s_invariants_batch(
            pd_codes=[[list(row) for row in diagrams[n].PD_code()]
                      for n in names],
            names=names,
            timeout=max(600, per_link_timeout * len(names)),
            num_workers=workers,
            per_link_timeout=per_link_timeout)
    except Exception as exc:
        return {}, f'{type(exc).__name__}: {exc}'
    return ({n: None if values.get(n) is None else int(values[n])
             for n in names}, None)


def _sha256(path: str) -> str:
    """
    Computes the SHA-256 hash of a file, returned as a hex string.
    
    This acts as a unique digital fingerprint of the file's exact contents, 
    which is used to track external dependencies (like the KnotJob jar) 
    and guarantee computational reproducibility. The file is read in 1MB 
    blocks to keep memory usage low.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _knotjob_jar_sha256() -> Optional[str]:
    """
    SHA-256 of the KnotJob jar, which pins its version; None
    without the jar.
    """
    try:
        from bounds_pipeline.knotjob_wrapper import KNOTJOB_JAR
        return _sha256(str(KNOTJOB_JAR))
    except Exception:
        return None


def _versions() -> dict:
    """The versions of Sage and spherogram."""
    import sage.version
    import spherogram
    return {'sage': str(sage.version.version),
            'spherogram': str(spherogram.__version__)}


def _build_date() -> str:
    """
    Returns the build date as an ISO-formatted UTC string.
    
    To ensure reproducible builds across different environments, 
    it prioritizes the 'SOURCE_DATE_EPOCH' environment variable if present. 
    If that variable is not set, it defaults to the current UTC date.
    """
    epoch = os.environ.get('SOURCE_DATE_EPOCH')
    moment = (datetime.fromtimestamp(int(epoch), timezone.utc) if epoch
              else datetime.now(timezone.utc))
    return moment.date().isoformat()


def _relative_path(path: str) -> str:
    """
    path relative to the repository when it lies inside, else its
    file name.
    """
    real, root = os.path.realpath(path), os.path.realpath(_ROOT)
    if os.path.commonpath([real, root]) == root:
        return os.path.relpath(real, root)
    return os.path.basename(real)


def _entry(name: str, summands: Sequence[str], prime: bool, c: int,
           diagram: Link, keys: dict, s: Optional[int],
           s_source: Optional[str]) -> TableEntry:
    """A TableEntry for a diagram and its keys."""
    return TableEntry(name=name, summands=tuple(summands), prime=prime,
                      crossing_number=c,
                      pd_code=[list(row) for row in diagram.PD_code()],
                      crossing_signs=[int(x.sign) for x in diagram.crossings],
                      s=s, s_source=s_source, **keys)


def _unknot_entry() -> TableEntry:
    """The unknot, whose keys need no computation."""
    return TableEntry(name=UNKNOT, summands=(), prime=False, crossing_number=0,
                      pd_code=[], crossing_signs=[], g=0, tau=0, hfk_rank=1,
                      det=1, alex=(1,), sig=0, jones=((0, 1),), s=0,
                      s_source=UNKNOT_SOURCE)


def build_knot_table(max_crossings: int = 10,
                     source_csv: str = DEFAULT_SOURCE_CSV,
                     out_path: Optional[str] = DEFAULT_TABLE_JSON,
                     use_knotjob: bool = True,
                     verbose: bool = True,
                     knotjob_timeout: int = 60,
                     knotjob_workers: int = 1) -> KnotTable:
    r"""
    Builds the knot table: the unknot; every prime knot of
    source_csv with at most max_crossings crossings, as K in the
    chirality of its PD code and as m(K) mirrored; and every
    connected sum of two or more of those whose crossing numbers
    add up to at most max_crossings.

    The keys of K are computed on its diagram through the code
    identify() uses.  Those of m(K) and of a connected sum are
    derived, then computed again on Link.mirror() and on
    Link.connected_sum() of the summands' diagrams, and the two
    must agree.  s(K) comes from one KnotJob batch, checked against
    c(D) - O(D) + 1 on a positive diagram and -\sigma on an
    alternating one, either on K or on m(K); without KnotJob those
    closed forms are used where they apply and s is None elsewhere.
    s(m(K)) = -s(K), and s of a connected sum is the sum.

    Args:
        max_crossings: the largest crossing number tabulated.
        source_csv: CSV with a Name column in KnotInfo's style
            (10_165) and a PD Notation column as in
            datasets/rolfsen.csv; it must list every prime knot up
            to max_crossings.
        out_path: where to write the JSON table; None skips it.
        use_knotjob: compute s with KnotJob, which needs Java.
        verbose: print the build report.
        knotjob_timeout: per-knot KnotJob timeout in seconds.
        knotjob_workers: parallel KnotJob processes.

    Returns:
        The KnotTable.

    Raises:
        ValueError: on an incomplete or malformed source, on a
            derived key the recomputation contradicts, on an s a
            closed form contradicts, and when the 3_1 of the source
            is not the positive trefoil with sig(3_1) = -2 and
            s(3_1) = 2, the conventions the pipeline uses.
    """
    start = time.time()
    rows = _read_source(source_csv, max_crossings)

    diagrams: Dict[str, Link] = {}
    keys_of: Dict[str, dict] = {}
    crossings_of: Dict[str, int] = {}
    for row in rows:
        name, c = row['name'], row['c']
        knot = Link(row['pd'])
        if len(knot.link_components) != 1 or knot.unlinked_unknot_components:
            raise ValueError(f'{name}: the PD code is not a knot diagram')
        if len(knot.crossings) != c:
            raise ValueError(f'{name}: the PD code has {len(knot.crossings)} '
                             f'crossings, not {c}')
        mirror, mname = knot.mirror(), mirror_name(name)
        keys = _keys_of(name, knot)
        derived = mirror_keys(keys)
        _check_keys(mname, derived, _keys_of(mname, mirror), 'Link.mirror()')
        for n, diagram, k in ((name, knot, keys), (mname, mirror, derived)):
            diagrams[n], keys_of[n], crossings_of[n] = diagram, k, c

    # s of the prime knots: KnotJob on K, the closed forms on K and on
    # m(K) as a check or as the fallback.
    names = [row['name'] for row in rows]
    knotjob_values: Dict[str, Optional[int]] = {}
    knotjob_error: Optional[str] = None
    if use_knotjob:
        knotjob_values, knotjob_error = _knotjob_s(
            {n: diagrams[n] for n in names}, knotjob_timeout, knotjob_workers)
        if knotjob_error and verbose:
            print(f'KnotJob unavailable, s from the closed forms only: '
                  f'{knotjob_error}')
    s_value: Dict[str, Optional[int]] = {}
    s_source: Dict[str, Optional[str]] = {}
    for name in names:
        mname = mirror_name(name)
        forms = []
        closed = _closed_form_s(diagrams[name], keys_of[name]['sig'])
        if closed is not None:
            forms.append(closed)
        closed = _closed_form_s(diagrams[mname], keys_of[mname]['sig'])
        if closed is not None:
            forms.append((-closed[0], closed[1]))
        value, source = knotjob_values.get(name), 'knotjob'
        if value is None and forms:
            value, source = forms[0]
        for form_value, form_source in forms:
            if value != form_value:
                raise ValueError(f'{name}: s = {value} ({source}) but the '
                                 f'{form_source} closed form gives {form_value}')
        if value is None:
            source = None
        s_value[name], s_source[name] = value, source
        s_value[mname] = None if value is None else -value
        s_source[mname] = source

    entries = [_unknot_entry()]
    for name in names:
        for n in (name, mirror_name(name)):
            entries.append(_entry(n, (n,), True, crossings_of[n], diagrams[n],
                                  keys_of[n], s_value[n], s_source[n]))

    # Connected sums: multisets of prime entries, the smallest prime
    # knot having three crossings.
    pool = sorted((n for n in keys_of if crossings_of[n] <= max_crossings - 3),
                  key=name_order)
    combos: List[List[str]] = []

    def extend(first: int, chosen: List[str], total: int) -> None:
        if len(chosen) >= 2:
            combos.append(chosen)
        for i in range(first, len(pool)):
            c = crossings_of[pool[i]]
            if total + c <= max_crossings:
                extend(i, chosen + [pool[i]], total + c)

    extend(0, [], 0)
    composites = 0
    for combo in combos:
        name = composite_name(combo)
        derived = connected_sum_keys([keys_of[p] for p in combo])
        diagram = diagrams[combo[0]].copy()
        for part in combo[1:]:
            diagram = diagram.connected_sum(diagrams[part].copy())
        _check_keys(name, derived, _keys_of(name, diagram),
                    'Link.connected_sum()')
        parts_s = [s_value[p] for p in combo]
        s = sum(parts_s) if all(v is not None for v in parts_s) else None
        closed = _closed_form_s(diagram, derived['sig'])
        if s is not None and closed is not None and closed[0] != s:
            raise ValueError(f'{name}: s = {s} by additivity but the '
                             f'{closed[1]} closed form gives {closed[0]}')
        entries.append(_entry(name, combo, False,
                              sum(crossings_of[p] for p in combo), diagram,
                              derived, s, None if s is None else 'additivity'))
        composites += 1

    table = KnotTable(entries, max_crossings=max_crossings)
    trefoil = table._by_name.get('3_1')
    if trefoil is not None and (not _is_positive_diagram(diagrams['3_1'])
                                or trefoil.sig != -2
                                or trefoil.s not in (None, 2)):
        raise ValueError(
            'chirality convention: 3_1 must be the positive trefoil with '
            f'sig = -2 and s = 2; here its diagram signs are '
            f'{trefoil.crossing_signs}, sig = {trefoil.sig}, s = {trefoil.s}')

    used = any(v is not None for v in knotjob_values.values())
    source_values = {row['name']: {'u': row['u'], 'g4': row['g4']}
                     for row in rows}
    table.report = {
        'max_crossings': max_crossings,
        'source_csv': _relative_path(source_csv),
        'source_csv_sha256': _sha256(source_csv),
        'knotjob': {
            'requested': bool(use_knotjob), 'used': used,
            'error': knotjob_error,
            'jar_sha256': _knotjob_jar_sha256() if used else None,
            'no_result': sorted((n for n, v in knotjob_values.items()
                                 if v is None), key=name_order)},
        'versions': _versions(),
        'build_date': _build_date(),
        'counts': {'prime_knots': len(names),
                   'prime_entries': 2 * len(names),
                   'composite_entries': composites,
                   'entries': len(table.entries)},
        'sweep': uniqueness_sweep(table.entries),
        'residual_classes': _residual_classes(table.entries, source_values),
        'chirality_undetected': _chirality_undetected(table.entries),
        's_sources': dict(sorted(Counter(e.s_source or 'none'
                                         for e in table.entries).items())),
    }
    if out_path:
        table.to_json(out_path)
    if verbose:
        print(format_build_report(table, elapsed=time.time() - start))
    return table


def _key_vector(entry: TableEntry) -> tuple:
    """The values of every LADDER key, in order."""
    return tuple(getattr(entry, key) for key in LADDER)


def uniqueness_sweep(entries: Sequence[TableEntry]) -> List[dict]:
    """
    How well each prefix of LADDER separates the knots of a table,
    counted up to chirality.

    Returns:
        One row per prefix: the keys, the largest number of
        different knots sharing one value of them (max_group), and
        the number of values shared by several (ambiguous_groups).
    """
    out = []
    for k in range(1, len(LADDER) + 1):
        groups: Dict[tuple, set] = {}
        for e in entries:
            value = tuple(getattr(e, key) for key in LADDER[:k])
            groups.setdefault(value, set()).add(mirror_class(e.name))
        sizes = [len(group) for group in groups.values()]
        out.append({'keys': list(LADDER[:k]),
                    'max_group': max(sizes, default=0),
                    'ambiguous_groups': sum(1 for size in sizes if size > 1)})
    return out


def _residual_classes(entries: Sequence[TableEntry],
                      source_values: Dict[str, dict]) -> List[dict]:
    """
    The groups of entries sharing every key while being different
    knots up to chirality: the queries identify() answers with
    AMBIGUOUS.  kind is 'summand_chirality' when the entries differ
    only in the mirror flags of their summands, 'distinct_knots'
    otherwise; known holds the source's u and g4 of every prime
    involved.
    """
    groups: Dict[tuple, List[TableEntry]] = {}
    for e in entries:
        groups.setdefault(_key_vector(e), []).append(e)
    out = []
    for members in groups.values():
        if len({mirror_class(e.name) for e in members}) < 2:
            continue
        bases = sorted({base_name(s) for e in members for s in e.summands},
                       key=_summand_key)
        shapes = {tuple(sorted(base_name(s) for s in e.summands))
                  for e in members}
        out.append({'names': [e.name for e in members],
                    'crossing_numbers': [e.crossing_number for e in members],
                    'kind': ('summand_chirality' if len(shapes) == 1
                             else 'distinct_knots'),
                    'known': {b: source_values.get(b, {}) for b in bases}})
    out.sort(key=lambda group: [name_order(n) for n in group['names']])
    return out


def _chirality_undetected(entries: Sequence[TableEntry]) -> dict:
    """
    The knots whose two chiralities share every key: amphichiral
    knots, and chiral knots the keys cannot orient, whose s is
    listed since s changes sign with the chirality.
    """
    by_name = {e.name: e for e in entries}
    primes, composites = [], []
    for e in entries:
        other = mirror_name(e.name)
        if other == e.name or name_order(other) < name_order(e.name):
            continue
        partner = by_name.get(other)
        if partner is None or _key_vector(partner) != _key_vector(e):
            continue
        if e.prime:
            primes.append({'name': e.name, 's': e.s})
        else:
            composites.append(e.name)
    return {'primes': primes, 'composites': composites}


def format_build_report(table: KnotTable, elapsed: Optional[float] = None) -> str:
    """
    The build report of a table as text: counts, inputs and
    versions, the uniqueness sweep, the residual classes, the knots
    whose chirality no key detects, and where s comes from.

    Args:
        table: a KnotTable carrying its report.
        elapsed: the build time in seconds, printed when given.
    """
    r = table.report
    counts = r.get('counts', {})
    knotjob = r.get('knotjob', {})
    versions = r.get('versions', {})
    if knotjob.get('used'):
        kj = f"used, jar sha256 {str(knotjob.get('jar_sha256'))[:16]}"
        if knotjob.get('no_result'):
            kj += f", no result for {knotjob['no_result']}"
    else:
        kj = 'not used' + (f" ({knotjob['error']})" if knotjob.get('error') else '')
    lines = [
        f"knot table up to {table.max_crossings} crossings: "
        f"{counts.get('prime_knots')} prime knots, "
        f"{counts.get('prime_entries')} prime entries (both chiralities), "
        f"{counts.get('composite_entries')} connected sums, "
        f"{counts.get('entries')} entries with the unknot",
        f"  source      {r.get('source_csv')}, sha256 "
        f"{str(r.get('source_csv_sha256'))[:16]}",
        f"  versions    Sage {versions.get('sage')}, "
        f"spherogram {versions.get('spherogram')}",
        f"  KnotJob     {kj}",
        f"  build date  {r.get('build_date')}",
        f"  s sources   {r.get('s_sources')}",
        '',
        '  uniqueness sweep, knots counted up to chirality:',
        f"    {'keys':56s}{'max group':>10s}{'ambiguous':>11s}"]
    for row in r.get('sweep', []):
        keys = '(' + ', '.join(row['keys']) + ')'
        lines.append(f"    {keys:56s}{row['max_group']:>10d}"
                     f"{row['ambiguous_groups']:>11d}")
    classes = r.get('residual_classes', [])
    lines += ['', f'  residual classes at the full ladder: {len(classes)}']
    for group in classes:
        members = ' | '.join(f'{n} (c = {c})' for n, c in
                             zip(group['names'], group['crossing_numbers']))
        known = ', '.join(f"{b}: u = {v.get('u') or '?'}, g4 = {v.get('g4') or '?'}"
                          for b, v in group['known'].items() if v)
        lines.append(f"    {group['kind']:18s} {members}"
                     + (f"   [{known}]" if known else ''))
    undetected = r.get('chirality_undetected', {})
    primes = undetected.get('primes', [])
    lines += ['', f'  prime knots whose chirality no key detects: {len(primes)}']
    for start in range(0, len(primes), 6):
        lines.append('    ' + ', '.join(f"{p['name']} (s = {p['s']})"
                                        for p in primes[start:start + 6]))
    composites = undetected.get('composites', [])
    if composites:
        lines.append(f'  connected sums whose chirality no key detects: '
                     f'{len(composites)}')
    if elapsed is not None:
        lines.append(f'  build time  {elapsed:.1f} s')
    return '\n'.join(lines)


# =============================================================================
# IDENTIFICATION
# =============================================================================

@dataclass
class Identification:
    """
    The outcome of identify().  names are the surviving entries in
    table order, chirality_known is True when one entry survives,
    crossings is n', the crossing count of the key diagram, and
    domain_max the table's max_crossings.  A diagram identified
    through its visible connected summands has them in summands and
    no names.
    """
    status: str
    names: List[str] = field(default_factory=list)
    chirality_known: bool = False
    crossings: Optional[int] = None
    domain_max: Optional[int] = None
    keys_used: List[str] = field(default_factory=list)
    summands: List['Identification'] = field(default_factory=list)

    @property
    def label(self) -> Optional[str]:
        """
        The knot as text: its name, the survivors joined by '|', or
        the summands joined by '#' with a class of survivors in
        braces; None when nothing was identified.
        """
        if self.summands:
            return '#'.join(_braced(s) for s in self.summands)
        return '|'.join(self.names) if self.names else None


def _braced(identification: Identification) -> str:
    """The label, in braces when it names several survivors."""
    label = identification.label or '?'
    return f'{{{label}}}' if len(identification.names) > 1 else label


def _check_genus_hint(genus_hint: Optional[int], genus: int) -> None:
    """
    Raises:
        ValueError: if genus_hint is given and is not genus.
    """
    if genus_hint is not None and genus_hint != genus:
        raise ValueError(f'genus hint {genus_hint}, but the knot has '
                         f'genus {genus}')


def _connected_summands(diagram: Link) -> List[Link]:
    """
    The visible connected summands of a diagram, dropping any
    without crossings; [diagram] when it does not decompose.
    """
    try:
        parts = diagram.copy().deconnect_sum()
    except Exception:
        return [diagram]
    parts = [p for p in parts if len(p.crossings) > 0]
    return parts if len(parts) > 1 else [diagram]


def _genus(identification: Identification, table: KnotTable) -> int:
    """The Seifert genus of an identified knot, shared by its survivors."""
    if identification.summands:
        return sum(_genus(s, table) for s in identification.summands)
    return table.by_name(identification.names[0]).g


def identify(knot: Link, table: KnotTable, seifert_matrix=None,
             hfk: Optional[dict] = None,
             genus_hint: Optional[int] = None) -> Identification:
    """
    Identifies a knot diagram in the table.

    The diagram is simplified to a key diagram with n' crossings:
    n' = 0 is the unknot, and n' > table.max_crossings is
    OUT_OF_DOMAIN.  Otherwise the candidates are the entries with at
    most n' crossings, narrowed by every key of LADDER in turn, a
    knot Floer genus of 0 giving the unknot.  One knot up to
    chirality surviving is IDENTIFIED, with the chirality known
    when one entry survives; several knots surviving is AMBIGUOUS,
    the knot being one of them; none, or a key that cannot be
    computed, is MISSING.  A MISSING or OUT_OF_DOMAIN diagram that
    visibly decomposes as a connected sum is identified through its
    summands, each against its own domain.

    Args:
        knot: a knot diagram, not necessarily simplified.
        table: the KnotTable.
        seifert_matrix: a Seifert matrix of the same knot, if the
            caller has one.
        hfk: spherogram's knot Floer homology of the same knot, if
            the caller has it.
        genus_hint: the Seifert genus, when the caller knows it by
            other means; it is checked against the genus found.

    Returns:
        The Identification.

    Raises:
        ValueError: if genus_hint disagrees with the genus found.
    """
    diagram = key_diagram(knot)
    n = len(diagram.crossings)
    limit = table.max_crossings
    if n == 0:
        _check_genus_hint(genus_hint, 0)
        return Identification(Status.IDENTIFIED, [UNKNOT], True, 0, limit)
    if n > limit:
        return _through_summands(
            diagram, table, genus_hint,
            Identification(Status.OUT_OF_DOMAIN, crossings=n, domain_max=limit))
    reader = _KeyReader(diagram, seifert_matrix, hfk)
    candidates = [e for e in table.entries if e.crossing_number <= n]
    used: List[str] = []
    try:
        for key in LADDER:
            value = reader.value(key)
            used.append(key)
            if key == 'g':
                _check_genus_hint(genus_hint, value)
                if value == 0:
                    return Identification(Status.IDENTIFIED, [UNKNOT], True,
                                          n, limit, used)
            candidates = [e for e in candidates if getattr(e, key) == value]
            if not candidates:
                break
    except _KeyError:
        candidates = []
    if not candidates:
        return _through_summands(
            diagram, table, genus_hint,
            Identification(Status.MISSING, crossings=n, domain_max=limit,
                           keys_used=used))
    names = [e.name for e in candidates]
    if len({mirror_class(name) for name in names}) == 1:
        return Identification(Status.IDENTIFIED, names, len(names) == 1, n,
                              limit, used)
    return Identification(Status.AMBIGUOUS, names, False, n, limit, used)


def _through_summands(diagram: Link, table: KnotTable,
                      genus_hint: Optional[int],
                      fallback: Identification) -> Identification:
    """
    identify() on each visible connected summand of diagram; the
    fallback when it does not decompose or a summand stays
    unidentified.
    """
    parts = _connected_summands(diagram)
    if len(parts) < 2:
        return fallback
    summands = [identify(part, table) for part in parts]
    if any(s.status not in (Status.IDENTIFIED, Status.AMBIGUOUS)
           for s in summands):
        return fallback
    _check_genus_hint(genus_hint, sum(_genus(s, table) for s in summands))
    status = (Status.IDENTIFIED
              if all(s.status == Status.IDENTIFIED for s in summands)
              else Status.AMBIGUOUS)
    return Identification(status, [], all(s.chirality_known for s in summands),
                          fallback.crossings, fallback.domain_max,
                          fallback.keys_used, summands)


# =============================================================================
# KNOWN VALUES
# =============================================================================

@dataclass
class KnownRecord:
    """
    The best known values of one prime knot: the admissible values
    of g_4 and u as dataset.parse_invariant returns them, None when
    not given, and the (lower, upper) bound source of each.
    """
    g4: Optional[List[int]]
    u: Optional[List[int]]
    g4_sources: Tuple[str, str]
    u_sources: Tuple[str, str]


@dataclass
class KnownValueTable:
    """
    The best known values read from one CSV: the records by
    KnotInfo name, the file, its identity for cache keys, and the
    column u was read from.
    """
    records: Dict[str, KnownRecord]
    path: str
    identity: tuple
    u_column: str


@dataclass
class KnownValues:
    """
    What the table says about an identified knot: bounds on g_4 and
    u, s, whether the knot is knotted, and a source for each value,
    None where there is no value.  A source reads
    '<kind>:<sources>@<knot>', e.g. known:KnotInfo@m(8_20),
    known:KnotInfo@{8_8|m(10_129)}, subadditive:KnotInfo@3_1#m(3_1),
    table:knotjob@3_1; the unknot's values read 'unknot'.
    """
    g4_lower: Optional[int] = None
    g4_upper: Optional[int] = None
    u_lower: Optional[int] = None
    u_upper: Optional[int] = None
    s: Optional[int] = None
    knotted: bool = False
    g4_lower_source: Optional[str] = None
    g4_upper_source: Optional[str] = None
    u_lower_source: Optional[str] = None
    u_upper_source: Optional[str] = None
    s_source: Optional[str] = None


@dataclass(frozen=True)
class _Value:
    """A value, with the kinds and the CSV sources it comes from."""
    value: Optional[int]
    kinds: frozenset = frozenset()
    sources: frozenset = frozenset()


_NO_VALUE = _Value(None)

_FIELDS = ('g4_lower', 'g4_upper', 'u_lower', 'u_upper', 's')


def _known_end(record: Optional[KnownRecord], invariant: str,
               side: int) -> _Value:
    """The lower (side 0) or upper (side 1) end of a known range."""
    values = getattr(record, invariant) if record is not None else None
    if not values:
        return _NO_VALUE
    value = values[0] if side == 0 else values[-1]
    if isinstance(value, float):
        # parse_invariant gives a float only for an infinite value.
        return _NO_VALUE
    source = getattr(record, f'{invariant}_sources')[side] or '?'
    return _Value(int(value), frozenset({KNOWN_SOURCE}), frozenset({source}))


def _sum_of(parts: Sequence[_Value], kind: str) -> _Value:
    """The sum of several values, when all of them are known."""
    if not parts or any(p.value is None for p in parts):
        return _NO_VALUE
    return _Value(sum(p.value for p in parts), frozenset({kind}),
                  frozenset().union(*(p.sources for p in parts)))


def _entry_values(entry: TableEntry,
                  known: Optional[KnownValueTable]) -> Dict[str, _Value]:
    """
    The values of one entry: the known range of a prime knot,
    sub-additive upper bounds only for a connected sum, and s.
    """
    if entry.name == UNKNOT:
        zero = _Value(0, frozenset({UNKNOT_SOURCE}))
        return dict.fromkeys(_FIELDS, zero)
    s = (_Value(entry.s, frozenset({TABLE_SOURCE}), frozenset({entry.s_source}))
         if entry.s is not None else _NO_VALUE)
    records = [known.records.get(base_name(p)) if known is not None else None
               for p in entry.summands]
    if entry.prime:
        return {'g4_lower': _known_end(records[0], 'g4', 0),
                'g4_upper': _known_end(records[0], 'g4', 1),
                'u_lower': _known_end(records[0], 'u', 0),
                'u_upper': _known_end(records[0], 'u', 1), 's': s}
    return {'g4_lower': _NO_VALUE,
            'g4_upper': _sum_of([_known_end(r, 'g4', 1) for r in records],
                                SUBADDITIVE_SOURCE),
            'u_lower': _NO_VALUE,
            'u_upper': _sum_of([_known_end(r, 'u', 1) for r in records],
                               SUBADDITIVE_SOURCE),
            's': s}


def _common(values: Sequence[_Value], pick) -> _Value:
    """
    The value every candidate satisfies: pick (min for a lower
    bound, max for an upper one) over the candidates, None when one
    of them has no value.
    """
    if not values or any(v.value is None for v in values):
        return _NO_VALUE
    return _Value(pick(v.value for v in values),
                  frozenset().union(*(v.kinds for v in values)),
                  frozenset().union(*(v.sources for v in values)))


def _common_s(values: Sequence[_Value]) -> _Value:
    """s when every candidate has the same one, else no value."""
    distinct = {v.value for v in values}
    if len(distinct) != 1 or None in distinct:
        return _NO_VALUE
    return _common(values, min)


def _raw_values(identification: Identification, table: KnotTable,
                known: Optional[KnownValueTable]) -> Optional[Dict[str, _Value]]:
    """
    The values of an identification by the common-value rule: a
    lower bound is the least of the candidates' lower bounds, an
    upper bound the greatest of their upper bounds, and s is kept
    only when every candidate has the same s.  A visible connected
    sum gets the sub-additive sum of its summands' upper bounds and
    the sum of their s.
    """
    if identification.status not in (Status.IDENTIFIED, Status.AMBIGUOUS):
        return None
    if identification.summands:
        parts = [_raw_values(s, table, known) for s in identification.summands]
        return {'g4_lower': _NO_VALUE,
                'g4_upper': _sum_of([p['g4_upper'] for p in parts],
                                    SUBADDITIVE_SOURCE),
                'u_lower': _NO_VALUE,
                'u_upper': _sum_of([p['u_upper'] for p in parts],
                                   SUBADDITIVE_SOURCE),
                's': _sum_of([p['s'] for p in parts], TABLE_SOURCE)}
    per = [_entry_values(table.by_name(n), known) for n in identification.names]
    return {'g4_lower': _common([p['g4_lower'] for p in per], min),
            'g4_upper': _common([p['g4_upper'] for p in per], max),
            'u_lower': _common([p['u_lower'] for p in per], min),
            'u_upper': _common([p['u_upper'] for p in per], max),
            's': _common_s([p['s'] for p in per])}


def _knotted(identification: Identification) -> bool:
    """Whether an identified knot is non-trivial."""
    if identification.summands:
        return any(_knotted(s) for s in identification.summands)
    return bool(identification.names) and identification.names != [UNKNOT]


def _source_label(value: _Value, label: str) -> Optional[str]:
    """The provenance string of a value, as documented on KnownValues."""
    if value.value is None:
        return None
    if value.kinds == {UNKNOT_SOURCE}:
        return UNKNOT_SOURCE
    kinds = '+'.join(sorted(value.kinds))
    sources = '+'.join(sorted(value.sources)) or '?'
    return f'{kinds}:{sources}@{label}'


def known_values(identification: Identification, table: KnotTable,
                 known: Optional[KnownValueTable] = None) -> KnownValues:
    r"""
    The values an identification yields.  An identified prime knot,
    K or m(K), takes g_4 and u from the known values of K, both
    being mirror-invariant, and s from its table entry.  A connected
    sum gets upper bounds only, by sub-additivity:
        g_4(K_1 # K_2) <= g_4(K_1) + g_4(K_2),
        u(K_1 # K_2) <= u(K_1) + u(K_2),
    and s = s(K_1) + s(K_2), s being additive.  Several survivors
    are combined by the common-value rule of _raw_values.  Any
    identified knot other than the unknot is knotted.

    Args:
        identification: the result of identify().
        table: the KnotTable it was made against.
        known: the known values; None gives s and knottedness only.

    Returns:
        The KnownValues; empty when nothing was identified.
    """
    raw = _raw_values(identification, table, known)
    if raw is None:
        return KnownValues()
    label = _braced(identification)
    out = KnownValues(knotted=_knotted(identification))
    for name in _FIELDS:
        setattr(out, name, raw[name].value)
        setattr(out, f'{name}_source', _source_label(raw[name], label))
    return out


# =============================================================================
# LOADING
# =============================================================================

_TABLE_CACHE: Dict[tuple, KnotTable] = {}
_KNOWN_CACHE: Dict[tuple, KnownValueTable] = {}
_SAID: set = set()


def _say_once(message: str) -> None:
    """Prints message the first time it comes up in this process."""
    if message not in _SAID:
        _SAID.add(message)
        print(message)


def table_path() -> str:
    """The table file: $KNOT_TABLE_JSON, else DEFAULT_TABLE_JSON."""
    return os.environ.get('KNOT_TABLE_JSON', DEFAULT_TABLE_JSON)


def known_values_path() -> str:
    """
    The known-values file: $KNOWN_KNOT_VALUES_CSV, else
    DEFAULT_KNOWN_VALUES_CSV.
    """
    return os.environ.get('KNOWN_KNOT_VALUES_CSV', DEFAULT_KNOWN_VALUES_CSV)


def default_table(path: Optional[str] = None) -> Optional[KnotTable]:
    """
    The knot table, read once per file and modification time.

    Args:
        path: the table file; defaults to table_path().

    Returns:
        The KnotTable, or None when the file is missing or cannot
        be read, which is printed once.
    """
    path = path or table_path()
    identity = _file_identity(path)
    if identity is None:
        _say_once(f'knot identification disabled: {path} not found')
        return None
    if identity not in _TABLE_CACHE:
        try:
            _TABLE_CACHE[identity] = KnotTable.from_json(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _say_once(f'knot identification disabled: {path}: '
                      f'{type(exc).__name__}: {exc}')
            return None
    return _TABLE_CACHE[identity]


def load_known_values(path: Optional[str] = None,
                      table: Optional[KnotTable] = None
                      ) -> Optional[KnownValueTable]:
    """
    Best known values of knots, read once per file and
    modification time.  The CSV has a Name column, a Genus-4D
    column, an Unknotting Number column (Unlinking Number is read
    when it is absent), and for each of the two a
    <column>_lower_bound_source and a <column>_upper_bound_source
    column; values parse with dataset.parse_invariant.

    Args:
        path: the CSV; defaults to known_values_path().
        table: when given, only its prime knots are kept.

    Returns:
        The KnownValueTable, or None when the file is missing or
        cannot be read, which is printed once.
    """
    path = path or known_values_path()
    identity = _file_identity(path)
    if identity is None:
        _say_once(f'known knot values disabled: {path} not found')
        return None
    key = (identity, None if table is None else table.cache_key)
    if key not in _KNOWN_CACHE:
        try:
            _KNOWN_CACHE[key] = _read_known_values(path, identity, table)
        except (OSError, ValueError, KeyError) as exc:
            _say_once(f'known knot values disabled: {path}: '
                      f'{type(exc).__name__}: {exc}')
            return None
    return _KNOWN_CACHE[key]


def _read_known_values(path: str, identity: tuple,
                       table: Optional[KnotTable]) -> KnownValueTable:
    """
    Reads the CSV of load_known_values.

    Raises:
        ValueError: when a required column is missing.
    """
    from dataset import parse_invariant

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    u_column = ('Unknotting Number' if 'Unknotting Number' in frame.columns
                else 'Unlinking Number')
    for column in ('Name', 'Genus-4D', u_column):
        if column not in frame.columns:
            raise ValueError(f'no {column!r} column')

    def parsed(text):
        try:
            return parse_invariant(str(text))
        except ValueError:
            return None

    def sources(record, column):
        return (str(record.get(f'{column}_lower_bound_source', '')),
                str(record.get(f'{column}_upper_bound_source', '')))

    records: Dict[str, KnownRecord] = {}
    for record in frame.to_dict('records'):
        name = str(record['Name']).strip()
        if _PRIME_NAME.fullmatch(name) is None:
            continue
        if table is not None and name not in table.prime_names:
            continue
        records[name] = KnownRecord(g4=parsed(record['Genus-4D']),
                                    u=parsed(record[u_column]),
                                    g4_sources=sources(record, 'Genus-4D'),
                                    u_sources=sources(record, u_column))
    return KnownValueTable(records, path, identity, u_column)
