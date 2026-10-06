"""
Slice genus and strong slice genus bounds computation.

This module computes all applicable bounds on the slice genus (g_4) and the
strong slice genus (g_4^*) of a link, using Sage for exact arithmetic and
spherogram for the diagram properties.

Usage:
    from bounds_pipeline.slice_bounds_py import compute_bounds_for_dataset
    from dataset import Dataset

    # For a dataset
    dataset = Dataset.read_csv('links.csv')
    results_df, summary = compute_bounds_for_dataset(dataset)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, replace
import math
from math import ceil
from time import time
import concurrent.futures
import multiprocessing
import re
from collections import defaultdict

import pandas as pd
from spherogram import Link
from sage.all import (QQbar, ZZ, PolynomialRing, matrix as sage_matrix,
                      block_matrix, Link as SageLink)

# Import KnotJob wrapper for s-invariant computation (batch mode)
try:
    from bounds_pipeline.knotjob_wrapper import compute_s_invariant, compute_s_invariants_batch
    KNOTJOB_AVAILABLE = True
except ImportError:
    KNOTJOB_AVAILABLE = False
    compute_s_invariant = None
    compute_s_invariants_batch = None


# =============================================================================
# LINK DIAGRAM PROPERTIES
# =============================================================================

def _is_positive_diagram(link: Link) -> bool:
    """Returns True if all crossings are positive."""
    return len(link.crossings) > 0 and all(c.sign == 1 for c in link.crossings)


def _is_negative_diagram(link: Link) -> bool:
    """Returns True if all crossings are negative."""
    return len(link.crossings) > 0 and all(c.sign == -1 for c in link.crossings)


def _is_split(link: Link) -> bool:
    """
    Returns True if the diagram falls apart: more than one
    connected piece, or a crossingless unknot beside it.  A
    property of the diagram, not of the link:
    braid_closure=[1,1,1,-2,-2,-2,3,-3] is a connected diagram
    of the square knot beside an unknot.
    """
    return len(link.split_link_diagram()) + link.unlinked_unknot_components > 1


def _num_split_components(link: Link) -> int:
    """Returns the number of split components."""
    return len(link.split_link_diagram()) + link.unlinked_unknot_components


def _is_alternating_diagram(link: Link) -> bool:
    """
    True iff the diagram itself is alternating: every edge of the PD code
    appears exactly once in an under position (0 or 2) and once in an over
    position (1 or 3).  Sage's Link.is_alternating() tests a braid word
    instead; its answer is BoundsResult.is_alternating.
    """
    positions: Dict[int, List[bool]] = {}
    for row in link.PD_code():
        for i, e in enumerate(row):
            positions.setdefault(e, []).append(i in (0, 2))
    return (len(positions) > 0
            and all(len(v) == 2 and v[0] != v[1] for v in positions.values()))


def _is_algebraically_split(link: Link) -> bool:
    """Returns True if all pairwise linking numbers are zero."""
    ell = len(link.link_components) + link.unlinked_unknot_components
    if ell <= 1:
        return True
    lk = link.linking_matrix()
    n = len(lk)
    for i in range(n):
        for j in range(i+1, n):
            if lk[i][j] != 0:
                return False
    return True


# =============================================================================
# SAGE-BASED INVARIANT COMPUTATION
# =============================================================================

def _abs_linking_number(link: Link) -> int:
    r"""
    The absolute linking number \abslk{L} = \sum_{i<j} |lk(K_i, K_j)|.

    Orientation-invariant: reversing a component flips the sign of each
    lk(K_i, K_j) it takes part in and leaves the absolute value alone.  Zero on
    a knot and on an algebraically split link.

    Reference: [collari2021slicetoruspositive, Theorem 1.9]
    """
    if len(link.link_components) + link.unlinked_unknot_components <= 1:
        return 0
    lk = link.linking_matrix()
    n = len(lk)
    return sum(abs(int(lk[i][j])) for i in range(n) for j in range(i + 1, n))


def _is_simply_linked(link: Link) -> bool:
    r"""
    True iff the crossings between any two fixed components all carry the same
    sign, i.e. the diagram certifies L as a simply-linked link.

    Orientation-independent: reversing one component flips the sign of every
    crossing it shares with another component simultaneously, so a pair that
    was sign-homogeneous stays sign-homogeneous.  A positive diagram is simply
    linked.

    Self-crossings are ignored; the condition constrains only the crossings
    between distinct components.

    Reference: [collari2021slicetoruspositive, Proposition 4.1 + Theorem 1.9]
    """
    ell = len(link.link_components)
    if ell < 2:
        return False
    seen: Dict[tuple, int] = {}
    for c in link.crossings:
        a, b = c.strand_components[0], c.strand_components[1]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if seen.setdefault(key, c.sign) != c.sign:
            return False
    return True


def _to_sage_link(link: Link) -> "SageLink":
    """Convert a spherogram Link to a Sage Link via PD code."""
    pd_code = [list(c) for c in link.PD_code(min_strand_index=1)]
    return SageLink(pd_code)


def _omega_signature_and_nullity(V: "sage.matrix", omega) -> tuple:
    r"""
    Levine-Tristram signature and nullity at \omega, for V the Seifert matrix:
        \sigma_L(\omega) = sign[(1 - \omega)V + (1 - \bar\omega)V^T]
        \eta_L(\omega)   = null[(1 - \omega)V + (1 - \bar\omega)V^T]
    Uses Sage exact arithmetic (QQbar) for the eigenvalues.

    Args:
        V: Sage integer matrix (Seifert matrix).
        omega: a root of unity \omega in S^1 \subset C.

    Returns:
        (signature, nullity) of the Hermitian form.

    Reference: [florensgilmer2003links], Section 2
    """
    omega = QQbar(omega)
    m = (1 - omega) * V + (1 - omega.conjugate()) * V.transpose()
    eigenvalues = m.eigenvalues()
    sig = 0
    nul = 0
    for ev in eigenvalues:
        s = ev.real().sign()
        if s == 1:
            sig += 1
        elif s == -1:
            sig -= 1
        else:
            nul += 1
    return sig, nul


def _sig_null_integer_symmetric(M) -> tuple:
    """
    Exact signature and nullity of an integer symmetric Sage matrix, read off
    its characteristic polynomial: no algebraic numbers involved.

    Args:
        M: Sage symmetric matrix over ZZ.

    Returns:
        (signature, nullity) = (#positive - #negative eigenvalues, #zero).
    """
    charpoly_coeffs = M.charpoly().coefficients(sparse=False)
    nul = 0
    while charpoly_coeffs[nul] == 0:
        nul += 1
    signs = [1 if c > 0 else -1 for c in charpoly_coeffs[nul:] if c != 0]
    num_pos = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    num_neg = M.nrows() - nul - num_pos
    return num_pos - num_neg, nul


# Denote \zeta_a = exp((2\pi i)/a).  The orders a at which the
# Levine-Tristram signature at \zeta_a reduces to integer linear algebra.
_INTEGER_REDUCTION_PRIMES = (2, 3)

# Default orders for the weak Murasugi-Tristram bound, evaluated at
# \zeta_a; any prime power is legal [kauffman1978signature, Theorem 4.1].
_MT_PRIMES_DEFAULT = [2, 3]

# Denote \omega_2 = -1 and \omega_p = \zeta_p^{(p-1)/2} = exp(2q\pi i/(2q+1))
# for p = 2q+1 an odd prime: the points where Tristram's Theorem 2.27 holds
# for disconnected surfaces in B^4 [tristram1969cobordism, Definitions 2.3,
# 2.14].
# Default p for the strong bound.
_STRONG_MT_PRIMES_DEFAULT = [2, 3]

# The p where \omega_p = \zeta_p, so the strong table reuses the weak one.
_TRISTRAM_OMEGA_IS_ZETA_P = (2, 3)


def _validate_mt_order(a: int) -> int:
    r"""
    Checks that \zeta_a is a legal point for the weak Murasugi-Tristram bound,
    i.e. that a is a prime power, and returns a as an int.  Elsewhere the
    bound is false.

    Args:
        a: the order of the root of unity.

    Returns:
        a, as a Python int.

    Raises:
        ValueError: if a is not an integer prime power greater than 1.

    Reference: [kauffman1978signature, Theorem 4.1]
    """
    try:
        n = ZZ(a)
    except (TypeError, ValueError):
        raise ValueError(
            f"Murasugi-Tristram: order {a!r} is not an integer.")
    if n <= 1 or not n.is_prime_power():
        raise ValueError(
            f"Murasugi-Tristram: order {a} is not a prime power greater than "
            f"1.  Kauffman's Theorem 4.1 holds only at roots of unity of prime-power order")
    return int(n)


def _tristram_omega(p: int):
    r"""
    Tristram's \omega_p for the prime p: -1 at p = 2, where (p-1)/2 is not an
    integer, and \zeta_p^{(p-1)/2} otherwise.  Prime powers are rejected; they
    are legal for the weak bound only.

    Args:
        p: a prime.

    Returns:
        \omega_p as a QQbar element.

    Raises:
        ValueError: if p is not prime.

    Reference: [tristram1969cobordism, Definitions 2.3, 2.14] (the point),
               [tristram1969cobordism, Theorem 2.27] (the inequality)
    """
    if p < 2 or not ZZ(p).is_prime():
        raise ValueError(
            f"strong Murasugi-Tristram: p={p} is not prime.  Theorem 2.27 is "
            f"stated for omega = -1 (p=2) or a primitive p-th root of unity "
            f"with p an odd prime; the prime-power extension (Kauffman Thm 4.1) "
            f"holds for the weak genus only.")
    if p == 2:
        return QQbar(-1)
    return QQbar.zeta(p) ** ((p - 1) // 2)


def _omega_signature_and_nullity_integer(V, p: int) -> tuple:
    r"""
    Levine-Tristram signature and nullity at \omega = \zeta_p via pure integer
    arithmetic, for p \in _INTEGER_REDUCTION_PRIMES.

    Args:
        V: Sage integer matrix (Seifert matrix).
        p: 2 or 3.

    Returns:
        (signature, nullity) of the Levine-Tristram form at \zeta_p.

    Reference: [conway2019signature, Definition 1]
    """
    S = V + V.transpose()
    if p == 2:
        return _sig_null_integer_symmetric(S)
    if p == 3:
        K = V - V.transpose()
        doubled = block_matrix([[3 * S, K], [-K, S]], subdivide=False)
        sig2, nul2 = _sig_null_integer_symmetric(doubled)
        return sig2 // 2, nul2 // 2
    raise ValueError(
        f"integer reduction not available for p={p}; "
        f"use _omega_signature_and_nullity(V, QQbar.zeta(p))")


def _sage_seifert_matrix_from_sage_link(sage_link):
    """
    Get the Seifert matrix from an already-converted Sage Link.

    Returns a Sage integer matrix, or None if the link has no crossings.
    """
    V = sage_link.seifert_matrix()
    if V.nrows() == 0:
        return None
    return sage_matrix(ZZ, V)


def _sage_signature_and_nullity_from_matrix(V) -> tuple:
    r"""
    The (Murasugi) signature and nullity of a Seifert matrix,
        \sigma(L) = \sigma_L(-1) = sign(V + V^T),
    i.e. the Levine-Tristram signature at \omega = -1.

    Args:
        V: Sage integer matrix (Seifert matrix), or None.

    Returns:
        (signature, nullity) tuple.

    Reference: [murasugi1965numerical]
    """
    if V is None:
        return 0, 0
    # \omega = \zeta_2 = -1: integer reduction, exact and with no algebraic
    # numbers.
    return _omega_signature_and_nullity_integer(V, 2)


def _sage_signature_and_nullity(link: Link) -> tuple:
    """Compute classical signature and nullity for a link using Sage exact arithmetic."""
    sage_link = _to_sage_link(link)
    V = _sage_seifert_matrix_from_sage_link(sage_link)
    return _sage_signature_and_nullity_from_matrix(V)


# =============================================================================
# FOX-MILNOR SLICENESS OBSTRUCTION (KNOTS ONLY)
# =============================================================================
#
# Two forms of the Fox-Milnor condition [foxmilnor1966cobordism, Theorem 2],
# det(K) is a perfect square, and \Delta_K(t) factors as t^m p(t) p(1/t).  
# A knot failing either is not slice, so g_4(K) >= 1.

def _knot_determinant_from_seifert(V) -> Optional[int]:
    r"""
    det(K) = |\Delta_K(-1)| = |det(V + V^T)|, from the Seifert matrix.

    Returns None when there is no Seifert matrix (crossingless diagram).

    Reference: [foxmilnor1966cobordism, Theorem 2] (a slice knot has square
               determinant)
    """
    if V is None:
        return None
    return int(abs((V + V.transpose()).determinant()))


def _is_perfect_square(n: int) -> bool:
    """True iff the non-negative integer n is a perfect square."""
    if n < 0:
        return False
    r = math.isqrt(int(n))
    return r * r == int(n)


def _alexander_polynomial_from_seifert(V):
    r"""
    \Delta_K(t) = det(V - t*V^T) in Z[t], normalised so that \Delta(0) != 0.
    Sage's Link.alexander_polynomial computes the same determinant, but in 
    Laurent form, which the Fox-Milnor factor test cannot use.
    
    Returns None when V is None.  
    """
    if V is None:
        return None
    R = PolynomialRing(ZZ, 't')
    t = R.gen()
    delta = R((V - t * V.transpose()).determinant())
    if delta == 0:
        return None
    return delta // t ** delta.valuation()


def _fox_milnor_satisfied(delta) -> Optional[bool]:
    r"""
    The full Fox-Milnor test on a knot Alexander polynomial. \Delta_K is
    reciprocal, so its irreducible factors that are not self-reciprocal pair
    up with their reverses, and it factors as t^m p(t) p(1/t) exactly when
    every self-reciprocal factor occurs to an even power.

    Returns:
        True: \Delta admits the factorisation (no obstruction);
        False: it does not, so the knot is not slice;
        None: the input is not a usable knot Alexander polynomial, so no
              claim is made.  For a knot, |\Delta(1)| = 1 always; if this is not 
              the case, something degenerated and the obstruction is not valid.

    Reference: [foxmilnor1966cobordism, Theorem 2]
    """
    if delta is None or delta == 0:
        return None
    # |Delta_K(1)| = |det(V - V^T)| = 1 for a knot, if this is 
    # not the case something degenerated and the bound is not valid.
    if abs(delta(1)) != 1:
        return None
    R = delta.parent()
    t = R.gen()
    p = delta // t ** delta.valuation()
    try:
        factors = p.factor()
    except (ArithmeticError, ValueError, NotImplementedError):
        return None
    return all(e % 2 == 0
               for q, e in factors
               if q.degree() > 0 and q.reverse() in (q, -q))


def _omega_sig_null_table(V, mt_primes: Optional[List[int]] = None) -> Dict[int, tuple]:
    r"""
    The pairs (\sigma_L(\zeta_p), \eta_L(\zeta_p)) for every p in mt_primes,
    computed once per link and shared by every bound that needs them.

    Every \omega here is a primitive a-th root of unity of prime-power order a,
    which is the domain of the weak bound (Kauffman Thm 4.1) and a subset of
    the domain of Conway's Thm 1 (all of S^1 minus 1).  It is not the domain of
    Tristram's Thm 2.27, whose \omega_p differs from \zeta_p for p >= 5; the
    strong bound takes _tristram_sig_null_table instead.

    The nullities returned are the raw matrix ones.  They are Conway's
    \eta_L(\omega) only when the Seifert surface underlying V is
    connected, so every bound reading them is claimed only then; see
    _has_connected_seifert_surface.

    Args:
        V: Sage integer matrix (Seifert matrix), or None.
        mt_primes: orders a, each a prime power greater than 1;
            \omega = \zeta_a.  Defaults to [2, 3].

    Returns:
        {p: (signature, nullity)}; empty when V is None.
    """
    if V is None:
        return {}
    if mt_primes is None:
        mt_primes = _MT_PRIMES_DEFAULT
    table = {}
    for p in (_validate_mt_order(a) for a in mt_primes):
        if p in _INTEGER_REDUCTION_PRIMES:
            # p \in {2, 3}: exact integer reduction (fast path).
            table[p] = _omega_signature_and_nullity_integer(V, p)
        else:
            # Other primes: QQbar reference path (slow but general).
            table[p] = _omega_signature_and_nullity(V, QQbar.zeta(p))
    return table


def _murasugi_tristram_bound_from_table(sig_null: Dict[int, tuple], ell: int) -> int:
    r"""
    Murasugi-Tristram lower bound on the weak slice genus:
        g_4(L) >= \lceil (|\sigma_L(\omega)| + \eta_L(\omega) - (\ell - 1))/2 \rceil
    for \omega a prime-power root of unity, where:
        - \sigma_L(\omega) is the Levine-Tristram signature at \omega,
        - \eta_L(\omega) is the nullity at \omega,
        - \ell is the number of link components.
    Returns the maximum over all the primes in the table.

    \eta must be Conway's nullity, which the raw matrix nullity is
    only on a connected Seifert surface: callers claim the bound
    only when _has_connected_seifert_surface holds.

    Args:
        sig_null: {p: (\sigma, \eta)} as produced by _omega_sig_null_table.
        ell: number of link components \ell.

    Reference: [tristram1969cobordism, Theorem 2.27],
               [kauffman1978signature, Theorem 4.1]
    """
    max_bound = 0
    for sig, nul in sig_null.values():
        bound = max(0, ceil((abs(sig) + nul - (ell - 1)) / 2))
        max_bound = max(max_bound, bound)
    return max_bound


def _tristram_sig_null_table(V, primes: Optional[List[int]] = None,
                             reuse: Optional[Dict[int, tuple]] = None) -> Dict[int, tuple]:
    r"""
    The pairs (\sigma_L(\omega_p), \eta_L(\omega_p)) at Tristram's \omega_p,
    for every p in primes: the strong bound's table.  It differs from
    _omega_sig_null_table, which evaluates at \zeta_p, for every p >= 5.

    Args:
        V: Sage integer matrix (Seifert matrix), or None.
        primes: primes p.  Defaults to _STRONG_MT_PRIMES_DEFAULT.  Non-primes
            raise (see _tristram_omega).
        reuse: an _omega_sig_null_table result to draw from for the primes
            where \omega_p == \zeta_p, so the charpoly is not paid twice.

    Returns:
        {p: (signature, nullity)}; empty when V is None.

    Reference: [tristram1969cobordism, Theorem 2.27]
    """
    if V is None:
        return {}
    if primes is None:
        primes = _STRONG_MT_PRIMES_DEFAULT
    table = {}
    for p in primes:
        omega = _tristram_omega(p)  # validates primality
        if p in _TRISTRAM_OMEGA_IS_ZETA_P:
            # \omega_p == \zeta_p: reuse the weak table if we have it,
            # otherwise take the fast path.
            if reuse is not None and p in reuse:
                table[p] = reuse[p]
            else:
                table[p] = _omega_signature_and_nullity_integer(V, p)
        else:
            # p >= 5: \omega_p is not \zeta_p, so there is no integer
            # reduction and the QQbar path, which is much slower, is taken.
            table[p] = _omega_signature_and_nullity(V, omega)
    return table


def _strong_murasugi_tristram_bound_from_table(sig_null: Dict[int, tuple], ell: int) -> int:
    r"""
    Murasugi-Tristram lower bound on the strong slice genus:
        g_4^*(L) >= \lceil (|\sigma_L(\omega)| + |\eta_L(\omega) + 1 - \ell|)/2 \rceil

    Returns the maximum over the table.

    The hypothesis on \omega is narrower than the weak bound's, so the table
    must come from _tristram_sig_null_table and not from _omega_sig_null_table;
    the two agree only at p \in {2, 3}.

    Args:
        sig_null: {p: (\sigma, \eta)} as produced by _tristram_sig_null_table
            from a V whose Seifert surface is connected, so that the
            nullities are Conway's \eta (_has_connected_seifert_surface).
        ell: number of link components \ell.

    Reference: [tristram1969cobordism, Theorem 2.27] (strong slice case,
               m = \ell)
    """
    max_bound = 0
    for sig, nul in sig_null.values():
        bound = max(0, ceil((abs(sig) + abs(nul + 1 - ell)) / 2))
        max_bound = max(max_bound, bound)
    return max_bound


def _murasugi_tristram_bound_from_matrix(V, ell: int, mt_primes: Optional[List[int]] = None) -> int:
    """
    Weak Murasugi-Tristram bound computed from a Seifert
    matrix: builds the signature/nullity table and applies
    the bound to it.  Returns 0 (not claimed) when the
    Seifert surface is disconnected.

    Used by murasugi_tristram_bound and the per-component
    pass.  compute_bounds builds the table itself, so the
    strong bound can reuse it.


    Reference: [tristram1969cobordism, Theorem 2.27],
               [kauffman1978signature, Theorem 4.1]
    """
    if not _has_connected_seifert_surface(V, ell):
        return 0
    return _murasugi_tristram_bound_from_table(_omega_sig_null_table(V, mt_primes), ell)


# =============================================================================
# BOUND FORMULA FUNCTIONS (pure arithmetic)
# =============================================================================

def _signature_lower_bound(sig: int, ell: int) -> int:
    r"""
    Signature-based lower bound on the (weak) slice genus:
        g_4(L) >= \lceil (|\sigma(L)| + 1 - \ell)/2 \rceil

    Valid for every link, with no alternating or non-split hypothesis: it is
    the Murasugi-Tristram inequality at \omega = -1 with the non-negative
    nullity term dropped.

    Args:
        sig: Murasugi signature \sigma(L).
        ell: number of link components \ell.

    Reference: [murasugi1965numerical, Theorem 9.1]
               [cavallo2015links, Corollary 3.2] (alternating case: s = -\sigma)
    """
    return max(0, ceil((abs(sig) + 1 - ell) / 2))


def _seifert_genus_from_diagram(c: int, o: int, ell: int, ell_s: int) -> int:
    r"""
    Upper bound on the Seifert genus of the current diagram:
        g_3 <= (c(D) - O(D) + 2\ell_s - \ell)/2
    where:
        - c = c(D) is the number of crossings of the diagram,
        - o = O(D) is the number of Seifert circles of the diagram,
        - \ell is the number of link components,
        - \ell_s is the number of split (connected) components of the diagram.

    For a positive link this is exactly the 3-genus, and also the 4-genus; by
    mirroring the same holds for a negative link.

    Reference: [cavallo2015links, Proposition 4.1]
               [nakamura2000positive, Theorem 1.1] (positive case)
    """
    return (c - o + 2 * ell_s - ell) // 2


def _s_invariant_positive(c: int, o: int) -> int:
    """
    Rasmussen s-invariant of a positive diagram:
        s(L) = c(D) - O(D) + 1
    where c = c(D) is the number of crossings and o = O(D) the number of
    Seifert circles of the diagram.

    Reference: [cavallo2015links, Proposition 3.3]
    """
    return c - o + 1


def _pseudo_thin_strong_signature_bound(sig: int, ell: int) -> int:
    r"""
    Lower bound on the strong slice genus of a non-split pseudo-thin
    algebraically split link:
        g_4^*(L) >= \lceil (|\sigma(L)| + \ell - 1)/2 \rceil
    It is the pseudo-thin bound with s(L) = -\sigma(L).

    Args:
        sig: Murasugi signature \sigma(L).
        ell: number of link components \ell.

    Reference: [cavallo2015links, Theorem 5.4 + Corollary 3.2]
    """
    return ceil((abs(sig) + ell - 1) / 2)


def _pseudo_thin_strong_bound(s: int, ell: int) -> int:
    r"""
    Lower bound on the strong slice genus of a non-split, pseudo-thin,
    algebraically split link:
        g_4^*(L) >= \lceil (|s(L)| + \ell - 1)/2 \rceil

    The pipeline certifies pseudo-thinness by a connected alternating
    diagram: such a link is quasi-alternating, hence pseudo-thin.

    Args:
        s: Rasmussen s-invariant of L.
        ell: number of link components \ell.

    Reference: [cavallo2015links, Theorem 5.4],
               [ozsvath2005branched, Lemma 3.2] (alternating is
               quasi-alternating)
    """
    return max(0, ceil((abs(s) + ell - 1) / 2))


def _slice_torus_strong_bound(s: int, ell: int) -> int:
    r"""
    Slice-torus lower bound on the strong slice genus of an algebraically
    split link, with \nu = \nu_s = (s + \ell - 1)/2:
        g_4^*(L) >= |\nu_s(L)| = |s(L) + \ell - 1| / 2

    Args:
        s: Rasmussen s-invariant of L.
        ell: number of link components \ell.

    Reference: [beliakova2008categorification, equation (7.1)]
               [cavallocollari2020links, Proposition 2.11] (general \nu)
    """
    return max(0, ceil(abs(s + ell - 1) / 2))


def _slice_torus_unknotting_bound(s: int, ell: int) -> int:
    r"""
    Slice-torus lower bound on the unknotting number, with \nu = \nu_s:
        u(L) >= |\nu_s(L)| = |s(L) + \ell - 1| / 2
    No hypothesis on L.

    Args:
        s: Rasmussen s-invariant of L.
        ell: number of link components \ell.

    Reference: [cavallocollari2020links, Proposition 2.10]
    """
    return max(0, ceil(abs(s + ell - 1) / 2))


def _splitting_number_unknotting_bound(s: int, s_components: List[Optional[int]],
                                      ell: int) -> Optional[int]:
    r"""
    Splitting-number lower bound on the unknotting number, with \nu = \nu_s:
        u(L) >= \widetilde{sp}(L) >= |\nu_s(L) - \sum_i \nu_s(K_i)|
              = |s(L) + \ell - 1 - \sum_i s(K_i)| / 2
    using \nu_s(K_i) = s(K_i)/2 for a knot.

    Returns None unless s(L) and every s(K_i) is available.

    Args:
        s: Rasmussen s-invariant of L.
        s_components: s(K_i) per component, None where not derivable.
        ell: number of link components \ell.

    Reference: [cavallocollari2020links, Theorem 1.5]
    """
    if s is None or any(v is None for v in s_components):
        return None
    return max(0, ceil(abs(s + ell - 1 - sum(s_components)) / 2))


def _conway_lt_unknotting_bound_from_table(sig_null: Dict[int, tuple],
                                          ell: int) -> int:
    r"""
    Levine-Tristram lower bound on the unknotting number:
        u(L) >= \lceil (|\sigma_L(\omega)| + |\eta_L(\omega) - \ell + 1|)/2 \rceil
    for any \omega \in S^1 \setminus \{1\}, so any table may be fed in.
    Returns the maximum over the table.

    \eta must be Conway's nullity, which the raw matrix nullity is
    only on a connected Seifert surface: callers claim the bound
    only when _has_connected_seifert_surface holds.

    Args:
        sig_null: {p: (\sigma, \eta)} at any set of evaluation points.
        ell: number of link components \ell.

    Reference: [conway2019signature, Theorem 1]
               [nagelowens2015unlinking, Lemma 2.1]
    """
    max_bound = 0
    for sig, nul in sig_null.values():
        bound = max(0, ceil((abs(sig) + abs(nul - ell + 1)) / 2))
        max_bound = max(max_bound, bound)
    return max_bound


def _unlink_certificates(sig: int, eta_conway: Optional[int],
                         s: Optional[int], det: Optional[int],
                         hfk_genus: Optional[int], ell: int) -> List[str]:
    r"""
    A certificate for L to not be the \ell-component unlink, 
    and hence u(L) >= 1.

    The unlink has \sigma = 0, \eta_L(\omega) = \ell - 1 at every \omega, 
    s = 1 - \ell, and g_3 = 0, and the unknot has det = 1, so a link 
    differing in any one of them is not trivial.

    Args:
        sig: Murasugi signature \sigma(L).
        eta_conway: the nullity at \omega = -1 when the Seifert
            surface is connected, hence Conway's; None otherwise.
        s: Rasmussen s-invariant, or None.
        det: |det(V + V^T)|, read only for a knot and only when the
            Seifert surface is connected; pass None otherwise.
        hfk_genus: the exact HFK Seifert genus (knots only), or None.
        ell: number of link components.

    Returns:
        The reasons certificating L is not the unlink, possibly empty.

    Reference: [murasugi1965numerical, Theorem 9.1] (\sigma),
               [conway2019signature, Definition 1] (\eta),
               [beliakova2008categorification, Section 7.1] (s of a split
               union), [ozsvath2004knotfloer] (HFK)
    """
    reasons = []
    if hfk_genus is not None and hfk_genus > 0:
        reasons.append('hfk_genus')
    if sig != 0:
        reasons.append('signature')
    if s is not None and s != 1 - ell:
        reasons.append('s_invariant')
    if eta_conway is not None and eta_conway != ell - 1:
        reasons.append('nullity')
    if ell == 1 and det is not None and det != 1:
        reasons.append('determinant')
    return reasons


def _has_connected_seifert_surface(V, ell: int) -> bool:
    r"""
    True when the Seifert surface S underlying V is connected, so that
    the matrix nullity is Conway's \eta_L(\omega).  The three bounds that
    read a nullity (weak and strong Murasugi-Tristram, Conway's unknotting
    bound) are claimed only then.

    S is the surface Seifert's algorithm builds on the closed braid Sage
    derives from the link (Link.seifert_matrix), not on the input diagram,
    completed by a disk for each crossingless component; hence
    \beta_0(S) = \ell - null(V - V^T), with \ell counting those components.

    Args:
        V: Sage integer matrix (Seifert matrix), or None.
        ell: number of link components, crossingless ones included.

    Returns:
        Whether S is connected; False when V is None.

    Reference: [conway2019signature, Definition 1] (the nullity convention)
    """
    if V is None:
        return False
    q = V.nrows() - (V - V.transpose()).rank()
    return ell - q == 1


def _compute_hfk(link: Link) -> Optional[dict]:
    """
    Knot Floer homology via spherogram-Python (knots only).

    Returns:
        spherogram's HFK dict (seifert_genus, tau, nu, epsilon,
        fibered, ...), or None for a link or on failure.
    """
    if len(link.link_components) != 1 or link.unlinked_unknot_components > 0:
        return None
    try:
        return link.knot_floer_homology()
    except Exception:
        return None


# =============================================================================
# COMPONENT-WISE BOUNDS
# =============================================================================
#
# Four bounds read the components K_i of a link:
#
#   g_4^*(L) >= \sum_i g_4(K_i)                  [strong slice]
#   u(L)     >= \abslk{L} + \sum_i u(K_i)        [unknotting]
#   u(L)     >= |\nu_s(L) - \sum_i \nu_s(K_i)|   [unknotting, s known]
#   u(L)     <= \abslk{L} + \sum_i \bar u(K_i)   [unknotting, simply-linked]
#
# The per-component numbers come from a knots-only cut of the pipeline on a
# simplified copy of Link.sublink(i) (KnotJob is never called; s(K_i) comes
# from the closed forms or from precomputed values), merged, when a 
# knot_identification.KnotTable is available, with the best known g_4 and u 
# of the identified knot, the stronger value winning. Sub-diagrams are simplified first.

# Source labels of a merged ComponentBounds field; 'known:...' and
# 'subadditive:...' come from knot_identification.known_values().
_COMPONENT_UNKNOT_SOURCE = 'unknot'                    # a crossingless component
_COMPONENT_UNEVALUATED_SOURCE = 'unevaluated'          # the pass failed on it
_FOX_MILNOR_DET_SOURCE = 'fox_milnor_det'              # lower: det not a square
_FOX_MILNOR_FULL_SOURCE = 'fox_milnor_full'            # lower: the full test
_COMPONENT_SEIFERT_GENUS_SOURCE = 'seifert_genus_diagram'  # upper: the diagram
_COMPONENT_HFK_SEIFERT_GENUS_SOURCE = 'hfk_seifert_genus'  # upper: HFK tightened it
_EXACT_POSITIVE_SOURCE = 'exact_positive'              # genus: positive diagram
_EXACT_NEGATIVE_SOURCE = 'exact_negative'              # genus: negative diagram
_COMPONENT_CROSSING_CHANGE_SOURCE = 'crossing_change'  # u upper: floor(c/2)
_COMPONENT_S_POSITIVE_SOURCE = 's_positive'            # s: positive diagram
_COMPONENT_S_ALTERNATING_SOURCE = 's_alternating'      # s: alternating diagram
_COMPONENT_CONTRADICTION_SOURCE = 'contradiction'       # dropped, see below


class _EmptyKnownValues:
    """
    A knot_identification.KnownValues with every field None or
    False, used when identification is off or found nothing.
    """
    g4_lower = g4_upper = u_lower = u_upper = s = None
    g4_lower_source = g4_upper_source = None
    u_lower_source = u_upper_source = s_source = None
    knotted = False


_EMPTY_KNOWN_VALUES = _EmptyKnownValues()


@dataclass
class ComponentBounds:
    """
    Bounds on one component knot K_i: lower and upper bounds on
    g_4(K_i) and u(K_i), s(K_i), whether K_i is certified knotted,
    and a source string per value.  name and identification are
    what knot_identification.identify() found, None when it was
    not run.  contradiction carries the message when a pipeline
    bound disagrees with a known value; the component then
    contributes the all-zero record.
    """
    index: int
    num_crossings: int
    genus_lower: int
    genus_upper: Optional[int]
    u_lower: int
    u_upper: Optional[int]
    s: Optional[int] = None
    knotted: bool = False
    genus_lower_source: str = _COMPONENT_UNEVALUATED_SOURCE
    genus_upper_source: str = _COMPONENT_UNEVALUATED_SOURCE
    u_lower_source: str = _COMPONENT_UNEVALUATED_SOURCE
    u_upper_source: str = _COMPONENT_UNEVALUATED_SOURCE
    s_source: Optional[str] = None
    name: Optional[str] = None
    identification: Optional[str] = None
    contradiction: Optional[str] = None


# Per-knot work, cached on the PD code of the simplified sub-diagram, the
# Murasugi-Tristram orders and the identity of the table and known-values
# files: the same component knots recur across a census.
_COMPONENT_CACHE: Dict[tuple, ComponentBounds] = {}

def _component_record(source: str, **fields) -> ComponentBounds:
    """A zero ComponentBounds with every source set to source."""
    values = dict(index=-1, num_crossings=0, genus_lower=0, genus_upper=None,
                  u_lower=0, u_upper=None, genus_lower_source=source,
                  genus_upper_source=source, u_lower_source=source,
                  u_upper_source=source)
    values.update(fields)
    return ComponentBounds(**values)


_COMPONENT_UNEVALUATED = _component_record(_COMPONENT_UNEVALUATED_SOURCE)
_COMPONENT_UNKNOT = _component_record(
    _COMPONENT_UNKNOT_SOURCE, genus_upper=0, u_upper=0, s=0,
    s_source=_COMPONENT_UNKNOT_SOURCE, name='0_1', identification='identified')
_COMPONENT_CONTRADICTION = _component_record(_COMPONENT_CONTRADICTION_SOURCE)


_IDENTIFICATION_IMPORT_ERROR_SAID = False


def _load_identification():
    """
    Imports knot_identification lazily, that module importing
    from this one; an import failure is printed once per process.

    Returns:
        The module, or None when it cannot be imported.
    """
    global _IDENTIFICATION_IMPORT_ERROR_SAID
    try:
        import bounds_pipeline.knot_identification as knot_identification
    except ImportError as exc:
        if not _IDENTIFICATION_IMPORT_ERROR_SAID:
            print(f"  component identification disabled: {exc}")
            _IDENTIFICATION_IMPORT_ERROR_SAID = True
        return None
    return knot_identification


def _merge_component_field(theo_value, theo_source: str,
                           known_value: Optional[int],
                           known_source: Optional[str], better) -> tuple:
    """
    Merges a pipeline value with a known one, the stronger winning
    and the known value winning ties.

    Args:
        theo_value: the pipeline's value.
        theo_source: its source label.
        known_value: the known value, or None.
        known_source: its source label.
        better: max for a lower bound, min for an upper one.

    Returns:
        (merged_value, merged_source).
    """
    if known_value is not None and better(theo_value, known_value) == known_value:
        return known_value, known_source
    return theo_value, theo_source


def _knot_component_bounds(knot: Link, raw_crossings: int,
                           mt_primes: Optional[List[int]] = None,
                           identification_module=None,
                           table=None, known=None) -> ComponentBounds:
    r"""
    The knots-only bound pass on one component sub-diagram, merged
    with the component's best known values when it identifies.

    g_4(K_i) is bounded below by the signature, Murasugi-Tristram,
    HFK tau and nu and Fox-Milnor, and above by the Seifert genus
    of the diagram, sharpened by HFK; on a positive or negative
    non-split sub-diagram that Seifert genus is the exact g_4 and
    settles both.  u(K_i) is bounded below by g_4(K_i), or by 1
    when K_i is certified knotted, and above by
    \lfloor c/2 \rfloor; s(K_i) comes from the positive or
    alternating closed form.  When table is given, the diagram is
    identified and each bound is merged with the known one, the
    stronger winning.  A pipeline lower bound above a known upper
    bound, or a closed-form s disagreeing with the table, is a
    contradiction: the component contributes the all-zero record,
    with the message in its contradiction field.

    Args:
        knot: the simplified sub-diagram of the component.
        raw_crossings: crossings of the unsimplified sub-diagram,
            for the \lfloor c/2 \rfloor bound.
        mt_primes: orders for the weak Murasugi-Tristram bound.
        identification_module: the knot_identification module,
            or None to skip identification.
        table: a knot_identification.KnotTable, or None.
        known: a knot_identification.KnownValueTable, or None.

    Reference: [nakamura2000positive, Theorem 1.1] (exact genus),
               [murasugi1965numerical, Theorem 9.1] (signature),
               [tristram1969cobordism, Theorem 2.27] and
               [kauffman1978signature, Theorem 4.1] (Murasugi-Tristram),
               [ozsvath2003fourball, Corollary 1.3] (\tau),
               [hom2016nuplus, Propositions 2.3 and 2.4] (\nu),
               [foxmilnor1966cobordism, Theorem 2] (Fox-Milnor)
    """
    c = len(knot.crossings)
    if c == 0:
        return replace(_COMPONENT_UNKNOT)

    sage_knot = _to_sage_link(knot)
    V = _sage_seifert_matrix_from_sage_link(sage_knot)
    sig, _nul = _sage_signature_and_nullity_from_matrix(V)
    o = len(sage_knot.seifert_circles())

    candidates: Dict[str, int] = {}
    # Signature: |\sigma(K)| <= 2 g_4(K).
    candidates['signature'] = max(0, ceil(abs(sig) / 2))
    # Murasugi-Tristram at \zeta_p with \ell = 1; a knot's Seifert surface
    # is connected, so the gate passes.
    try:
        candidates['murasugi_tristram'] = _murasugi_tristram_bound_from_matrix(
            V, 1, mt_primes)
    except Exception:
        pass
    # Fox-Milnor at t = -1: det(K) is a perfect square for a slice knot.
    det_k = _knot_determinant_from_seifert(V)
    if det_k is not None and not _is_perfect_square(det_k):
        candidates[_FOX_MILNOR_DET_SOURCE] = 1

    # Heegaard Floer: |\tau| <= g_4 and \nu <= g_4, the latter on the 
    # mirror too since g_4 is mirror-invariant.
    hfk = _compute_hfk(knot)
    hfk_genus = None
    if hfk is not None:
        hfk_genus = hfk.get('seifert_genus')
        if hfk.get('tau') is not None:
            candidates['tau_hfk'] = abs(int(hfk['tau']))
        nu_vals = [hfk.get('nu')]
        try:
            hfk_m = _compute_hfk(knot.mirror())
            nu_vals.append(hfk_m.get('nu') if hfk_m else None)
        except Exception:
            pass
        nu_vals = [int(v) for v in nu_vals if v is not None]
        if nu_vals:
            candidates['nu_hfk'] = max(0, max(nu_vals))

    theo_genus_lower = max(candidates.values()) if candidates else 0
    theo_genus_lower_src = ' + '.join(
        sorted(n for n, v in candidates.items()
              if v == theo_genus_lower and v > 0)) or 'none'

    # Full Fox-Milnor, last resort exactly as in the main pipeline.
    if theo_genus_lower == 0:
        try:
            if _fox_milnor_satisfied(
                    _alexander_polynomial_from_seifert(V)) is False:
                theo_genus_lower = 1
                theo_genus_lower_src = _FOX_MILNOR_FULL_SOURCE
        except Exception:
            pass

    # Upper bound on g_4: the Seifert genus of the diagram, sharpened by HFK.
    split = _num_split_components(knot)
    theo_genus_upper = _seifert_genus_from_diagram(c, o, 1, split)
    theo_genus_upper_src = _COMPONENT_SEIFERT_GENUS_SOURCE
    # On a positive non-split sub-diagram that same number is the exact g_3,
    # and a positive knot has g_4 = g_3, so it is a lower bound as well.  A
    # negative diagram is the mirror of a positive one and both genera are
    # mirror-invariant.
    exact_genus_src = None
    if split == 1 and _is_positive_diagram(knot):
        exact_genus_src = _EXACT_POSITIVE_SOURCE
    elif split == 1 and _is_negative_diagram(knot):
        exact_genus_src = _EXACT_NEGATIVE_SOURCE
    exact_genus = theo_genus_upper if exact_genus_src else None
    if hfk_genus is not None and int(hfk_genus) < theo_genus_upper:
        theo_genus_upper = int(hfk_genus)
        theo_genus_upper_src = _COMPONENT_HFK_SEIFERT_GENUS_SOURCE

    # Certificates that K is knotted, hence u(K) >= 1: the exact HFK
    # Seifert genus, det != 1, \sigma != 0.
    knot_certs = []
    if hfk_genus is not None and hfk_genus > 0:
        knot_certs.append('hfk_genus')
    if sig != 0:
        knot_certs.append('signature')
    if det_k is not None and det_k != 1:
        knot_certs.append('determinant')
    pipeline_knotted = bool(theo_genus_lower > 0 or knot_certs)

    theo_u_upper = min(c, raw_crossings) // 2

    # s(K_i) from the closed forms only; a component never goes to KnotJob.
    s_theo: Optional[int] = None
    s_theo_src: Optional[str] = None
    if _is_positive_diagram(knot):
        s_theo = _s_invariant_positive(c, o)
        s_theo_src = _COMPONENT_S_POSITIVE_SOURCE
    elif _is_alternating_diagram(knot) and not _is_split(knot):
        s_theo = -sig
        s_theo_src = _COMPONENT_S_ALTERNATING_SOURCE

    identification = None
    kv = None
    if identification_module is not None and table is not None:
        identification = identification_module.identify(
            knot, table, seifert_matrix=V, hfk=hfk)
        kv = identification_module.known_values(identification, table, known)
    if kv is None:
        kv = _EMPTY_KNOWN_VALUES

    contradictions = []
    if exact_genus is not None:
        if theo_genus_lower > exact_genus:
            contradictions.append(
                f"component genus_4 lower bound {theo_genus_lower} exceeds "
                f"the exact genus {exact_genus} of its {exact_genus_src} "
                "sub-diagram")
        theo_genus_lower, theo_genus_lower_src = exact_genus, exact_genus_src
    if kv.g4_upper is not None and theo_genus_lower > kv.g4_upper:
        contradictions.append(
            f"component genus_4 lower bound {theo_genus_lower} exceeds the "
            f"known genus_4 upper bound {kv.g4_upper} of {identification.label}")
    genus_lower, genus_lower_src = _merge_component_field(
        theo_genus_lower, theo_genus_lower_src, kv.g4_lower,
        kv.g4_lower_source, max)
    genus_upper, genus_upper_src = _merge_component_field(
        theo_genus_upper, theo_genus_upper_src, kv.g4_upper,
        kv.g4_upper_source, min)

    knotted = pipeline_knotted or kv.knotted
    if kv.knotted and not pipeline_knotted:
        knot_certs.append('identified')
    theo_u_lower = max(genus_lower, 1 if knotted else 0)
    theo_u_lower_src = (genus_lower_src if theo_u_lower == genus_lower
                        else 'knotted:' + '+'.join(sorted(knot_certs)))
    if kv.u_upper is not None and theo_u_lower > kv.u_upper:
        contradictions.append(
            f"component u lower bound {theo_u_lower} exceeds the known u "
            f"upper bound {kv.u_upper} of {identification.label}")
    u_lower, u_lower_src = _merge_component_field(
        theo_u_lower, theo_u_lower_src, kv.u_lower, kv.u_lower_source, max)
    u_upper, u_upper_src = _merge_component_field(
        theo_u_upper, _COMPONENT_CROSSING_CHANGE_SOURCE, kv.u_upper,
        kv.u_upper_source, min)

    if s_theo is not None and kv.s is not None and s_theo != kv.s:
        contradictions.append(
            f"component s = {s_theo} ({s_theo_src}) disagrees with the "
            f"known s = {kv.s} ({kv.s_source}) of {identification.label}")
    s, s_src = ((s_theo, s_theo_src) if s_theo is not None
               else (kv.s, kv.s_source))

    name = identification.label if identification is not None else None
    status = identification.status if identification is not None else None
    if contradictions:
        return replace(_COMPONENT_CONTRADICTION, num_crossings=c, name=name,
                       identification=status,
                       contradiction=' | '.join(contradictions))

    return ComponentBounds(
        index=-1, num_crossings=c, genus_lower=genus_lower,
        genus_upper=genus_upper, u_lower=u_lower, u_upper=u_upper, s=s,
        knotted=knotted, genus_lower_source=genus_lower_src,
        genus_upper_source=genus_upper_src, u_lower_source=u_lower_src,
        u_upper_source=u_upper_src, s_source=s_src, name=name,
        identification=status)


def _component_bounds(link: Link, mt_primes: Optional[List[int]] = None,
                      use_identification: bool = True
                      ) -> Optional[List[ComponentBounds]]:
    r"""
    Per-component bounds for every component of link, the
    crossingless unknots included.

    A component the pass cannot evaluate, or whose pipeline bound
    contradicts its known value, gets a zero record: its lower
    bounds count as 0 in the sums \sum_i g_4(K_i) and
    \abslk + \sum_i u(K_i), and its upper bound and s are None,
    which switches off the consumers needing them.  The caller
    adds a contradiction to the link's consistency_errors.

    Args:
        link: the link.
        mt_primes: orders for the weak Murasugi-Tristram bound.
        use_identification: merge in the known values of
            knot_identification.identify(); True by default.

    Reference: [collari2021slicetoruspositive, proof of Lemma 2.8]
    """
    ki = _load_identification() if use_identification else None
    table = ki.default_table() if ki is not None else None
    known = (ki.load_known_values(table=table) if ki is not None
             and table is not None else None)

    out: List[ComponentBounds] = []
    for i in range(len(link.link_components)):
        try:
            sub = link.sublink(i)
        except Exception:
            out.append(replace(_COMPONENT_UNEVALUATED, index=i))
            continue
        raw_c = len(sub.crossings)
        try:
            simplified = sub.copy()
            simplified.simplify('global')
            # The orders and the table/known-values files change the
            # answer, so they are part of the cache key.
            key = (str(simplified.PD_code()) if simplified.crossings else 'unknot',
                   tuple(mt_primes) if mt_primes else None,
                   table.cache_key if table is not None else None,
                   known.identity if known is not None else None)
            cached = _COMPONENT_CACHE.get(key)
            if cached is None:
                cached = _knot_component_bounds(
                    simplified, raw_c, mt_primes, ki, table, known)
                _COMPONENT_CACHE[key] = cached
            if (cached.genus_upper is not None
                    and cached.genus_lower > cached.genus_upper):
                # The lower bound exceeds the upper bound, a bug: the
                # component contributes nothing.
                out.append(replace(_COMPONENT_UNEVALUATED, index=i))
                continue
            cb = replace(cached, index=i)
            # \lfloor c/2 \rfloor is a diagram quantity: the raw sub-diagram may
            # beat the cached (simplified) one, and vice versa.
            if cb.u_upper is not None and raw_c // 2 < cb.u_upper:
                cb.u_upper = raw_c // 2
                cb.u_upper_source = _COMPONENT_CROSSING_CHANGE_SOURCE
            elif cb.u_upper is None:
                cb.u_upper = raw_c // 2
                cb.u_upper_source = _COMPONENT_CROSSING_CHANGE_SOURCE
            out.append(cb)
        except Exception:
            out.append(replace(_COMPONENT_UNEVALUATED, index=i))
    for j in range(link.unlinked_unknot_components):
        out.append(replace(_COMPONENT_UNKNOT,
                           index=len(link.link_components) + j))
    return out


# =============================================================================
# BOUNDS RESULT DATACLASS
# =============================================================================

@dataclass
class BoundsResult:
    """Result of computing all bounds for a link."""
    # Link properties
    is_knot: bool = False
    num_components: int = 0
    is_positive: bool = False
    is_negative: bool = False
    is_alternating: bool = False
    is_split: bool = False
    is_algebraically_split: bool = False
    # Every crossing between two fixed components has the same sign.
    is_simply_linked: bool = False
    # \abslk{L} = \sum_{i<j} |lk(K_i, K_j)|
    abs_linking_number: int = 0

    # Diagram invariants
    num_crossings: int = 0
    num_seifert_circles: int = 0
    writhe: int = 0
    signature: int = 0
    nullity: int = 0

    # Rasmussen s-invariant and where it came from: the positive closed
    # form, KnotJob, or s = -\sigma on a non-split alternating diagram.
    s_invariant: Optional[int] = None
    s_invariant_from_positive: bool = False
    s_invariant_from_knotjob: bool = False
    s_invariant_from_alternating_signature: bool = False

    # Slice-torus invariant \nu_s = (s + \ell - 1)/2
    nu_s: Optional[float] = None

    # Knots only: det(K) = |det(V + V^T)|, whether it is a perfect square,
    # and the full Fox-Milnor test (False = K is not slice, None = not run).
    determinant: Optional[int] = None
    determinant_is_square: Optional[bool] = None
    fox_milnor_satisfied: Optional[bool] = None

    # Slice genus bounds
    slice_upper_seifert: int = 0
    slice_lower_signature: int = 0
    slice_lower_mt: int = 0
    slice_lower_nu_s: Optional[int] = None  # from \nu_s = (s + \ell - 1)/2
    nu_s_bound1: Optional[int] = None  # g_4 >= ceil(-\nu_s)
    nu_s_bound2: Optional[int] = None  # g_4 >= ceil(\nu_s - \ell + 1)
    slice_lower_tau: Optional[int] = None  # |\tau(K)|, knots only
    slice_lower_nu_hfk: Optional[int] = None  # max(\nu(K), \nu(mK)), knots only
    # Fox-Milnor, knots only: 1 when det(K) is not a perfect square, and 1
    # when \Delta_K has a self-reciprocal factor to an odd power (last resort).
    slice_lower_fox_milnor_det: Optional[int] = None
    slice_lower_fox_milnor_full: Optional[int] = None
    slice_exact: Optional[int] = None
    slice_genus_is_exact: bool = False

    # Strong slice genus bounds
    strong_slice_lower_from_slice: int = 0
    strong_slice_lower_signature: Optional[int] = None
    # Murasugi-Tristram for a strong slice surface (m = \ell); None outside
    # the algebraically split branch or on a disconnected Seifert surface.
    strong_slice_lower_mt: Optional[int] = None
    # g_4^* >= |\nu_s| = |s + \ell - 1|/2, the slice-torus strong bound.
    strong_slice_lower_nu_s: Optional[int] = None
    # g_4^* >= (|s| + \ell - 1)/2 on a non-split pseudo-thin diagram.
    strong_slice_lower_pseudo_thin: Optional[int] = None
    # g_4^* >= \sum_i g_4(K_i), from the components' own slice bounds.
    strong_slice_lower_components: Optional[int] = None
    strong_slice_lower_obstruction: int = 0  # 1 if obstructed
    # 9999 is the dataset convention for "no upper bound"; the strong slice
    # genus of the link is stated by strong_slice_genus_lower / _upper.
    strong_slice_upper: int = 9999

    # The strong slice genus of the link: zero for a nonempty crossingless unlink,
    # g_4^*(K) = g_4(K) for a knot, infinity when some linking number is non-zero
    # (then strong_slice_genus_is_infinite is True), and the aggregate of the Phase-7
    # bounds, best_strong_slice_lower, for an algebraically split link.
    strong_slice_genus_lower: float = 0.0
    strong_slice_genus_upper: float = math.inf
    strong_slice_genus_is_infinite: bool = False

    # Obstructions
    slice_obstructed: bool = False
    strong_slice_obstructed: bool = False
    strong_slice_obstruction_reason: str = ""

    # Two of the module's own bounds disagree; such a row never updates a
    # dataset.  bounds_inconsistent is True iff consistency_errors is non-empty.
    bounds_inconsistent: bool = False
    consistency_errors: List[str] = field(default_factory=list)

    # HFK invariants (knots only)
    hfk_computed: bool = False
    hfk_seifert_genus: Optional[int] = None
    hfk_tau: Optional[int] = None
    hfk_nu: Optional[int] = None
    hfk_nu_mirror: Optional[int] = None
    hfk_epsilon: Optional[int] = None
    hfk_fibered: Optional[bool] = None

    # Unknotting-number bounds that do not factor through the slice genus,
    # per orientation; the dataset path maximises them over the class.
    # u >= (|\sigma_\omega| + |\eta_\omega - \ell + 1|)/2  (Conway Thm 1)
    unknotting_lower_lt: Optional[int] = None
    # u >= |\nu_s| = |s + \ell - 1|/2                     (slice-torus)
    unknotting_lower_slice_torus: Optional[int] = None
    # u >= \abslk{L} + \sum_i u(K_i)                      (components)
    unknotting_lower_components: Optional[int] = None
    # u >= |\nu_s(L) - \sum_i \nu_s(K_i)|                (splitting number)
    unknotting_lower_splitting: Optional[int] = None
    # u <= \abslk{L} + \sum_i \bar u(K_i)                (simply-linked)
    unknotting_upper_simply_linked: Optional[int] = None
    # max of the four lower rows above, 0 when none applies
    best_unknotting_lower_direct: int = 0
    # L is provably not the unlink, hence u >= 1; see _unlink_certificates.
    nontrivial_certified: bool = False
    nontrivial_certificates: List[str] = field(default_factory=list)

    # The ComponentBounds fields, one list per field; None when the pass
    # was skipped.
    component_genus_lower: Optional[List[int]] = None
    component_genus_upper: Optional[List[Optional[int]]] = None
    component_u_lower: Optional[List[int]] = None
    component_u_upper: Optional[List[Optional[int]]] = None
    component_s: Optional[List[Optional[int]]] = None
    component_names: Optional[List[Optional[str]]] = None
    component_identification: Optional[List[Optional[str]]] = None
    component_genus_lower_source: Optional[List[str]] = None
    component_genus_upper_source: Optional[List[str]] = None
    component_u_lower_source: Optional[List[str]] = None
    component_u_upper_source: Optional[List[str]] = None
    component_s_source: Optional[List[Optional[str]]] = None

    # Derived
    best_slice_lower: int = 0
    best_strong_slice_lower: int = 0

    # Per-phase timing (populated when called from compute_bounds_for_dataset)
    phase_times: Optional[Dict[str, float]] = None


# The per-component lists of BoundsResult and the ComponentBounds field each
# one collects.
_COMPONENT_COLUMNS = (
    ('component_genus_lower', 'genus_lower'),
    ('component_genus_upper', 'genus_upper'),
    ('component_u_lower', 'u_lower'),
    ('component_u_upper', 'u_upper'),
    ('component_s', 's'),
    ('component_names', 'name'),
    ('component_identification', 'identification'),
    ('component_genus_lower_source', 'genus_lower_source'),
    ('component_genus_upper_source', 'genus_upper_source'),
    ('component_u_lower_source', 'u_lower_source'),
    ('component_u_upper_source', 'u_upper_source'),
    ('component_s_source', 's_source'),
)


# =============================================================================
# SINGLE LINK FUNCTIONS
# =============================================================================

def compute_bounds(
    link: Link,
    use_knotjob: bool = False,
    knotjob_timeout: int = 15,
    precomputed_s_invariant: Optional[int] = None,
    mt_primes: Optional[List[int]] = None,
    strong_mt_primes: Optional[List[int]] = None,
    use_component_bounds: bool = True,
    use_component_identification: bool = True
) -> BoundsResult:
    r"""
    Computes every applicable bound on the slice and strong slice
    genus and on the unknotting number of a single link.  Without
    KnotJob the s-invariant is still filled in on positive and on
    non-split alternating diagrams.

    Args:
        link: a spherogram Link.
        use_knotjob: compute the s-invariant via KnotJob (needs
            Java).  Default False.
        knotjob_timeout: timeout in seconds for the KnotJob call.
        precomputed_s_invariant: an s-invariant computed in batch.
        mt_primes: orders a for the weak Murasugi-Tristram bound,
            each a prime power greater than 1; \omega = \zeta_a.
            Defaults to [2, 3].
        strong_mt_primes: primes p for the strong Murasugi-Tristram
            bound; \omega_p = \zeta_p^{(p-1)/2}.  Defaults to [2, 3].
        use_component_bounds: run the per-component pass on a
            multi-component link, which feeds the component,
            splitting-number and simply-linked bounds.  Default
            True; the pass never calls KnotJob.
        use_component_identification: with use_component_bounds,
            merge in the best known g_4 and u of each identified
            component (knot_identification.py).  Default True.

    Returns:
        BoundsResult with the bounds and the diagram properties.
    """
    result = BoundsResult()
    timings: Dict[str, float] = {}

    # --- Phase 1: Link properties ---
    result.num_components = len(link.link_components) + link.unlinked_unknot_components
    result.is_knot = result.num_components == 1
    result.is_positive = _is_positive_diagram(link)
    result.is_negative = _is_negative_diagram(link)
    result.is_split = _is_split(link)
    result.is_algebraically_split = _is_algebraically_split(link)
    result.is_simply_linked = _is_simply_linked(link)
    result.abs_linking_number = _abs_linking_number(link)

    ell = result.num_components
    ell_s = _num_split_components(link)

    # --- Phase 2: Sage-based diagram invariants ---
    t0 = time()
    sage_link = _to_sage_link(link)
    result.is_alternating = bool(sage_link.is_alternating())
    result.num_crossings = len(link.crossings)
    result.num_seifert_circles = len(sage_link.seifert_circles()) + link.unlinked_unknot_components
    result.writhe = int(sage_link.writhe())

    c = result.num_crossings
    o = result.num_seifert_circles

    # --- Phase 3: Signature and nullity via Sage ---
    V = _sage_seifert_matrix_from_sage_link(sage_link)
    sig, nul = _sage_signature_and_nullity_from_matrix(V)
    result.signature = sig
    result.nullity = nul
    timings['sage'] = time() - t0

    # --- Phase 3b: Fox-Milnor determinant obstruction (knots only) ---
    # det(K) is a perfect square for a slice knot, so a non-square
    # determinant proves g_4(K) >= 1.  The full test waits for Phase 10b.
    t0 = time()
    if result.is_knot:
        det_k = _knot_determinant_from_seifert(V)
        result.determinant = det_k
        if det_k is not None:
            result.determinant_is_square = _is_perfect_square(det_k)
            result.slice_lower_fox_milnor_det = 0 if result.determinant_is_square else 1
    timings['fox_milnor_det'] = time() - t0

    # --- Phase 4: Rasmussen s-invariant ---
    s_inv = None
    s_from_positive = False
    s_from_knotjob = False
    s_from_alternating = False

    # A solver only ever sees link.PD_code(), which omits the crossingless
    # components, so its answer belongs to a different link and is refused.
    solver_usable = link.unlinked_unknot_components == 0

    if precomputed_s_invariant is not None and solver_usable:
        s_inv = precomputed_s_invariant
        s_from_knotjob = True

    if s_inv is None and use_knotjob and KNOTJOB_AVAILABLE and solver_usable:
        try:
            s_inv = compute_s_invariant(link, timeout=knotjob_timeout)
            s_from_knotjob = (s_inv is not None)
        except Exception:
            pass

    if s_inv is None:
        if result.is_positive:
            s_inv = _s_invariant_positive(c, o)
            s_from_positive = True
        elif _is_alternating_diagram(link) and not result.is_split:
            # s(L) = -\sigma(L) for a non-split alternating link
            # ([rasmussen2010slice, Theorem 3] for knots,
            # [cavallo2015links] for links); a connected alternating
            # diagram certifies both hypotheses.
            s_inv = -sig
            s_from_alternating = True

    result.s_invariant = s_inv
    result.s_invariant_from_positive = s_from_positive
    result.s_invariant_from_knotjob = s_from_knotjob
    result.s_invariant_from_alternating_signature = s_from_alternating

    # --- Phase 5: slice-torus invariants ---
    # \nu_s = (s + \ell - 1)/2
    if s_inv is not None:
        result.nu_s = (s_inv + ell - 1) / 2.0
    else:
        result.nu_s = None

    # --- Phase 6: Slice genus bounds ---
    result.slice_upper_seifert = _seifert_genus_from_diagram(c, o, ell, ell_s)
    result.slice_lower_signature = _signature_lower_bound(sig, ell)

    # Every bound reading a nullity is stated for Conway's \eta_L(\omega),
    # which the raw matrix nullity equals exactly when the Seifert surface S
    # underlying V is connected; those bounds are claimed only then.
    connected_surface = _has_connected_seifert_surface(V, ell)

    # Weak Murasugi-Tristram bound, knots included (Kauffman's Theorem 4.1
    # has no \ell >= 2 hypothesis).  Phases 7 and 11 reuse the p \in {2, 3}
    # entries of this table.
    t0 = time()
    mt_sig_null: Dict[int, tuple] = _omega_sig_null_table(V, mt_primes)
    if connected_surface:
        result.slice_lower_mt = _murasugi_tristram_bound_from_table(
            mt_sig_null, ell)
    timings['mt'] = time() - t0

    # \nu_s bound: g_4(L) >= max{-\nu_s, \nu_s - \ell + 1}
    if s_inv is not None:
        # Bound 1: g_4 >= ceil(-\nu_s)
        bound1 = ceil(-(s_inv + ell - 1) / 2)
        # Bound 2: g_4 >= ceil(\nu_s - \ell + 1)
        bound2 = ceil((s_inv - ell + 1) / 2)
        result.slice_lower_nu_s = max(0, max(bound1, bound2))
        result.nu_s_bound1 = bound1
        result.nu_s_bound2 = bound2

    # Exact slice genus of a positive or negative non-split diagram,
    # g_4 = g_3 = (c(D) - O(D) + 2 - \ell)/2 [nakamura2000positive,
    # Theorem 1.1]; a negative diagram is the mirror of a positive one.
    if (result.is_positive or result.is_negative) and not result.is_split:
        result.slice_exact = result.slice_upper_seifert
        result.slice_genus_is_exact = True

    # --- Phase 6b: component sub-diagram bounds (multi-component only) ---
    # One pass, four consumers: the strong-slice component bound just below,
    # and the unknotting component, splitting-number and simply-linked
    # bounds of Phase 11.
    t0 = time()
    components: Optional[List[ComponentBounds]] = None
    # ell counts the crossingless components too: a knot together with one of
    # them is a two-component link and does have components to read.
    if use_component_bounds and not result.is_knot and ell > 1:
        components = _component_bounds(
            link, mt_primes, use_identification=use_component_identification)
        if components is not None:
            for column, attr in _COMPONENT_COLUMNS:
                setattr(result, column, [getattr(cb, attr) for cb in components])
            # A component whose pipeline bound contradicted its known value
            # was dropped to the zero record (_component_bounds); the
            # disagreement is still surfaced here, on the link.
            for cb in components:
                if cb.contradiction:
                    result.consistency_errors.append(cb.contradiction)
    timings['components'] = time() - t0

    # --- Phase 7: Strong slice genus bounds ---
    # Declared here so that Phase 11 can reuse the p >= 5 entries.
    tristram_sig_null: Dict[int, tuple] = {}
    if not result.is_knot and result.is_algebraically_split:
        # All slice lower bounds apply to strong slice
        best_slice_lower = max(result.slice_lower_signature,
                               result.slice_lower_mt)
        result.strong_slice_lower_from_slice = best_slice_lower

        # Pseudo-thin bound, (|s| + \ell - 1)/2, certified by a connected
        # alternating diagram (the PD-level test of Phase 4).  `not is_split`
        # is a hypothesis of Cavallo's theorem.
        if (s_inv is not None and not result.is_split
                and _is_alternating_diagram(link)):
            result.strong_slice_lower_pseudo_thin = _pseudo_thin_strong_bound(
                s_inv, ell)
            # The same number, stored under the strong-signature column too.
            result.strong_slice_lower_signature = (
                _pseudo_thin_strong_signature_bound(sig, ell))

        # Strong Murasugi-Tristram (m = \ell) at Tristram's \omega_p,
        # claimed only on a connected Seifert surface.
        if connected_surface:
            t0 = time()
            tristram_sig_null = _tristram_sig_null_table(
                V, strong_mt_primes, reuse=mt_sig_null)
            result.strong_slice_lower_mt = (
                _strong_murasugi_tristram_bound_from_table(tristram_sig_null, ell))
            timings['strong_mt'] = time() - t0

        # Slice-torus bound |\nu_s| <= g_4^*.
        if s_inv is not None:
            result.strong_slice_lower_nu_s = _slice_torus_strong_bound(s_inv, ell)

        # Component bound \sum_i g_4(K_i) <= g_4^*(L), summing the lower
        # bounds of Phase 6b.
        if components is not None:
            result.strong_slice_lower_components = sum(
                cb.genus_lower for cb in components)

        # Strong slice obstruction; we are already inside the algebraically
        # split branch, so only \sigma and s remain to be checked.
        obstructed = False
        reason = ""
        if sig != 0:
            obstructed = True
            reason = f"signature σ = {sig} ≠ 0"
        elif s_inv is not None and s_inv != 1 - ell:
            obstructed = True
            reason = f"s-invariant s = {s_inv} ≠ {1 - ell}"
        result.strong_slice_obstructed = obstructed
        result.strong_slice_obstruction_reason = reason
        result.strong_slice_lower_obstruction = 1 if obstructed else 0

    # --- Phase 8: Slice obstruction check ---
    slice_obstructed = False
    if s_inv is not None and abs(s_inv) > ell - 1:
        slice_obstructed = True
    elif result.is_alternating and not result.is_split and abs(sig) > ell - 1:
        slice_obstructed = True
    elif result.slice_lower_fox_milnor_det:
        # Fox-Milnor at t = -1; the full test sets the flag in Phase 10b.
        slice_obstructed = True
    # A reporting-only flag; \nu (Phase 9) and the full Fox-Milnor test
    # (Phase 10b) also set it, Murasugi-Tristram does not.
    result.slice_obstructed = slice_obstructed

    # --- Phase 9: HFK invariants (knots only) ---
    t0 = time()
    if result.is_knot:
        hfk = _compute_hfk(link)
        if hfk is not None:
            result.hfk_computed = True
            result.hfk_seifert_genus = hfk.get('seifert_genus')
            result.hfk_tau = hfk.get('tau')
            result.hfk_nu = hfk.get('nu')
            result.hfk_epsilon = hfk.get('epsilon')
            result.hfk_fibered = hfk.get('fibered')
            if result.hfk_tau is not None:
                result.slice_lower_tau = abs(result.hfk_tau)
            # \nu(K) <= g_4(K) [hom2016nuplus, Prop. 2.3-2.4], and g_4 is
            # mirror-invariant, so \nu of the mirror bounds it too.
            try:
                hfk_m = _compute_hfk(link.mirror())
                result.hfk_nu_mirror = hfk_m.get('nu') if hfk_m else None
            except Exception:
                result.hfk_nu_mirror = None
            nu_vals = [v for v in (result.hfk_nu, result.hfk_nu_mirror)
                       if v is not None]
            if nu_vals:
                result.slice_lower_nu_hfk = max(0, max(int(v) for v in nu_vals))
                if result.slice_lower_nu_hfk > 0:
                    # \nu(K) >= 1 proves g_4(K) >= 1.
                    result.slice_obstructed = True
            if (result.hfk_seifert_genus is not None
                    and result.hfk_seifert_genus < result.slice_upper_seifert):
                result.slice_upper_seifert = result.hfk_seifert_genus
    timings['hfk'] = time() - t0

    # --- Phase 10: Best bounds ---
    best_lower = max(result.slice_lower_signature, result.slice_lower_mt)
    if result.slice_lower_nu_s is not None:
        best_lower = max(best_lower, result.slice_lower_nu_s)
    if result.slice_lower_tau is not None:
        best_lower = max(best_lower, result.slice_lower_tau)
    if result.slice_lower_nu_hfk is not None:
        best_lower = max(best_lower, result.slice_lower_nu_hfk)
    if result.slice_lower_fox_milnor_det is not None:
        best_lower = max(best_lower, result.slice_lower_fox_milnor_det)
    if result.slice_exact is not None:
        # The exact value wins; a lower bound above it is recorded as an
        # inconsistency.
        if best_lower > result.slice_exact:
            result.consistency_errors.append(
                f"a slice lower bound of {best_lower} exceeds the exact genus "
                f"{result.slice_exact} of a positive/negative non-split diagram")
        best_lower = result.slice_exact

    # --- Phase 10b: full Fox-Milnor test, last resort (knots only) ---
    # The polynomial determinant is expensive, so the test runs only when
    # every other bound left the lower bound at 0 and the genus is not exact.
    t0 = time()
    if result.is_knot and best_lower == 0 and result.slice_exact is None:
        delta = _alexander_polynomial_from_seifert(V)
        result.fox_milnor_satisfied = _fox_milnor_satisfied(delta)
        if result.fox_milnor_satisfied is False:
            result.slice_lower_fox_milnor_full = 1
            result.slice_obstructed = True
            best_lower = 1
        elif result.fox_milnor_satisfied is True:
            result.slice_lower_fox_milnor_full = 0
    timings['fox_milnor_full'] = time() - t0

    result.best_slice_lower = best_lower

    # Best strong slice lower
    if not result.is_knot and result.is_algebraically_split:
        best_strong = result.strong_slice_lower_from_slice
        if result.strong_slice_lower_signature is not None:
            best_strong = max(best_strong, result.strong_slice_lower_signature)
        if result.strong_slice_lower_pseudo_thin is not None:
            best_strong = max(best_strong, result.strong_slice_lower_pseudo_thin)
        if result.strong_slice_lower_mt is not None:
            best_strong = max(best_strong, result.strong_slice_lower_mt)
        if result.strong_slice_lower_nu_s is not None:
            best_strong = max(best_strong, result.strong_slice_lower_nu_s)
        if result.strong_slice_lower_components is not None:
            best_strong = max(best_strong, result.strong_slice_lower_components)
        best_strong = max(best_strong, result.strong_slice_lower_obstruction)
        result.best_strong_slice_lower = best_strong

    # --- Phase 11: unknotting bounds that do not factor through g_4 ---
    # Per orientation; _add_unknotting_bounds_to_results maximises them
    # over the orientation class.
    direct = 0

    # Conway's Theorem 1 holds at every \omega != 1, so both tables are
    # legal inputs (the \omega_p keys are negated to keep them apart).
    # Claimed only on a connected Seifert surface.
    if connected_surface:
        conway_table = dict(mt_sig_null)
        conway_table.update({-p: v for p, v in tristram_sig_null.items()})
        if conway_table:
            result.unknotting_lower_lt = _conway_lt_unknotting_bound_from_table(
                conway_table, ell)
            direct = max(direct, result.unknotting_lower_lt)

    # Slice-torus: |\nu_s(L)| <= u(L), no hypothesis.
    if s_inv is not None:
        result.unknotting_lower_slice_torus = _slice_torus_unknotting_bound(
            s_inv, ell)
        direct = max(direct, result.unknotting_lower_slice_torus)

    if components is not None:
        # \abslk{L} + \sum_i u(K_i) <= u(L).
        result.unknotting_lower_components = (
            result.abs_linking_number + sum(cb.u_lower for cb in components))
        direct = max(direct, result.unknotting_lower_components)

        # |\nu_s(L) - \sum_i \nu_s(K_i)| <= \widetilde{sp}(L) <= u(L).
        splitting = _splitting_number_unknotting_bound(
            s_inv, [cb.s for cb in components], ell)
        if splitting is not None:
            result.unknotting_lower_splitting = splitting
            direct = max(direct, splitting)

        # Simply-linked: Collari's equality turns per-component upper
        # bounds into one for the link.
        if result.is_simply_linked and all(cb.u_upper is not None
                                           for cb in components):
            result.unknotting_upper_simply_linked = (
                result.abs_linking_number
                + sum(int(cb.u_upper) for cb in components))

    result.best_unknotting_lower_direct = direct

    # Is L provably not the unlink?  The nullity and det are offered only on
    # a connected Seifert surface, where they are Conway's \eta and det(L).
    eta_conway = (mt_sig_null[2][1]
                  if connected_surface and 2 in mt_sig_null else None)
    det_for_cert = result.determinant if connected_surface else None
    result.nontrivial_certificates = _unlink_certificates(
        sig, eta_conway, s_inv, det_for_cert, result.hfk_seifert_genus, ell)
    result.nontrivial_certified = bool(result.nontrivial_certificates)

    # --- Phase 11b: the strong slice genus of this link, by topology ---
    # Phase 7 runs only for an algebraically split multi-component link, so
    # best_strong_slice_lower is 0 outside it and is not the answer there.
    if result.num_crossings == 0 and result.num_components > 0:
        # The unlink bounds disjoint disks, one for each component.
        result.strong_slice_genus_lower = 0.0
        result.strong_slice_genus_upper = 0.0
    elif result.is_knot:
        # A slice surface of a knot is a strong slice surface, so the
        # two genera coincide.
        result.strong_slice_genus_lower = float(result.best_slice_lower)
        result.strong_slice_genus_upper = float(result.slice_upper_seifert)
        result.strong_slice_obstructed = result.slice_obstructed
        if result.strong_slice_obstructed and not result.strong_slice_obstruction_reason:
            result.strong_slice_obstruction_reason = (
                "g_4*(K) = g_4(K) for a knot, and g_4(K) >= 1")
    elif not result.is_algebraically_split:
        # Some pair of components has non-zero linking number.  Linking numbers
        # are invariants of strong concordance, so no strong slice surface
        # exists at any genus.
        result.strong_slice_genus_lower = math.inf
        result.strong_slice_genus_upper = math.inf
        result.strong_slice_genus_is_infinite = True
        result.strong_slice_obstructed = True
        result.strong_slice_obstruction_reason = (
            "some pairwise linking number is non-zero, so no strong slice "
            "surface exists and g_4* = infinity")
    else:
        result.strong_slice_genus_lower = float(result.best_strong_slice_lower)
        # No finite upper bound on g_4^* is proved here for the remaining
        # algebraically split links: the Seifert surface of the diagram is
        # not a strong slice surface.
        result.strong_slice_genus_upper = math.inf

    # --- Phase 12: internal consistency ---
    # Each check compares two independently derived bounds, so a failure
    # means a violated hypothesis upstream.
    if result.best_slice_lower > result.slice_upper_seifert:
        result.consistency_errors.append(
            f"slice lower bound {result.best_slice_lower} exceeds the upper "
            f"bound {result.slice_upper_seifert}")
    if (result.best_strong_slice_lower > 0
            and result.best_strong_slice_lower < result.best_slice_lower):
        # g_4^*(L) >= g_4(L): a strong slice surface is a slice surface.
        result.consistency_errors.append(
            f"strong slice lower bound {result.best_strong_slice_lower} is "
            f"below the slice lower bound {result.best_slice_lower}")
    u_upper = result.num_crossings // 2
    if result.unknotting_upper_simply_linked is not None:
        u_upper = min(u_upper, result.unknotting_upper_simply_linked)
    if result.best_unknotting_lower_direct > u_upper:
        result.consistency_errors.append(
            f"unknotting lower bound {result.best_unknotting_lower_direct} "
            f"exceeds the upper bound {u_upper}")
    result.bounds_inconsistent = bool(result.consistency_errors)

    result.phase_times = timings
    return result


# =============================================================================
# PUBLIC API: single-link convenience functions
# =============================================================================

def signature(link: Link) -> int:
    """
    The Murasugi signature of the link, by Sage exact arithmetic.

    Reference: [murasugi1965numerical, Theorem 3.1]
    """
    sig, _ = _sage_signature_and_nullity(link)
    return sig


def signature_and_nullity(link: Link) -> tuple:
    """
    (signature, nullity) of the link, by Sage exact arithmetic.  The nullity is
    the raw matrix one, which is Conway's link nullity only when the Seifert
    surface is connected.

    Reference: [murasugi1965numerical, Theorem 3.1],
               [conway2019signature, Definition 1]
    """
    return _sage_signature_and_nullity(link)


def seifert_genus_from_diagram(link: Link) -> int:
    """Returns the Seifert genus of the diagram (upper bound on g_3)."""
    sage_link = _to_sage_link(link)
    ell = len(link.link_components) + link.unlinked_unknot_components
    c = len(link.crossings)
    o = len(sage_link.seifert_circles()) + link.unlinked_unknot_components
    return _seifert_genus_from_diagram(c, o, ell, _num_split_components(link))


def knot_determinant(link: Link) -> Optional[int]:
    r"""
    det(K) = |\Delta_K(-1)|, from the Seifert matrix of the diagram.

    Returns None for a link (Fox-Milnor is a knot statement) or for a
    crossingless diagram.
    """
    if not (len(link.link_components) == 1 and link.unlinked_unknot_components == 0):
        return None
    return _knot_determinant_from_seifert(
        _sage_seifert_matrix_from_sage_link(_to_sage_link(link)))


def fox_milnor_obstruction(link: Link) -> Optional[bool]:
    r"""
    True iff the Fox-Milnor condition fails, i.e. K is not slice and
    g_4(K) >= 1: the opposite polarity to _fox_milnor_satisfied.

    Runs the determinant test first and factors \Delta_K only if that is
    inconclusive.  Returns None when no claim can be made: a link, a
    crossingless diagram, or a Seifert matrix that is not a knot's.

    Reference: [foxmilnor1966cobordism, Theorem 2]
    """
    if not (len(link.link_components) == 1 and link.unlinked_unknot_components == 0):
        return None
    V = _sage_seifert_matrix_from_sage_link(_to_sage_link(link))
    det_k = _knot_determinant_from_seifert(V)
    if det_k is not None and not _is_perfect_square(det_k):
        return True
    satisfied = _fox_milnor_satisfied(_alexander_polynomial_from_seifert(V))
    if satisfied is None:
        return None
    return not satisfied


is_positive_diagram = _is_positive_diagram
is_alternating_diagram = _is_alternating_diagram
is_split = _is_split
is_algebraically_split = _is_algebraically_split


def writhe(link: Link) -> int:
    """Returns the writhe of the diagram."""
    sage_link = _to_sage_link(link)
    return int(sage_link.writhe())


def signature_lower_bound(link: Link) -> int:
    """
    Signature-based lower bound on the slice genus; see
    _signature_lower_bound.

    Reference: [murasugi1965numerical, Theorem 9.1]
    """
    sig, _ = _sage_signature_and_nullity(link)
    ell = len(link.link_components) + link.unlinked_unknot_components
    return _signature_lower_bound(sig, ell)


def murasugi_tristram_bound(link: Link, mt_primes: Optional[List[int]] = None) -> int:
    r"""
    Lower bound on the slice genus from the Murasugi-Tristram inequality: for
    \omega a prime-power root of unity,
        g_4 >= \lceil (|\sigma_L(\omega)| + \eta_L(\omega) - (\ell - 1))/2 \rceil
    Returns 0 (not claimed) when the Seifert surface is disconnected.

    Reference: [tristram1969cobordism, Theorem 2.27],
               [kauffman1978signature, Theorem 4.1]
    """
    sage_link = _to_sage_link(link)
    V = _sage_seifert_matrix_from_sage_link(sage_link)
    ell = len(link.link_components) + link.unlinked_unknot_components
    return _murasugi_tristram_bound_from_matrix(V, ell, mt_primes)


def strong_murasugi_tristram_bound(link: Link,
                                   strong_mt_primes: Optional[List[int]] = None) -> int:
    r"""
    Lower bound on the strong slice genus from the Murasugi-Tristram
    inequality for a strong slice surface (m = \ell):
        g_4^* >= \lceil (|\sigma_L(\omega)| + |\eta_L(\omega) + 1 - \ell|)/2 \rceil
    Returns 0 (not claimed) when the Seifert surface is disconnected.

    Reference: [tristram1969cobordism, Theorem 2.27]
    """
    sage_link = _to_sage_link(link)
    V = _sage_seifert_matrix_from_sage_link(sage_link)
    ell = len(link.link_components) + link.unlinked_unknot_components
    if not _has_connected_seifert_surface(V, ell):
        return 0
    return _strong_murasugi_tristram_bound_from_table(
        _tristram_sig_null_table(V, strong_mt_primes), ell)


def slice_genus_range(link: Link) -> tuple:
    """Returns (lower, upper) bounds on slice genus."""
    bounds = compute_bounds(link)
    return (bounds.best_slice_lower, bounds.slice_upper_seifert)


def best_slice_lower_bound(link: Link) -> int:
    """
    The highest lower bound on the slice genus over every bound the module
    computes; see compute_bounds for the individual references.
    """
    return compute_bounds(link).best_slice_lower


def best_strong_slice_lower_bound(link: Link) -> float:
    r"""
    Best lower bound on the strong slice genus: g_4(K) for a knot,
    math.inf for a link with a non-zero pairwise linking number,
    and the aggregate of the Phase-7 bounds for an algebraically
    split link.

    Returns:
        The bound, as a float.
    """
    return compute_bounds(link).strong_slice_genus_lower


def is_slice_obstructed(link: Link) -> bool:
    """Returns True if the link cannot be slice (g_4 >= 1)."""
    return compute_bounds(link).slice_obstructed


def is_strongly_slice_obstructed(link: Link) -> bool:
    """
    True if the link cannot be strongly slice, i.e. g_4^* >= 1.  Covers a knot
    (through g_4^* = g_4) and a link with a non-zero pairwise linking number
    (where g_4^* is infinite), not only the algebraically split case.
    """
    return compute_bounds(link).strong_slice_obstructed


# =============================================================================
# DATASET FUNCTIONS
# =============================================================================

@dataclass
class DatasetBoundsSummary:
    """Summary statistics from computing bounds for a dataset."""
    total_links: int = 0
    knot_count: int = 0
    positive_count: int = 0
    alternating_count: int = 0
    split_count: int = 0
    alg_split_count: int = 0

    # s-invariant statistics
    s_invariant_computed_count: int = 0
    s_invariant_from_positive_count: int = 0
    s_invariant_from_knotjob_count: int = 0
    s_invariant_from_alternating_signature_count: int = 0

    # Fox-Milnor (knots only)
    fox_milnor_det_checked_count: int = 0       # determinant test evaluated
    fox_milnor_det_obstruction_count: int = 0   # det not a perfect square
    fox_milnor_full_checked_count: int = 0      # full test actually run
    fox_milnor_full_obstruction_count: int = 0  # full test failed => not slice

    # HFK statistics (knots only)
    hfk_computed_count: int = 0
    hfk_fibered_count: int = 0

    # Bounds statistics
    slice_known_count: int = 0
    slice_consistent_count: int = 0
    slice_tight_count: int = 0
    strong_slice_known_count: int = 0
    strong_slice_consistent_count: int = 0
    unknotting_known_count: int = 0
    unknotting_consistent_count: int = 0
    # Rows whose strong-slice lower bound was raised by another orientation of
    # the same unoriented link (g_4^* being orientation-invariant).
    strong_slice_orientation_improved_count: int = 0

    # Component and orientation-class bounds.
    simply_linked_count: int = 0          # diagrams certifying simply-linked
    component_bounds_count: int = 0       # links with a full component pass
    nu_hfk_improved_count: int = 0        # \nu beat |\tau| on a knot
    strong_components_best_count: int = 0  # \sum g_4(K_i) is the best strong bound
    strong_nu_s_best_count: int = 0        # |\nu_s| is the best strong bound
    unknotting_direct_best_count: int = 0   # a Phase-11 bound is the best u lower
    # knot_identification.identify() over every component of every link with
    # a component pass (not links): how many were settled, left ambiguous, or
    # had a known value strictly sharpen the pipeline's own bound.
    component_identified_count: int = 0
    component_ambiguous_count: int = 0
    component_known_improved_count: int = 0

    # Positive-link unknotting (Collari Thm. 1.9, see positive_unknotting.py)
    positive_unknotting_certified_count: int = 0
    positive_unknotting_exact_count: int = 0
    positive_unknotting_improved_count: int = 0

    # Best bound distribution
    best_bound_counts: Dict[str, int] = field(default_factory=dict)

    # Inconsistent indices (for debugging)
    slice_inconsistent_indices: List[int] = field(default_factory=list)
    strong_slice_inconsistent_indices: List[int] = field(default_factory=list)
    unknotting_inconsistent_indices: List[int] = field(default_factory=list)

    # Internal inconsistencies (best_lower > best_upper, indicates bug)
    bounds_inconsistent_count: int = 0
    bounds_inconsistent_indices: List[int] = field(default_factory=list)

    # Detailed inconsistency info
    slice_inconsistent_details: List[Dict[str, Any]] = field(default_factory=list)
    strong_slice_inconsistent_details: List[Dict[str, Any]] = field(default_factory=list)
    unknotting_inconsistent_details: List[Dict[str, Any]] = field(default_factory=list)


def _exact_source(bounds: BoundsResult) -> str:
    """
    Source label of an exact slice genus: the two theorems are
    Nakamura's for a positive diagram and its mirror image for a
    negative one.  Read by both the lower and the upper resolver,
    which must agree.
    """
    return _EXACT_POSITIVE_SOURCE if bounds.is_positive else _EXACT_NEGATIVE_SOURCE


def _get_best_bound_names(bounds: BoundsResult) -> List[str]:
    """Returns all bound names that jointly achieve the best lower bound."""
    best = bounds.best_slice_lower
    names = []

    if bounds.slice_exact is not None:
        return [_exact_source(bounds)]
    if bounds.slice_lower_tau is not None and bounds.slice_lower_tau == best:
        names.append('tau_hfk')
    if (bounds.slice_lower_nu_hfk is not None
            and bounds.slice_lower_nu_hfk == best):
        names.append('nu_hfk')
    # A Fox-Milnor value of 0 is no claim, so it is never a source.
    if (bounds.slice_lower_fox_milnor_det is not None
            and bounds.slice_lower_fox_milnor_det > 0
            and bounds.slice_lower_fox_milnor_det == best):
        names.append(_FOX_MILNOR_DET_SOURCE)
    if (bounds.slice_lower_fox_milnor_full is not None
            and bounds.slice_lower_fox_milnor_full > 0
            and bounds.slice_lower_fox_milnor_full == best):
        names.append(_FOX_MILNOR_FULL_SOURCE)
    if bounds.slice_lower_nu_s is not None and bounds.slice_lower_nu_s == best:
        if bounds.s_invariant_from_knotjob:
            names.append('nu_s_knotjob')
        elif bounds.s_invariant_from_alternating_signature:
            names.append('nu_s_alternating_signature')
        else:
            names.append('nu_s_positive')
    if bounds.slice_lower_mt == best:
        names.append('murasugi_tristram')
    if bounds.slice_lower_signature == best:
        names.append('signature')
    return names if names else ['unknown']


# =============================================================================
# BEST-BOUND VALUE + SOURCE RESOLUTION
#
# Each helper returns (value, source) for one bound, merging the theoretical
# bounds of a BoundsResult with the value the dataset already knows.  Known
# wins ties: when the known value equals or dominates the best theoretical
# bound, the source is the corresponding "known_*" label.  Every value column
# is emitted next to a sibling "_source" one.
# =============================================================================

def _slice_lower_with_source(bounds: BoundsResult, known) -> tuple:
    """
    Best slice-genus lower bound and its source: the theoretical
    best_slice_lower merged with the known g_4 floor.  Known wins ties, so
    min(known) >= theoretical gives the source known_slice_genus; otherwise
    the source is the joined bound name(s) from _get_best_bound_names.
    """
    theo = bounds.best_slice_lower
    known_min = min(known) if known else None
    if known_min is not None and known_min >= theo:
        return known_min, 'known_slice_genus'
    return theo, " + ".join(_get_best_bound_names(bounds))


def _slice_upper_with_source(bounds: BoundsResult, known) -> tuple:
    """
    Best slice-genus upper bound and its source.  The theoretical upper bound
    is slice_upper_seifert, already tightened by HFK where available and equal
    to slice_exact on a positive or negative non-split link.  Known wins ties,
    so max(known) <= theoretical gives the source known_slice_genus.
    """
    theo = bounds.slice_upper_seifert
    known_max = max(known) if known else None
    if known_max is not None and known_max <= theo:
        return known_max, 'known_slice_genus'
    if bounds.slice_exact is not None and bounds.slice_exact == theo:
        src = _exact_source(bounds)
    elif (bounds.hfk_seifert_genus is not None
          and bounds.hfk_seifert_genus == theo):
        src = _COMPONENT_HFK_SEIFERT_GENUS_SOURCE
    else:
        src = _COMPONENT_SEIFERT_GENUS_SOURCE
    return theo, src


def _strong_slice_lower_with_source(bounds: BoundsResult, known, is_knot: bool,
                                    slice_lower_value: int,
                                    slice_lower_source: str) -> tuple:
    """
    Best strong-slice-genus lower bound and its source.  For a knot
    g_4^* = g_4, so the slice lower bound and its source are mirrored; for an
    algebraically split link the strong-slice bounds are merged, with
    known_strong_slice_genus winning ties.
    """
    if is_knot:
        # g_4*(K) = g_4(K); reuse the slice lower bound and its source verbatim.
        known_min = min(known) if known else None
        if known_min is not None and known_min >= slice_lower_value:
            return known_min, 'known_strong_slice_genus'
        return slice_lower_value, slice_lower_source

    if bounds.strong_slice_genus_is_infinite:
        # No strong slice surface exists, so the lower bound is infinite too;
        # the upper resolver already reported inf here.
        return math.inf, 'linking_number_obstruction'

    theo = bounds.best_strong_slice_lower
    known_min = min(known) if known else None
    if known_min is not None and known_min >= theo:
        return known_min, 'known_strong_slice_genus'

    # Identify which strong-slice sub-bound(s) achieve the best lower bound.
    names = []
    if bounds.strong_slice_lower_obstruction == theo and theo > 0:
        names.append('strong_obstruction')
    if bounds.strong_slice_lower_from_slice == theo:
        names.append('from_slice')
    if (bounds.strong_slice_lower_signature is not None
            and bounds.strong_slice_lower_signature == theo):
        names.append('strong_signature')
    if (bounds.strong_slice_lower_pseudo_thin is not None
            and bounds.strong_slice_lower_pseudo_thin == theo):
        names.append('strong_pseudo_thin')
    if (bounds.strong_slice_lower_components is not None
            and bounds.strong_slice_lower_components == theo and theo > 0):
        names.append('strong_components')
    if (bounds.strong_slice_lower_mt is not None
            and bounds.strong_slice_lower_mt == theo):
        names.append('strong_murasugi_tristram')
    if (bounds.strong_slice_lower_nu_s is not None
            and bounds.strong_slice_lower_nu_s == theo):
        names.append('strong_nu_s')
    return theo, " + ".join(names) if names else 'unknown'


def _strong_slice_upper_with_source(bounds: BoundsResult, known, is_knot: bool,
                                    is_alg_split: bool, num_components: int,
                                    slice_upper_value: int,
                                    slice_upper_source: str) -> tuple:
    """
    Best strong-slice-genus upper bound and its source.  The three topology
    cases mirror update_dataset_with_bounds: a knot mirrors the slice upper
    bound and its source; a multi-component link that is not algebraically
    split gets inf; and for an algebraically split one no theoretical upper
    bound is computed, so a finite value can come only from the known range.
    """
    if is_knot:
        known_max = max(known) if known else None
        if known_max is not None and known_max <= slice_upper_value:
            return known_max, 'known_strong_slice_genus'
        return slice_upper_value, slice_upper_source

    if num_components > 1 and not is_alg_split:
        return math.inf, 'linking_number_obstruction'

    # Algebraically split: only the known range can give a finite upper bound.
    known_max = max(known) if known else None
    if known_max is not None and not (isinstance(known_max, float)
                                      and math.isinf(known_max)):
        return known_max, 'known_strong_slice_genus'
    return bounds.strong_slice_upper, 'trivial'


# The lower bounds an inconsistency detail lists, as (label, field).
_SLICE_LOWER_BOUNDS = (
    ('signature', 'slice_lower_signature'),
    ('murasugi_tristram', 'slice_lower_mt'),
    ('nu_s', 'slice_lower_nu_s'),
    ('tau_hfk', 'slice_lower_tau'),
    ('nu_hfk', 'slice_lower_nu_hfk'),
    (_FOX_MILNOR_DET_SOURCE, 'slice_lower_fox_milnor_det'),
    (_FOX_MILNOR_FULL_SOURCE, 'slice_lower_fox_milnor_full'),
)
_STRONG_LOWER_BOUNDS = (
    ('signature', 'slice_lower_signature'),
    ('murasugi_tristram', 'slice_lower_mt'),
    ('strong_signature', 'strong_slice_lower_signature'),
    ('strong_murasugi_tristram', 'strong_slice_lower_mt'),
    ('strong_nu_s', 'strong_slice_lower_nu_s'),
    ('strong_pseudo_thin', 'strong_slice_lower_pseudo_thin'),
    ('strong_components', 'strong_slice_lower_components'),
)


def _lower_bounds_of(bounds: BoundsResult, fields) -> List[tuple]:
    """The (label, value) pairs of the fields of bounds that are set."""
    return [(label, getattr(bounds, attr)) for label, attr in fields
            if getattr(bounds, attr) is not None]


def _inconsistency_detail(name: Optional[str], index: int, known,
                          lower, upper, lower_bounds) -> Dict[str, Any]:
    """
    Record of a link whose computed bounds miss its known values:
    each lower bound, and the upper bound when there is one,
    sorted into those consistent with the known set and those
    not.

    Args:
        name: the link name, or None.
        index: its index in the dataset.
        known: the admissible values of the invariant.
        lower: the best lower bound.
        upper: the upper bound, or None when none is computed.
        lower_bounds: the individual lower bounds, (label, value).

    Returns:
        A dict with index, name, known, lower, upper,
        inconsistent_bounds and consistent_bounds.
    """
    known = list(known)
    checks = [(label, value, value > max(known)) for label, value in lower_bounds]
    if upper is not None:
        checks.append(('upper', upper, upper < min(known)))
    return {'index': index, 'name': name if name else f'link_{index}',
            'known': known, 'lower': lower, 'upper': upper,
            'inconsistent_bounds': [(l, v) for l, v, bad in checks if bad],
            'consistent_bounds': [(l, v) for l, v, bad in checks if not bad]}


# =============================================================================
# PARALLEL WORKER (module-level for spawn pickling)
# =============================================================================

def _compute_bounds_worker(args):
    r"""
    Worker for the parallel bounds computation, run in a ProcessPoolExecutor.
    Defined at module level so that 'spawn' can pickle it; the link travels as
    its PD code, not as a Link, for the same reason.

    Args:
        args: (pd_code, unlinked_unknots, signs, link_name, idx,
            precomputed_s, knotjob_timeout, mt_primes, strong_mt_primes,
            use_component_bounds, use_component_identification).

    Returns:
        (idx, link_name, bounds, error_str), the last two being exclusive.

    The crossingless components travel separately, PD_code()
    omitting them, and the crossing signs are re-checked on
    arrival: a mismatch is returned as an error.
    """
    (pd_code, unlinked_unknots, signs, link_name, idx, precomputed_s,
     knotjob_timeout, mt_primes, strong_mt_primes,
     use_component_bounds, use_component_identification) = args
    try:
        link = Link(pd_code) if pd_code else Link([])
        link.unlinked_unknot_components = unlinked_unknots
        rebuilt = [c.sign for c in link.crossings]
        if list(signs) != rebuilt:
            raise ValueError(
                f"worker rebuilt a different oriented link: crossing signs "
                f"{rebuilt} against {list(signs)}")
        bounds = compute_bounds(
            link,
            use_knotjob=False,  # s-invariants pre-computed; no KnotJob in worker
            knotjob_timeout=knotjob_timeout,
            precomputed_s_invariant=precomputed_s,
            mt_primes=mt_primes,
            strong_mt_primes=strong_mt_primes,
            use_component_bounds=use_component_bounds,
            use_component_identification=use_component_identification,
        )
        return idx, link_name, bounds, None
    except Exception as e:
        return idx, link_name, None, str(e)


# Columns of the results frame read straight off a BoundsResult, in order.
_RESULT_COLUMNS = (
    'is_knot', 'num_components', 'num_crossings', 'is_positive',
    'is_alternating', 'is_split', 'is_algebraically_split',
    'is_simply_linked', 'abs_linking_number', 'signature', 'nullity',
    'writhe', 's_invariant', 's_invariant_from_positive',
    's_invariant_from_knotjob', 's_invariant_from_alternating_signature',
    'nu_s', 'determinant', 'determinant_is_square', 'fox_milnor_satisfied',
    'slice_upper_seifert', 'slice_lower_signature', 'slice_lower_mt',
    'slice_lower_nu_s', 'slice_lower_tau', 'slice_lower_nu_hfk',
    'slice_lower_fox_milnor_det', 'slice_lower_fox_milnor_full',
    'slice_exact', 'best_slice_lower', 'best_strong_slice_lower',
    'strong_slice_upper', 'strong_slice_genus_lower',
    'strong_slice_genus_upper', 'strong_slice_genus_is_infinite',
    'strong_slice_obstructed', 'slice_obstructed', 'slice_genus_is_exact',
    'strong_slice_lower_mt', 'strong_slice_lower_nu_s',
    'strong_slice_lower_signature', 'strong_slice_lower_pseudo_thin',
    'strong_slice_lower_components', 'strong_slice_lower_from_slice',
    'strong_slice_lower_obstruction', 'unknotting_lower_lt',
    'unknotting_lower_slice_torus', 'unknotting_lower_components',
    'unknotting_lower_splitting', 'unknotting_upper_simply_linked',
    'best_unknotting_lower_direct', 'nontrivial_certified')

# The HFK columns of the results frame.
_HFK_COLUMNS = ('hfk_computed', 'hfk_tau', 'hfk_nu', 'hfk_nu_mirror',
                'hfk_seifert_genus', 'hfk_epsilon', 'hfk_fibered')


def _precompute_s_invariants(dataset, knotjob_timeout: int,
                             knotjob_num_workers: int,
                             verbose: bool) -> Dict[str, Optional[int]]:
    """
    The s-invariants of the links of dataset without a closed
    form, from KnotJob; positive and non-split alternating
    diagrams are skipped.  Sequentially the links go to KnotJob
    in chunks of 50; with several workers in one call spread over
    them.

    Args:
        dataset: the Dataset.
        knotjob_timeout: per-link timeout in seconds.
        knotjob_num_workers: parallel Java processes.
        verbose: print progress.

    Returns:
        {link name: s or None}; empty when the batch fails.
    """
    links, names, skipped = [], [], 0
    for i in range(len(dataset)):
        link, name = dataset[i][0], dataset[i][-1]
        try:
            closed_form = (_is_positive_diagram(link)
                           or (_is_alternating_diagram(link)
                               and not _is_split(link)))
        except Exception:
            closed_form = False
        if closed_form:
            skipped += 1
            continue
        links.append(link)
        names.append(name if name else f"link_{i}")
    if verbose and skipped:
        print(f"  KnotJob pre-pass: {len(links)}/{len(dataset)} links "
              f"(skipped {skipped} with a closed form for s)", flush=True)

    workers = max(1, knotjob_num_workers)
    chunk = max(1, len(links) if workers > 1 else 50)
    t0 = time()
    out: Dict[str, Optional[int]] = {}
    try:
        for start in range(0, len(links), chunk):
            part = slice(start, start + chunk)
            out.update(compute_s_invariants_batch(
                links=links[part], names=names[part], characteristic=0,
                timeout=max(600, knotjob_timeout * (len(links[part]) // workers + 1)),
                verbose=verbose and workers > 1, num_workers=workers,
                per_link_timeout=knotjob_timeout))
            if verbose:
                done = min(start + chunk, len(links))
                print(f"    KnotJob progress: {100 * done // len(links)}% "
                      f"({done}/{len(links)} links) in {time() - t0:.1f}s",
                      flush=True)
        if verbose:
            print(f"  Computed {sum(v is not None for v in out.values())}"
                  f"/{len(links)} s-invariants in {time() - t0:.1f}s")
    except Exception as e:
        if verbose:
            print(f"  Batch invariant computation failed after "
                  f"{time() - t0:.1f}s: {e}")
        out = {}
    return out


def _check_known_values(summary: DatasetBoundsSummary, bounds: BoundsResult,
                        slice_genus, strong_slice_genus, i: int,
                        name: Optional[str]) -> Dict[str, Any]:
    """
    Compares the bounds of one link with its known slice and
    strong slice genus and records the outcome in summary.

    Returns:
        The slice_consistent, slice_tight and
        strong_slice_consistent columns of the row.
    """
    slice_consistent, slice_tight = True, False
    if slice_genus:
        # Compare with the set itself: {0, 2} misses [1, 1] although
        # the endpoints overlap.
        if not any(bounds.best_slice_lower <= v <= bounds.slice_upper_seifert
                   for v in slice_genus):
            slice_consistent = False
            summary.slice_inconsistent_indices.append(i)
            summary.slice_inconsistent_details.append(_inconsistency_detail(
                name, i, slice_genus, bounds.best_slice_lower,
                bounds.slice_upper_seifert,
                _lower_bounds_of(bounds, _SLICE_LOWER_BOUNDS)))
        slice_tight = (len(slice_genus) == 1
                       and bounds.best_slice_lower == slice_genus[0])
        summary.slice_known_count += 1
        summary.slice_consistent_count += int(slice_consistent)
        summary.slice_tight_count += int(slice_tight)

    strong_consistent = True
    if (not bounds.is_knot and bounds.is_algebraically_split
            and strong_slice_genus):
        if not any(v >= bounds.best_strong_slice_lower
                   for v in strong_slice_genus):
            strong_consistent = False
            summary.strong_slice_inconsistent_indices.append(i)
            summary.strong_slice_inconsistent_details.append(
                _inconsistency_detail(
                    name, i, strong_slice_genus,
                    bounds.best_strong_slice_lower, None,
                    _lower_bounds_of(bounds, _STRONG_LOWER_BOUNDS)))
        summary.strong_slice_known_count += 1
        summary.strong_slice_consistent_count += int(strong_consistent)

    if bounds.bounds_inconsistent:
        summary.bounds_inconsistent_count += 1
        summary.bounds_inconsistent_indices.append(i)
    return {'slice_consistent': slice_consistent, 'slice_tight': slice_tight,
            'strong_slice_consistent': strong_consistent}


def _count_in_summary(summary: DatasetBoundsSummary, bounds: BoundsResult,
                      best_bound_names: List[str]) -> None:
    """Adds one link to the counters of summary."""
    summary.knot_count += int(bounds.is_knot)
    summary.positive_count += int(bounds.is_positive)
    summary.alternating_count += int(bounds.is_alternating)
    summary.split_count += int(bounds.is_split)
    summary.alg_split_count += int(bounds.is_algebraically_split)
    summary.simply_linked_count += int(bounds.is_simply_linked)
    summary.component_bounds_count += int(bounds.component_genus_lower is not None)
    if bounds.component_identification is not None:
        statuses = bounds.component_identification
        summary.component_identified_count += statuses.count('identified')
        summary.component_ambiguous_count += statuses.count('ambiguous')
        # A known value strictly sharpened at least one field of at least
        # one component: counted once per link.
        sources = (bounds.component_genus_lower_source,
                   bounds.component_genus_upper_source,
                   bounds.component_u_lower_source,
                   bounds.component_u_upper_source)
        if any(src is not None
               and (src.startswith('known:') or src.startswith('subadditive:'))
               for lst in sources if lst is not None for src in lst):
            summary.component_known_improved_count += 1
    if (bounds.slice_lower_nu_hfk is not None
            and bounds.slice_lower_tau is not None
            and bounds.slice_lower_nu_hfk > bounds.slice_lower_tau):
        summary.nu_hfk_improved_count += 1
    if (bounds.strong_slice_lower_components is not None
            and bounds.strong_slice_lower_components
            == bounds.best_strong_slice_lower > 0):
        summary.strong_components_best_count += 1
    if (bounds.strong_slice_lower_nu_s is not None
            and bounds.strong_slice_lower_nu_s
            == bounds.best_strong_slice_lower > 0):
        summary.strong_nu_s_best_count += 1
    if bounds.s_invariant is not None:
        summary.s_invariant_computed_count += 1
        if bounds.s_invariant_from_positive:
            summary.s_invariant_from_positive_count += 1
        elif bounds.s_invariant_from_alternating_signature:
            summary.s_invariant_from_alternating_signature_count += 1
        elif bounds.s_invariant_from_knotjob:
            summary.s_invariant_from_knotjob_count += 1
    if bounds.slice_lower_fox_milnor_det is not None:
        summary.fox_milnor_det_checked_count += 1
        summary.fox_milnor_det_obstruction_count += int(
            bounds.slice_lower_fox_milnor_det > 0)
    if bounds.fox_milnor_satisfied is not None:
        summary.fox_milnor_full_checked_count += 1
        summary.fox_milnor_full_obstruction_count += int(
            bounds.fox_milnor_satisfied is False)
    if bounds.hfk_computed:
        summary.hfk_computed_count += 1
        summary.hfk_fibered_count += int(bool(bounds.hfk_fibered))
    combo = " + ".join(sorted(best_bound_names))
    summary.best_bound_counts[combo] = summary.best_bound_counts.get(combo, 0) + 1


def _merged_bound_columns(bounds: BoundsResult, slice_genus,
                          strong_slice_genus) -> Dict[str, Any]:
    """
    The best-bound value and source columns of one row: the
    theoretical bounds merged with the known values, known
    winning ties.
    """
    lo, lo_src = _slice_lower_with_source(bounds, slice_genus)
    up, up_src = _slice_upper_with_source(bounds, slice_genus)
    s_lo, s_lo_src = _strong_slice_lower_with_source(
        bounds, strong_slice_genus, bounds.is_knot, lo, lo_src)
    s_up, s_up_src = _strong_slice_upper_with_source(
        bounds, strong_slice_genus, bounds.is_knot,
        bounds.is_algebraically_split, bounds.num_components, up, up_src)
    return {'best_lower_bound_slice_genus': lo,
            'best_lower_bound_slice_genus_source': lo_src,
            'best_upper_bound_slice_genus': up,
            'best_upper_bound_slice_genus_source': up_src,
            'best_lower_bound_strong_slice_genus': s_lo,
            'best_lower_bound_strong_slice_genus_source': s_lo_src,
            'best_upper_bound_strong_slice_genus': s_up,
            'best_upper_bound_strong_slice_genus_source': s_up_src}


def _result_row(i: int, name: str, link: Link, bounds: BoundsResult,
                best_bound_names: List[str], known: Dict[str, str],
                merged: Dict[str, Any], consistency: Dict[str, Any]) -> dict:
    """
    One row of the results frame: the BoundsResult fields, the
    known values and the merged best-bound columns.

    Args:
        i: the index of the link in the dataset.
        name: the link name.
        link: the link.
        bounds: its BoundsResult.
        best_bound_names: the sources of best_slice_lower.
        known: the known_* columns, as strings.
        merged: the best_*_bound_* columns.
        consistency: the slice_consistent, slice_tight and
            strong_slice_consistent columns.
    """
    row: Dict[str, Any] = {
        'index': i, 'name': name,
        'PD_code': str([list(t) for t in link.PD_code()]).replace(' ', '').replace(',', ';')}
    row.update((c, getattr(bounds, c)) for c in _RESULT_COLUMNS)
    row['nontrivial_certificates'] = " + ".join(bounds.nontrivial_certificates)
    for column, _attr in _COMPONENT_COLUMNS:
        value = getattr(bounds, column)
        row[column] = None if value is None else str(value)
    row['strong_slice_obstruction_reason'] = bounds.strong_slice_obstruction_reason
    row['best_bound_names'] = best_bound_names
    row['bounds_inconsistent'] = bounds.bounds_inconsistent
    row['consistency_errors'] = " | ".join(bounds.consistency_errors)
    row.update((c, getattr(bounds, c)) for c in _HFK_COLUMNS)
    row.update(known)
    row.update(merged)
    row.update(consistency)
    row['error'] = None
    return row


def compute_bounds_for_dataset(
    dataset,
    verbose: bool = True,
    use_knotjob: bool = False,
    knotjob_timeout: int = 15,
    mt_primes: Optional[List[int]] = None,
    strong_mt_primes: Optional[List[int]] = None,
    num_workers: int = 1,
    knotjob_num_workers: int = 1,
    simplify_dataset: bool = False,
    use_positive_unknotting: bool = True,
    use_component_bounds: bool = True,
    use_component_identification: bool = True
):
    r"""
    Computes every slice genus bound for each link of a Dataset.

    Args:
        dataset: a Dataset object (see dataset.py).
        verbose: whether to print progress and summary statistics.
        use_knotjob: call KnotJob (needs Java) for the s-invariant
            of the links without a closed form; positive and
            non-split alternating diagrams are skipped from the
            batch.  Default False.
        knotjob_timeout: per-link timeout in seconds for KnotJob.
        mt_primes: orders a for the weak Murasugi-Tristram bound, each a
            prime power greater than 1; \omega = \zeta_a.  Defaults to [2, 3].
        strong_mt_primes: primes p for the strong Murasugi-Tristram bound;
            \omega_p = \zeta_p^{(p-1)/2}.  Defaults to [2, 3].
        simplify_dataset: if True, run dataset.simplify() once before any bound
            is computed, mutating the diagrams in place.  This changes c(D) and
            O(D), hence every diagram-dependent bound, and may destroy
            positivity.  Default False.
        use_positive_unknotting: certify each diagram as positive
            and apply u(L) = lk(L) + \sum_i u(K_i) to the
            unknotting bounds (positive_unknotting.py); skipped
            with a printed line when the knot table is missing.
            Default True.
        use_component_bounds: run the per-component pass on every
            multi-component link.  Default True.
        use_component_identification: with use_component_bounds,
            merge in the best known g_4 and u of each identified
            component (knot_identification.py).  Default True.
        num_workers: number of parallel Python/Sage worker processes for the
            per-link loop (default 1, i.e. sequential); budget a few hundred MB
            of memory per worker.
        knotjob_num_workers: number of parallel Java processes for the KnotJob
            batch (default 1).  Each worker spawns its own JVM, so the memory
            budget is workers x heap.

    Returns:
        (results_df, summary), a DataFrame of all the bounds and a
        DatasetBoundsSummary.
    """
    from dataset import invariant_to_str

    if simplify_dataset:
        t_simp = time()
        n_simp = dataset.simplify()
        if verbose:
            print(f"Simplified {n_simp}/{len(dataset)} links "
                  f"in {time() - t_simp:.1f}s", flush=True)

    summary = DatasetBoundsSummary(total_links=len(dataset))
    total = len(dataset)

    knotjob_time = 0.0
    precomputed: Dict[str, Optional[int]] = {}
    if use_knotjob and KNOTJOB_AVAILABLE:
        t_kj = time()
        precomputed = _precompute_s_invariants(
            dataset, knotjob_timeout, knotjob_num_workers, verbose)
        knotjob_time = time() - t_kj

    start_time = time()
    cumulative_times: Dict[str, float] = dict.fromkeys(
        ('sage', 'mt', 'strong_mt', 'hfk', 'fox_milnor_det',
         'fox_milnor_full', 'components'), 0.0)
    last_milestone_times = dict(cumulative_times)
    next_milestone = total // 10 if total >= 10 else total + 1

    # Parallel path: each worker gets one link's PD code and its precomputed
    # s, and the main loop below reads the results by index.
    parallel_bounds: Dict[int, Any] = {}  # idx -> (bounds_or_None, error_str_or_None)
    if num_workers > 1:
        if verbose:
            print(f"  Pre-computing bounds for {total} links "
                  f"using {num_workers} parallel workers (spawn context)...",
                  flush=True)
        par_start = time()
        par_tasks = []
        for i in range(total):
            link, name = dataset[i][0], dataset[i][-1]
            link_name = name if name else f"link_{i}"
            par_tasks.append((
                link.PD_code(),
                link.unlinked_unknot_components,
                [c.sign for c in link.crossings],
                link_name,
                i,
                precomputed.get(link_name),
                knotjob_timeout,
                mt_primes,
                strong_mt_primes,
                use_component_bounds,
                use_component_identification,
            ))
        ctx = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers, mp_context=ctx) as pool:
            for idx, _name, bounds_w, error_w in pool.map(
                    _compute_bounds_worker, par_tasks):
                parallel_bounds[idx] = (bounds_w, error_w)
        if verbose:
            n_ok = sum(1 for b, e in parallel_bounds.values() if e is None)
            err_str = f", {total - n_ok} errors" if total - n_ok else ""
            print(f"  Parallel bounds done in {time() - par_start:.1f}s "
                  f"({n_ok}/{total} succeeded{err_str})", flush=True)

    results = []
    for i in range(total):
        if verbose and i > 0 and i % next_milestone == 0:
            delta = {k: cumulative_times[k] - last_milestone_times[k]
                     for k in cumulative_times}
            print(f"  Progress: {100 * i // total}% ({i}/{total}), "
                  f"{time() - start_time:.1f}s elapsed"
                  f"  [sage {delta['sage']:.1f}s, mt {delta['mt']:.1f}s,"
                  f" strong_mt {delta['strong_mt']:.1f}s, hfk {delta['hfk']:.1f}s,"
                  f" comp {delta['components']:.1f}s]")
            last_milestone_times = dict(cumulative_times)

        # Dataset 9-tuple: link, unknotting, slice, strong slice, ribbon,
        # strong ribbon, splitting, weak splitting, name.  The two ribbon
        # genera are arguments to no bound; update_dataset_with_bounds
        # writes them.
        (link, unknotting_num, slice_genus, strong_slice_genus,
         _ribbon_genus, _strong_ribbon_genus,
         _splitting_num, _weak_splitting_num, name) = dataset[i]
        link_name = name if name else f"link_{i}"

        try:
            if parallel_bounds:
                bounds, error_w = parallel_bounds[i]
                if error_w is not None:
                    raise Exception(error_w)
            else:
                # In dataset mode the batch pre-pass is the only KnotJob
                # entry point.
                bounds = compute_bounds(
                    link,
                    use_knotjob=False,
                    knotjob_timeout=knotjob_timeout,
                    precomputed_s_invariant=precomputed.get(link_name),
                    mt_primes=mt_primes,
                    strong_mt_primes=strong_mt_primes,
                    use_component_bounds=use_component_bounds,
                    use_component_identification=use_component_identification
                )
            for k in cumulative_times:
                cumulative_times[k] += (bounds.phase_times or {}).get(k, 0.0)

            best_bound_names = _get_best_bound_names(bounds)
            consistency = _check_known_values(
                summary, bounds, slice_genus, strong_slice_genus, i, name)
            _count_in_summary(summary, bounds, best_bound_names)
            # Compact range form [0;9999], so that a trivial genus does not
            # bloat the bounds CSV.
            known = {'known_slice_genus': invariant_to_str(slice_genus),
                     'known_strong_slice_genus': invariant_to_str(strong_slice_genus),
                     'known_unknotting_number': invariant_to_str(unknotting_num)}
            results.append(_result_row(
                i, link_name, link, bounds, best_bound_names, known,
                _merged_bound_columns(bounds, slice_genus, strong_slice_genus),
                consistency))
        except Exception as e:
            results.append({'index': i, 'name': link_name, 'error': str(e)})

    df = pd.DataFrame(results)

    # Post-processing: the unknotting-number columns, maximised over the
    # orientation class, and the strong-slice columns, g_4^* not depending
    # on the orientation.
    _add_unknotting_bounds_to_results(
        df, dataset, summary=summary,
        use_positive_unknotting=use_positive_unknotting)
    _group_strong_slice_bounds_over_orientations(df, summary=summary,
                                                 dataset=dataset)

    bounds_time = time() - start_time
    if verbose:
        other_time = bounds_time - sum(cumulative_times.values())
        kj_str = f"knotjob {knotjob_time:.1f}s, " if knotjob_time > 0 else ""
        print(f"  Bounds computation completed in {knotjob_time + bounds_time:.1f}s"
              f"  [{kj_str}sage {cumulative_times['sage']:.1f}s, mt {cumulative_times['mt']:.1f}s,"
              f" strong_mt {cumulative_times['strong_mt']:.1f}s,"
              f" hfk {cumulative_times['hfk']:.1f}s,"
              f" comp {cumulative_times['components']:.1f}s,"
              f" fm_det {cumulative_times['fox_milnor_det']:.1f}s,"
              f" fm_full {cumulative_times['fox_milnor_full']:.1f}s,"
              f" other {other_time:.1f}s]")
        print(format_summary(summary))

    return df, summary


def _missing(v) -> bool:
    """True for None and NaN, the missing values of a results frame."""
    return v is None or (isinstance(v, float) and math.isnan(v))


def _strip_orientation_suffix(name: Optional[str]) -> Optional[str]:
    """
    Strips a trailing '{N}' orientation tag from a link name, so
    that the oriented variants of one link share a base key:
    'L10n1{0}' and 'L10n1{1}' give 'L10n1', '3_1' is unchanged.
    """
    if name is None:
        return None
    m = re.match(r"^(.*?)\{[^}]*\}\s*$", name)
    return m.group(1) if m else name


def _orientation_classes(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    The rows of df without an error, grouped by unoriented link:
    {base name: [row positions]}, in order of first appearance.
    """
    classes: Dict[str, List[int]] = defaultdict(list)
    for pos, (_, row) in enumerate(df.iterrows()):
        base = _strip_orientation_suffix(row.get('name'))
        if base is not None and _missing(row.get('error')):
            classes[base].append(pos)
    return classes


# Column -> provenance label for the unknotting bounds that do not factor
# through g_4 (Phase 11 of compute_bounds).  Order = reporting priority when
# several tie.
# Collari's equality on the two classes that certify it: the positive pass of
# positive_unknotting.py and the simply-linked component pass.  The _bounded
# form marks a positive row that stayed an interval.
_COLLARI_POSITIVE_SOURCE = 'collari_positive'
_COLLARI_SIMPLY_LINKED_SOURCE = 'collari_simply_linked'

_DIRECT_UNKNOTTING_SOURCES = (
    ('unknotting_lower_components', 'components_abslk'),
    ('unknotting_lower_lt', 'levine_tristram_conway'),
    ('unknotting_lower_slice_torus', 'slice_torus_nu_s'),
    ('unknotting_lower_splitting', 'splitting_number'),
)


def _direct_unknotting_source(row) -> str:
    """Name(s) of the Phase-11 unknotting bound(s) achieving the row's max."""
    best = row.get('best_unknotting_lower_direct')
    if best is None or (isinstance(best, float) and pd.isna(best)):
        return 'unknown'
    names = []
    for col, label in _DIRECT_UNKNOTTING_SOURCES:
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if int(v) == int(best):
            names.append(label)
    return " + ".join(names) if names else 'unknown'


def _positive_unknotting_rows(df: pd.DataFrame, dataset,
                             summary: Optional['DatasetBoundsSummary'] = None
                             ) -> Dict[int, dict]:
    """
    Runs the positive-link pipeline (Collari's theorem) over the dataset and
    returns {df row position: record} for every diagram certified positive.
    positive_unknotting is imported lazily, since that module itself imports
    from this one.

    Certification runs on the dataset's current diagrams, so simplifying
    them beforehand can lose rows here.
    """
    try:
        from bounds_pipeline.positive_unknotting import positive_unknotting
        import bounds_pipeline.knot_identification as ki
    except ImportError as exc:
        print(f"  positive-link unknotting disabled: {exc}")
        return {}
    table = ki.default_table()
    if table is None:
        return {}
    known = ki.load_known_values(table=table)

    out: Dict[int, dict] = {}
    for pos, (_, row) in enumerate(df.iterrows()):
        err = row.get('error')
        if err is not None and not (isinstance(err, float) and pd.isna(err)):
            continue
        i = int(row['index'])
        try:
            record = positive_unknotting(dataset.links[i], table, known,
                                         link_id=row.get('name'),
                                         record_diagram=False)
        except Exception as exc:
            print(f"  positive-link unknotting failed on {row.get('name')}: {exc}")
            continue
        if record is None:
            continue
        out[pos] = record
        if summary is not None:
            summary.positive_unknotting_certified_count += 1
            if record.exact:
                summary.positive_unknotting_exact_count += 1
    return out


def _group_strong_slice_bounds_over_orientations(
        df: pd.DataFrame,
        summary: Optional['DatasetBoundsSummary'] = None,
        dataset=None) -> None:
    r"""
    Replaces the strong-slice columns of df in place by their best
    value over the orientation class of each unoriented link,
    g_4^*(L) not depending on the orientation: each row takes the
    largest lower and the smallest upper bound of its class, the
    lower-bound source gaining the variant that produced it.
    Writes strong_slice_lower_over_orientations (the class maximum
    of the theoretical bound, which update_dataset_with_bounds
    narrows with), best_lower_bound_strong_slice_genus[_source]
    and best_upper_bound_strong_slice_genus[_source].  A class
    whose grouped lower bound overtakes its grouped upper bound is
    printed, the crossing being visible only after grouping.

    Args:
        df: the results frame, modified in place.
        summary: a DatasetBoundsSummary to count improvements in.
        dataset: a Dataset, for recomputing strong_slice_consistent
            against the class-wide bound.
    """
    if len(df) == 0 or 'best_strong_slice_lower' not in df.columns:
        return

    rows = [row for _, row in df.iterrows()]
    n = len(rows)
    col_theo: List[Optional[int]] = [None] * n
    col_lo: List[Any] = [None] * n
    col_lo_src: List[Optional[str]] = [None] * n
    col_up: List[Any] = [None] * n
    col_up_src: List[Optional[str]] = [None] * n
    improved = 0
    for base, positions in _orientation_classes(df).items():
        members = [rows[p] for p in positions]
        theo = max(int(r['best_strong_slice_lower']) for r in members)
        lower = [(r['best_lower_bound_strong_slice_genus'], r) for r in members
                 if not _missing(r.get('best_lower_bound_strong_slice_genus'))]
        best = max(lower, key=lambda t: t[0]) if lower else None
        upper = [(r['best_upper_bound_strong_slice_genus'], r) for r in members
                 if not _missing(r.get('best_upper_bound_strong_slice_genus'))]
        least = min(upper, key=lambda t: t[0]) if upper else None
        if best is not None and least is not None and best[0] > least[0]:
            # Only visible after grouping: the class maximum of the lower
            # bounds can overtake the class minimum of the upper bounds.
            print(f"WARNING: {base}: strong slice bounds cross after grouping "
                  f"over the orientation class ({best[0]} > {least[0]})")
        for p, r in zip(positions, members):
            col_theo[p] = theo
            improved += int(theo > int(r['best_strong_slice_lower']))
            if best is not None:
                value, winner = best
                src = winner.get('best_lower_bound_strong_slice_genus_source',
                                 'unknown')
                col_lo[p] = value
                col_lo_src[p] = (src if winner.get('name') == r.get('name')
                                 else f"{src}_{winner.get('name')}")
            if least is not None:
                col_up[p] = least[0]
                col_up_src[p] = least[1].get(
                    'best_upper_bound_strong_slice_genus_source', 'unknown')

    # strong_slice_consistent was decided against the per-orientation bound;
    # decide it again against the class maximum.
    if dataset is not None and 'strong_slice_consistent' in df.columns:
        recomputed: List[Any] = list(df['strong_slice_consistent'])
        for p, r in enumerate(rows):
            if (col_theo[p] is None or bool(r.get('is_knot', False))
                    or not bool(r.get('is_algebraically_split', False))):
                continue
            known = dataset.strong_slice_genera[int(r['index'])]
            if known:
                recomputed[p] = max(known) >= col_theo[p]
        df['strong_slice_consistent'] = recomputed

    df['strong_slice_lower_over_orientations'] = col_theo
    if any(v is not None for v in col_lo):
        df['best_lower_bound_strong_slice_genus'] = col_lo
        df['best_lower_bound_strong_slice_genus_source'] = col_lo_src
    if any(v is not None for v in col_up):
        df['best_upper_bound_strong_slice_genus'] = col_up
        df['best_upper_bound_strong_slice_genus_source'] = col_up_src
    if summary is not None:
        summary.strong_slice_orientation_improved_count = improved


# The columns _add_unknotting_bounds_to_results writes, in order.
_UNKNOTTING_COLUMNS = (
    'unknotting_lower_unproved', 'unknotting_lower', 'unknotting_upper',
    'unknotting_consistent', 'best_lower_bound_unknotting_number',
    'best_lower_bound_unknotting_number_source',
    'best_upper_bound_unknotting_number',
    'best_upper_bound_unknotting_number_source')
_POSITIVE_COLUMNS = (
    'positive_certified', 'positive_mirrored', 'positive_orientation',
    'positive_lk', 'positive_u', 'positive_u_lower', 'positive_u_upper',
    'positive_components', 'positive_provenance')


def _per_orientation_unknotting(row, dataset, record) -> dict:
    """
    The unknotting bounds of one results row before the grouping
    over the orientation class: lower with its source src, upper
    with upper_src, the crossing-count heuristic unproved, and the
    positive-link bounds pos_lower and pos_upper, None without a
    record.

    Args:
        row: the results row.
        dataset: the Dataset.
        record: the PositiveUnknottingRecord of the row, or None.
    """
    i = int(row['index'])
    known = dataset.slice_genera[i]
    lower = int(row['best_slice_lower'])
    src = row.get('best_lower_bound_slice_genus_source', 'unknown')
    # u >= g_4; the known g_4 floor wins ties.
    known_lower = min(known) if known else 0
    if known_lower >= lower:
        lower, src = known_lower, 'known_slice_genus'
    # The Phase-11 bounds, taken only when strictly stronger.
    direct = row.get('best_unknotting_lower_direct')
    if not _missing(direct) and int(direct) > lower:
        lower, src = int(direct), _direct_unknotting_source(row)
    # u >= 1 when a certificate proves L is not the unlink.
    cert = row.get('nontrivial_certificates')
    cert = '' if _missing(cert) else str(cert)
    if lower == 0 and cert:
        lower, src = 1, 'not_unlink_' + cert.replace(' + ', '_')
    # Collari's theorem on a certified positive diagram; for an exact
    # record u itself is the lower bound.
    pos_lower = pos_upper = None
    if record is not None:
        pos_lower = record.u if record.exact else record.u_lower
        pos_upper = record.u if record.exact else record.u_upper
        if pos_lower is not None and pos_lower > lower:
            lower, src = pos_lower, record.provenance
    upper, upper_src = int(row['num_crossings']) // 2, 'crossing_change_diagram'
    # On a simply-linked diagram \abslk + \sum_i \bar u(K_i) is an upper
    # bound (Collari's equality).
    sl_up = row.get('unknotting_upper_simply_linked')
    if not _missing(sl_up) and int(sl_up) < upper:
        upper, upper_src = int(sl_up), _COLLARI_SIMPLY_LINKED_SOURCE
    unproved = max(lower, 1 if len(dataset.links[i].crossings) > 0 else 0)
    return {'lower': lower, 'src': src, 'upper': upper, 'upper_src': upper_src,
            'unproved': unproved, 'pos_lower': pos_lower, 'pos_upper': pos_upper}


def _add_unknotting_bounds_to_results(
    df: pd.DataFrame,
    dataset,
    summary: Optional['DatasetBoundsSummary'] = None,
    use_positive_unknotting: bool = True,
) -> None:
    """
    Populates the unknotting columns of df in place.  Per row the
    lower bound is the largest of best_slice_lower, the known g_4
    floor, the Phase-11 bounds, 1 when a certificate proves L is
    not the unlink, and the positive-link value when the diagram
    certifies positive; the upper bound is the smallest of
    num_crossings // 2, the simply-linked bound and the positive
    value.  Rows are then grouped by unoriented link, u not
    depending on the orientation: each lower bound becomes the
    group maximum, its source suffixed with the variant that
    produced it (murasugi_tristram_L2a1{0}), and each upper bound
    the group minimum.  Also writes unknotting_consistent and,
    when summary is given, the consistency statistics.
    """
    n = len(df)
    cols: Dict[str, list] = {c: [None] * n
                             for c in _UNKNOTTING_COLUMNS + _POSITIVE_COLUMNS}
    cols['positive_certified'] = [False] * n
    positive = (_positive_unknotting_rows(df, dataset, summary)
                if n and use_positive_unknotting else {})
    rows = [row for _, row in df.iterrows()]

    for base, positions in _orientation_classes(df).items():
        per = [(pos, _per_orientation_unknotting(rows[pos], dataset,
                                                 positive.get(pos)))
               for pos in positions]
        # The class maximum of the lower bounds, with the variant that
        # produced it, and the class minimum of the upper bounds; a
        # positive-certified upper bound holds for every orientation too.
        win_pos, win = max(per, key=lambda m: m[1]['lower'])
        final_lower = win['lower']
        lower_src = f"{win['src']}_{rows[win_pos]['name']}"
        _, least = min(per, key=lambda m: m[1]['upper'])
        upper, upper_src = least['upper'], least['upper_src']
        pos_uppers = [m['pos_upper'] for _, m in per if m['pos_upper'] is not None]
        if pos_uppers and min(pos_uppers) < upper:
            upper, upper_src = min(pos_uppers), _COLLARI_POSITIVE_SOURCE
        if final_lower > upper:
            # Only visible after grouping: the class maximum of the lower
            # bounds can overtake the class minimum of the upper bounds.
            print(f"WARNING: {base}: unknotting bounds cross after grouping "
                  f"over the orientation class ({final_lower} > {upper})")

        for pos, m in per:
            cols['unknotting_lower_unproved'][pos] = m['unproved']
            cols['unknotting_lower'][pos] = final_lower
            cols['unknotting_upper'][pos] = upper
            # Best-bound columns: merged with the known unknotting number,
            # as the slice columns are.  The theoretical values stay in
            # unknotting_lower / unknotting_upper.
            i = int(rows[pos]['index'])
            known_u = dataset.unknotting_nums[i]
            best_lo, best_hi = final_lower, upper
            lo_src, hi_src = lower_src, upper_src
            if known_u:
                if min(known_u) >= best_lo:
                    best_lo, lo_src = min(known_u), 'known_unknotting_number'
                if max(known_u) <= best_hi:
                    best_hi, hi_src = max(known_u), 'known_unknotting_number'
            cols['best_lower_bound_unknotting_number'][pos] = best_lo
            cols['best_lower_bound_unknotting_number_source'][pos] = lo_src
            cols['best_upper_bound_unknotting_number'][pos] = best_hi
            cols['best_upper_bound_unknotting_number_source'][pos] = hi_src
            # Consistency against the known unknotting number, when there
            # is one.
            if not known_u:
                continue
            consistent = any(final_lower <= v <= upper for v in known_u)
            cols['unknotting_consistent'][pos] = consistent
            if summary is not None:
                summary.unknotting_known_count += 1
                if consistent:
                    summary.unknotting_consistent_count += 1
                else:
                    summary.unknotting_inconsistent_indices.append(i)
                    summary.unknotting_inconsistent_details.append(
                        _inconsistency_detail(
                            rows[pos].get('name'), i, known_u, final_lower,
                            upper, [('unknotting_lower', final_lower)]))

    # Provenance of the positive-link pass, so that any row can be re-derived.
    for pos, record in positive.items():
        cols['positive_certified'][pos] = True
        cols['positive_mirrored'][pos] = record.mirrored
        cols['positive_orientation'][pos] = str(record.orientation)
        cols['positive_lk'][pos] = record.lk
        cols['positive_u'][pos] = record.u
        cols['positive_u_lower'][pos] = record.u_lower
        cols['positive_u_upper'][pos] = record.u_upper
        cols['positive_components'][pos] = str(record.components)
        cols['positive_provenance'][pos] = record.provenance
    for column, values in cols.items():
        df[column] = values

    if summary is not None:
        lo_srcs = cols['best_lower_bound_unknotting_number_source']
        hi_srcs = cols['best_upper_bound_unknotting_number_source']
        summary.positive_unknotting_improved_count = sum(
            1 for lo, hi in zip(lo_srcs, hi_srcs)
            if (lo is not None and 'collari' in lo)
            or hi == _COLLARI_POSITIVE_SOURCE)
        direct_labels = tuple(label for _, label in _DIRECT_UNKNOTTING_SOURCES)
        summary.unknotting_direct_best_count = sum(
            1 for lo in lo_srcs
            if lo is not None and any(lb in lo for lb in direct_labels))


# =============================================================================
# DATASET UPDATE
# =============================================================================

def _strong_lower_of(row) -> int:
    """
    The strong-slice lower bound of one results row: the orientation-class
    maximum when _group_strong_slice_bounds_over_orientations has run, else the
    per-orientation value.
    """
    v = row.get('strong_slice_lower_over_orientations')
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return int(row['best_strong_slice_lower'])
    return int(v)


def _usable_rows(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    The rows of a results frame a dataset update may act on: no
    error and no internally contradictory pair of bounds.  An
    empty frame has no columns, and a frame mixing successful and
    error rows has float columns, so consumers cast after
    filtering.
    """
    if len(results_df) == 0 or 'error' not in results_df.columns:
        return results_df.iloc[0:0]
    df = results_df[results_df['error'].isna()]
    if 'bounds_inconsistent' in df.columns:
        df = df[~df['bounds_inconsistent'].fillna(False).astype(bool)]
    return df


def _narrow_known(known, lower, upper) -> tuple:
    """
    Intersects a known value set with the interval [lower, upper];
    an unknown invariant becomes the whole interval.

    Returns:
        (values, conflict); on an empty intersection the known set
        is kept and conflict is True.
    """
    if known is None or len(known) == 0:
        return list(range(int(lower), int(upper) + 1)), False
    new = [v for v in known if lower <= v <= upper]
    if new:
        return new, False
    return list(known), True


def _intersect_sets(a, b) -> tuple:
    """
    Intersects two known value sets, either of which may be None
    or empty.

    Returns:
        (values, conflict); values is None when neither side knows
        anything, and on an empty intersection a is kept and
        conflict is True.
    """
    if not a:
        return (list(b) if b else None), False
    if not b:
        return list(a), False
    common = [v for v in a if v in b]
    if common:
        return common, False
    return list(a), True


def update_dataset_with_bounds(dataset, results_df: pd.DataFrame) -> None:
    r"""
    Updates the slice and strong slice genera of a Dataset in place from the
    computed bounds, and mirrors them onto the ribbon and strong ribbon genera,
    a ribbon surface being in particular a slice surface.

    Every write is an intersection: a known value is never widened or
    discarded, and when it contradicts a computed bound it is kept and the
    conflict printed.  For a knot g_4^*(K) = g_4(K), so each narrows the
    other.

    The strong genera are canonicalised over the orientation class, g_4^*
    not depending on the orientation: every oriented variant of one link
    ends with the same set.  The weak slice genus stays per row.
    """
    df = _usable_rows(results_df)
    if len(df) == 0:
        return

    conflicts = []

    # ---- weak slice and ribbon genus: per row -----------------------------
    for _, row in df.iterrows():
        i = int(row['index'])
        lower = int(row['best_slice_lower'])
        upper = int(row['slice_upper_seifert'])
        name = row.get('name', f'link_{i}')

        new, clash = _narrow_known(dataset.slice_genera[i], lower, upper)
        dataset.slice_genera[i] = new
        if clash:
            conflicts.append(f"{name}: known slice genus {dataset.slice_genera[i]} "
                             f"misses the computed range [{lower};{upper}]")

        new_r, clash_r = _narrow_known(dataset.ribbon_genera[i], lower, upper)
        dataset.ribbon_genera[i] = new_r
        if clash_r:
            conflicts.append(f"{name}: known ribbon genus {dataset.ribbon_genera[i]} "
                             f"misses the computed range [{lower};{upper}]")

    # ---- strong slice and strong ribbon genus: per orientation class ------
    # Three cases by topology, decided per class.  is_algebraically_split is
    # orientation-invariant; if the variants disagree anyway, the infinite
    # case wins, being the only one backed by a proof.
    rows = [row for _, row in df.iterrows()]
    for base, positions in _orientation_classes(df).items():
        members = [rows[p] for p in positions]
        idxs = [int(r['index']) for r in members]
        is_knot = any(bool(r.get('is_knot', False)) for r in members)
        not_alg_split = any(not bool(r.get('is_algebraically_split', False))
                            for r in members)

        if is_knot:
            # (a) g_4^*(K) = g_4(K).  Intersect in both directions, then write
            # the same set to the weak and the strong invariant.
            for i, r in zip(idxs, members):
                for weak_attr, strong_attr in (
                        ('slice_genera', 'strong_slice_genera'),
                        ('ribbon_genera', 'strong_ribbon_genera')):
                    weak = getattr(dataset, weak_attr)[i]
                    strong = getattr(dataset, strong_attr)[i]
                    combined, clash = _intersect_sets(weak, strong)
                    if clash:
                        conflicts.append(
                            f"{r.get('name')}: {weak_attr} {weak} and "
                            f"{strong_attr} {strong} do not intersect, though "
                            f"they are equal for a knot")
                        continue
                    if combined is not None:
                        getattr(dataset, weak_attr)[i] = list(combined)
                        getattr(dataset, strong_attr)[i] = list(combined)
            continue

        if not_alg_split:
            # (b) some pair of components has non-zero linking number, so no
            # strong slice surface exists.  This is a proof, so it overwrites.
            for i, r in zip(idxs, members):
                for attr, label in (('strong_slice_genera', 'strong_slice_genus'),
                                    ('strong_ribbon_genera', 'strong_ribbon_genus')):
                    prev = getattr(dataset, attr)[i]
                    if prev is not None and prev != [math.inf]:
                        print(f"WARNING: overwriting {label} for "
                              f"{r.get('name')}: previous value {prev} replaced "
                              f"with [inf] because the linking number is "
                              f"non-zero and hence it must be infinite.")
                    getattr(dataset, attr)[i] = [math.inf]
            continue

        # (c) algebraically split: one canonical set for the whole class.
        lower = max(_strong_lower_of(r) for r in members)
        for attr in ('strong_slice_genera', 'strong_ribbon_genera'):
            known = None
            clash = False
            for i in idxs:
                known, step_clash = _intersect_sets(known, getattr(dataset, attr)[i])
                clash = clash or step_clash
            if clash:
                conflicts.append(
                    f"{base}: the oriented variants disagree about {attr}, "
                    f"which does not depend on the orientation")
            values, clash_lo = _narrow_known(known, lower, 9999)
            if clash_lo:
                conflicts.append(
                    f"{base}: known {attr} {known} lies entirely below the "
                    f"computed lower bound {lower}")
            for i in idxs:
                getattr(dataset, attr)[i] = list(values)

    for line in conflicts:
        print(f"WARNING: {line}")


def update_dataset_with_unknotting_bounds(dataset, results_df: pd.DataFrame) -> None:
    """
    Updates the unknotting_nums of a Dataset in place from the
    'unknotting_lower' and 'unknotting_upper' columns of results_df.

    Like the genus updater, every write is an intersection: a known value is
    never widened, and an empty intersection is reported rather than applied.
    The unknotting number does not depend on the orientation, so the rows of one
    orientation class are canonicalised together, ending with the intersection
    of everything any variant knew and the bounds proved on any of them.
    """
    df = _usable_rows(results_df)
    if (len(df) == 0
            or 'unknotting_lower' not in df.columns
            or 'unknotting_upper' not in df.columns):
        return

    rows = [row for _, row in df.iterrows()]
    conflicts = []
    for base, positions in _orientation_classes(df).items():
        members = [rows[p] for p in positions]
        bounds = []
        for r in members:
            lo, hi = r.get('unknotting_lower'), r.get('unknotting_upper')
            if _missing(lo) or _missing(hi):
                continue
            bounds.append((int(lo), int(hi)))
        if not bounds:
            continue
        lower = max(lo for lo, _ in bounds)
        upper = min(hi for _, hi in bounds)
        if lower > upper:
            conflicts.append(f"{base}: computed unknotting bounds cross "
                             f"({lower} > {upper}); the known value is kept")
            continue

        idxs = [int(r['index']) for r in members]
        known = None
        clash = False
        for i in idxs:
            known, step_clash = _intersect_sets(known, dataset.unknotting_nums[i])
            clash = clash or step_clash
        if clash:
            conflicts.append(f"{base}: the oriented variants disagree about the "
                             f"unknotting number, which does not depend on the "
                             f"orientation")
        values, clash_lo = _narrow_known(known, lower, upper)
        if clash_lo:
            conflicts.append(f"{base}: known unknotting number {known} misses the "
                             f"computed range [{lower};{upper}]")
        for i in idxs:
            dataset.unknotting_nums[i] = list(values)

    for line in conflicts:
        print(f"WARNING: {line}")


# The reports live in bounds_report.py and are re-exported here, so that
# every caller imports them from one module.
from bounds_pipeline.bounds_report import (format_bounds, format_summary,  # noqa: E402
                           format_inconsistency_report, print_improved_bounds,
                           print_improved_unknotting_bounds)
