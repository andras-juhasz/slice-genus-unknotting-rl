# The Bar-Natan / van der Veen Theta = (Delta, theta) invariant, from D.
# Bar-Natan and R. van der Veen, "A Fast, Strong, Topologically Meaningful
# and Fun Knot Invariant", arXiv:2509.18456.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 2 of the License, or (at your option)
# any later version. Step 2 translates Theta.sage (described below) and is
# used with the permission of Roland van der Veen and Dror Bar-Natan, under
# the terms of Dror Bar-Natan's copyleft notice,
# https://www.math.toronto.edu/~drorbn/Copyleft/.
#
# Two steps, kept in this order in the file:
#
#   1. Diagram bookkeeping: From PD code to the upright (Xings, rotations)
#      presentation the invariant is defined on.
#   2. Numerical evaluation at a rational point (T1, T2). Float64 linear
#      algebra over that data.
#
# Xings[k] is [sign, over_in_arc, under_in_arc] for crossing k. rotations[k]
# is phi_k for arc k, with arcs numbered 0 .. 2c+m-1 where c is the number of
# crossings and m the number of components of the braid closure.
#
# Step 2 transcribes the algorithm of Theta.sage, Roland van der Veen's Sage
# implementation, published at https://www.rolandvdv.nl/Theta/. Every monomial
# of F1, F2 and Gam1 is preserved one to one, and the conventions are its
# own: strands numbered 1..b from left to right at the top, arc 0 entering the
# closure of strand 1, arcs numbered along the braid word.
#
# One deliberate difference: while Theta.sage substitutes t1 := t2 (or
# t1 := t1*t2) into a symbolic inverse, this code re-inverts A numerically at
# T2 (or T1*T2). That is the same quantity at the chosen evaluation point and
# is stable in float64 for diagrams of a few dozen crossings.
#
# Checked against a Sage evaluation of the same invariant on the 354 links of
# the LinkInfo census up to 9 crossings.
#
# Not implemented: rho1 (residue formulas R1, R2, R3 of the paper, section
# 4.2). Conjecturally rho1(K) = -theta(K) at T2 = 1, which is a pole of the
# F1 and F2 denominators, so it needs those formulas rather than a limit.

import std/algorithm
import std/math
import std/sequtils
import std/sets
import std/tables
import "../spherogram-nim/src/links"
import "../spherogram-nim/src/seifert"

# =============================================================================
# 1. PD CODE -> UPRIGHT PRESENTATION
# =============================================================================

type CrossingRecord = tuple[sign: int, over_strand: int, under_strand: int, gen_idx: int]

proc braid_to_upright*(b: int, word: seq[int]):
        (seq[array[0..2, int]], seq[int]) =
    discard """
    Given a braid word on b strands, converts it to the
    upright (Xings, rotations) presentation.

    Args:
        b: the number of strands of the braid
        word: the braid word, as a Tietze list of signed
            integers in {-(b-1), ..., -1, 1, ..., b-1}

    Returns:
        (Xings, rotations), of lengths c and 2c+m, where c
        is the number of crossings and m the number of
        components of the braid closure.

    Raises:
        ValueError: if a generator of word is out of range
            for a braid on b strands
    """
    let c = word.len

    var pos = newSeq[int](b + 1)
    var pos_to_strand = newSeq[int](b + 1)
    for k in 0 .. b:
        pos[k] = k
        pos_to_strand[k] = k

    var crossings_data: seq[CrossingRecord] = @[]
    for k in 0 ..< c:
        let g = word[k]
        if g == 0 or abs(g) >= b:
            raise newException(ValueError,
                "Generator " & $g & " out of range for braid on " & $b & " strands")
        let i = abs(g)
        let sign = if g > 0: 1 else: -1
        let s_left = pos_to_strand[i]
        let s_right = pos_to_strand[i + 1]
        let over_strand = if sign == 1: s_left else: s_right
        let under_strand = if sign == 1: s_right else: s_left
        crossings_data.add((sign, over_strand, under_strand, k))
        pos_to_strand[i] = s_right
        pos_to_strand[i + 1] = s_left
        pos[s_left] = i + 1
        pos[s_right] = i

    # Cycles of the closure permutation.
    var visited = newSeq[bool](b + 1)
    visited[0] = true
    var cycles: seq[seq[int]] = @[]
    for start in 1 .. b:
        if visited[start]: continue
        var cycle: seq[int] = @[]
        var cur = start
        while not visited[cur]:
            visited[cur] = true
            cycle.add(cur)
            cur = pos[cur]
        cycles.add(cycle)

    # Per-strand ordered (top-to-bottom) list of (gen_idx, role).
    var strand_to_crossings = initTable[int, seq[(int, string)]]()
    for k in 1 .. b:
        strand_to_crossings[k] = @[]
    for cd in crossings_data:
        strand_to_crossings[cd.over_strand].add((cd.gen_idx, "over"))
        strand_to_crossings[cd.under_strand].add((cd.gen_idx, "under"))
    for k in 1 .. b:
        strand_to_crossings[k].sort(proc (x, y: (int, string)): int = x[0] - y[0])

    # Traverse components, numbering arcs in orientation order.
    var crossing_arcs = initTable[int, Table[string, int]]()
    var closure_arcs: seq[int] = @[]
    var cur_arc = 0
    for cycle in cycles:
        for idx_in_cycle, s in cycle:
            for visit in strand_to_crossings[s]:
                let (gen_idx, role) = visit
                if gen_idx notin crossing_arcs:
                    crossing_arcs[gen_idx] = initTable[string, int]()
                crossing_arcs[gen_idx][role] = cur_arc
                cur_arc += 1
            if idx_in_cycle < cycle.len - 1:
                closure_arcs.add(cur_arc)
        # End of cycle: advance past this component's cut +infinity arc.
        cur_arc += 1

    let total_arcs = 2 * c + cycles.len
    doAssert cur_arc == total_arcs,
        "arc count " & $cur_arc & " != 2c+m " & $total_arcs

    var xings: seq[array[0..2, int]] = @[]
    for cd in crossings_data:
        xings.add([cd.sign,
                   crossing_arcs[cd.gen_idx]["over"],
                   crossing_arcs[cd.gen_idx]["under"]])

    var rotations = newSeq[int](total_arcs)
    for arc in closure_arcs:
        rotations[arc] -= 1

    return (xings, rotations)


proc braid_data*[T](link: Link[T]): (int, seq[int]) =
    discard """
    Given a link, calculates the Vogel braid form of its
    diagram. Companion of seifert.braid_word, which does not
    expose the strand count that reconstructing the closure
    permutation needs.

    Args:
        link: the given link

    Returns:
        (b, word): the number of strands and the Tietze
        braid word of the Vogel braid form of link.
    """
    var arrows = braid_arrows(link)
    var word: seq[int] = @[]
    var max_strand = -1
    for arrow in arrows:
        let strand = arrow[1]
        let over_or_under = arrow[2]
        word.add([-1, 1][over_or_under] * (strand + 1))
        if strand > max_strand: max_strand = strand
    let b = if max_strand < 0: 1 else: max_strand + 2
    return (b, word)

proc pd_to_upright*[T](link: Link[T]):
        (seq[array[0..2, int]], seq[int]) =
    discard """
    Given a link, converts it to the upright presentation by
    way of its Vogel braid form.

    braid_arrows (spherogram-nim) does integer winding
    arithmetic that can overflow on a degenerate diagram,
    raising an OverflowDefect that callers cannot catch by
    default. That library is vendored and left unmodified,
    so the defect is converted here into a ValueError, which
    callers read as "Theta is undefined on this diagram".

    Args:
        link: the given link

    Returns:
        (Xings, rotations) for the diagram of link.

    Raises:
        ValueError: if the braid form of the diagram cannot
            be built
    """
    var b: int
    var word: seq[int]
    try:
        (b, word) = braid_data(link)
    except OverflowDefect as e:
        raise newException(ValueError,
            "pd_to_upright: braid_arrows overflowed on a degenerate diagram: " &
            e.msg)
    return braid_to_upright(b, word)


# =============================================================================
# 2. NUMERICAL EVALUATION OF THETA
# =============================================================================

# Below this a pivot counts as zero and the matrix as
# singular, which makes Theta undefined at that evaluation
# point. Raise it if float64 noise at large crossing numbers
# starts producing false inverses.
const PIVOT_EPS = 1e-15

proc build_A*(xings: seq[array[0..2, int]], N: int, T: float): seq[seq[float]] =
    discard """
    Builds the N x N matrix A of Theta.sage, evaluated at
    t1 = T. For Delta2 and G2 the caller passes T = T2, for
    Delta3 and G3, T = T1*T2.

    Args:
        xings: the crossings of the upright presentation,
            each [sign, over_in arc, under_in arc] with sign
            in {+1, -1} and arc indices in 0..N-1
        N: the number of arcs
        T: the point at which to evaluate

    Returns:
        The matrix A, as a sequence of rows.
    """
    result = newSeqWith(N, newSeq[float](N))
    for k in 0 ..< N:
        result[k][k] = 1.0
    for c in xings:
        let s = c[0]
        let i = c[1]
        let j = c[2]
        let ts = pow(T, float(s))
        result[i][i + 1] = -ts
        result[i][j + 1] = ts - 1.0
        result[j][j + 1] = -1.0

proc invert_with_det*(A: var seq[seq[float]]): (seq[seq[float]], float) =
    discard """
    Inverts a matrix by Gauss-Jordan elimination with
    partial pivoting, computing its determinant on the way.

    Args:
        A: the matrix to invert, used as scratch space and
            left in an unspecified state

    Returns:
        (A^-1, det(A)).

    Raises:
        ValueError: if a pivot falls below PIVOT_EPS, in
            which case the matrix counts as singular
    """
    let N = A.len
    var inv = newSeqWith(N, newSeq[float](N))
    for k in 0 ..< N:
        inv[k][k] = 1.0
    var det = 1.0
    for col in 0 ..< N:
        var pivot_row = col
        var max_abs = abs(A[col][col])
        for r in (col + 1) ..< N:
            let v = abs(A[r][col])
            if v > max_abs:
                max_abs = v
                pivot_row = r
        if max_abs < PIVOT_EPS:
            raise newException(ValueError,
                "singular matrix: pivot < " & $PIVOT_EPS & " at column " & $col)
        if pivot_row != col:
            swap(A[col], A[pivot_row])
            swap(inv[col], inv[pivot_row])
            det = -det
        let p = A[col][col]
        det *= p
        let inv_p = 1.0 / p
        for j in 0 ..< N:
            A[col][j] *= inv_p
            inv[col][j] *= inv_p
        for r in 0 ..< N:
            if r == col: continue
            let f = A[r][col]
            if f == 0.0: continue
            for j in 0 ..< N:
                A[r][j] -= f * A[col][j]
                inv[r][j] -= f * inv[col][j]
    return (inv, det)

proc compute_delta*(T: float, det_A: float,
                    phi: seq[int], xings: seq[array[0..2, int]]): float =
    discard """
    Calculates Delta = t^((-sum(phi) - sum(signs))/2) det(A)
    at a single evaluation point.

    Args:
        T: the point at which A was evaluated
        det_A: the determinant of A at that point
        phi: the rotation numbers of the arcs
        xings: the crossings of the upright presentation

    Returns:
        The value of Delta at T.
    """
    var sphi = 0
    for p in phi: sphi += p
    var sxs = 0
    for c in xings: sxs += c[0]
    let num = -sphi - sxs
    doAssert num mod 2 == 0,
        "Delta exponent (-sum(phi) - sum(signs)) must be even, got " & $num
    return pow(T, float(num div 2)) * det_A

proc F1(s: int, i, j: int, T1, T2: float,
        G1, G2, G3: seq[seq[float]]): float =
    discard """
    Calculates the contribution of a single crossing to
    theta.

    Args:
        s: the sign of the crossing
        i: its over_in arc
        j: its under_in arc
        T1: the first evaluation point
        T2: the second evaluation point
        G1: the inverse of A at T1
        G2: the inverse of A at T2
        G3: the inverse of A at T1*T2

    Returns:
        The contribution of the crossing.
    """
    let t1s = pow(T1, float(s))
    let t2s = pow(T2, float(s))
    let t2_2s = t2s * t2s
    let t1t2s = t1s * t2s             # (T1*T2)^s = T1^s * T2^s for integer s
    let denom = t2s - 1.0
    let inv_denom = 1.0 / denom

    var acc = 0.5
    acc += t2s * G1[i][i] * G2[j][i]
    acc += (t1s - 1.0) * t2_2s * G1[j][i] * G2[j][i] * inv_denom
    acc -= G1[i][i] * G2[j][j]
    acc -= (t1s - 1.0) * t2s * G1[j][i] * G2[j][j] * inv_denom
    acc -= G3[i][i]
    acc -= (t2s - 1.0) * G2[j][i] * G3[i][i]
    acc += 2.0 * G2[j][j] * G3[i][i]
    acc += (t1t2s - 1.0) * G3[j][i] * inv_denom
    acc -= t2s * (t1t2s - 1.0) * G1[i][i] * G3[j][i] * inv_denom
    acc -= (t1s - 1.0) * (t2s + 1.0) * (t1t2s - 1.0) * G1[j][i] * G3[j][i] * inv_denom
    acc += (t1t2s - 1.0) * G2[i][j] * G3[j][i] * inv_denom
    acc += (t1t2s - 1.0) * G2[j][i] * G3[j][i]
    acc += (t2s - 2.0) * (t1t2s - 1.0) * G2[j][j] * G3[j][i] * inv_denom
    acc += G1[i][i] * G3[j][j]
    acc += (t1s - 1.0) * t2s * G1[j][i] * G3[j][j] * inv_denom
    acc -= G2[i][i] * G3[j][j]
    acc -= t2s * G2[j][i] * G3[j][j]
    return float(s) * acc

proc F2(s0, i0, j0, s1, i1, j1: int, T1, T2: float,
        G1, G2, G3: seq[seq[float]]): float =
    discard """
    Calculates the contribution of an ordered pair of
    crossings (c0, c1) to theta.

    Args:
        s0: the sign of c0
        i0: the over_in arc of c0
        j0: the under_in arc of c0
        s1: the sign of c1
        i1: the over_in arc of c1
        j1: the under_in arc of c1
        T1: the first evaluation point
        T2: the second evaluation point
        G1: the inverse of A at T1
        G2: the inverse of A at T2
        G3: the inverse of A at T1*T2

    Returns:
        The contribution of the pair.
    """
    let t1s0 = pow(T1, float(s0))
    let t2s0 = pow(T2, float(s0))
    let t1s1 = pow(T1, float(s1))
    let t2s1 = pow(T2, float(s1))
    let t1t2s1 = t1s1 * t2s1
    let inv_denom = 1.0 / (t2s1 - 1.0)

    let a = (t1s0 - 1.0) * t2s0 * (t1t2s1 - 1.0) * G1[j1][i0] * G2[i1][i0] * G3[j0][i1]
    let b = (t1s0 - 1.0) * (t1t2s1 - 1.0) * G1[j1][i0] * G2[i1][j0] * G3[j0][i1]
    let c = (t1s0 - 1.0) * t2s0 * (t1t2s1 - 1.0) * G1[j1][i0] * G2[j1][i0] * G3[j0][i1]
    let d = (t1s0 - 1.0) * (t1t2s1 - 1.0) * G1[j1][i0] * G2[j1][j0] * G3[j0][i1]
    return float(s1) * (a - b - c + d) * inv_denom

proc Gam1(ph: int, k: int, G3: seq[seq[float]]): float =
    discard """
    Calculates the contribution of a single arc to theta,
    namely ph G3[k,k] - ph/2.

    Args:
        ph: the rotation number of the arc
        k: the index of the arc
        G3: the inverse of A at T1*T2

    Returns:
        The contribution of the arc.
    """
    return float(ph) * G3[k][k] - float(ph) / 2.0

# Rational points (T1, T2) at which Theta is evaluated. The
# first is the one used in the Bar-Natan / van der Veen
# paper, where Delta_i is non-zero up to 15 crossings; the
# other two are generic rationals held in reserve for a
# diagram that is singular at the first. Any triple of
# rationals avoiding the poles T2 = 1 and T1*T2 = 1 will do.
const THETA_EVAL_POINTS*: array[3, (float, float)] = [
    (22.0 / 7.0, 21.0 / 13.0),
    (17.0 / 5.0, 19.0 / 11.0),
    (-3.0 / 2.0, 5.0 / 3.0),
]

const THETA_PAPER_POINT* = THETA_EVAL_POINTS[0]

proc theta_eval*(xings: seq[array[0..2, int]], phi: seq[int],
                 T1, T2: float): tuple[delta1, delta2, delta3, theta: float] =
    discard """
    Given an upright presentation, evaluates
    Theta = (Delta, theta) numerically at (T1, T2).

    Args:
        xings: the crossings of the upright presentation
        phi: the rotation numbers of the arcs
        T1: the first evaluation point
        T2: the second evaluation point

    Returns:
        (Delta1, Delta2, Delta3, theta), where Delta_i is
        Delta evaluated at T1, T2 and T1*T2 respectively.

    Raises:
        ValueError: if A is singular at one of the three
            points, in which case Theta is undefined there
    """
    let N = phi.len
    var A1 = build_A(xings, N, T1)
    var A2 = build_A(xings, N, T2)
    var A3 = build_A(xings, N, T1 * T2)
    let (G1, det1) = invert_with_det(A1)
    let (G2, det2) = invert_with_det(A2)
    let (G3, det3) = invert_with_det(A3)
    let Delta1 = compute_delta(T1, det1, phi, xings)
    let Delta2 = compute_delta(T2, det2, phi, xings)
    let Delta3 = compute_delta(T1 * T2, det3, phi, xings)

    var theta = 0.0
    for c in xings:
        theta += F1(c[0], c[1], c[2], T1, T2, G1, G2, G3)
    for c1 in xings:
        for c2 in xings:
            theta += F2(c1[0], c1[1], c1[2], c2[0], c2[1], c2[2],
                        T1, T2, G1, G2, G3)
    for k in 0 ..< N:
        theta += Gam1(phi[k], k, G3)
    theta *= Delta1 * Delta2 * Delta3
    return (Delta1, Delta2, Delta3, theta)


proc theta_at_points*(
    link: Link[int],
    points: openArray[(float, float)]
): tuple[values: seq[float], defined: bool] =
    discard """
    Given a link, evaluates Theta at each of the given points.

    Args:
        link: the given link
        points: the evaluation points (T1, T2)

    Returns:
        (values, defined), where values holds five floats per
        point: Delta1, Delta2, Delta3, theta and
        theta0 = theta / (Delta1 Delta2 Delta3), in that
        order. When defined is false the evaluation failed at
        one of the points and values is all zeros, so a
        partial evaluation is never returned. A link with no
        components is reported as defined, with zeros.
    """
    let K = points.len
    var values = newSeq[float](5 * K)

    if link.link_components.len == 0:
        return (values, true)

    # Rebuild a canonical strand layout. Keep explicit signs when present and
    # fall back to deducing them from the PD code otherwise; an orientation
    # that cannot be made consistent leaves Theta undefined.
    var oriented_link = link
    if link.crossings.len > 0:
        try:
            oriented_link = link_from_PD_code(
                link.PD_code(),
                if link.signs.len > 0 and link.signs[0] != 0: link.signs
                else: @[],
                unlinked_unknot_components = link.unlinked_unknot_components)
        except ValueError, IndexDefect, RangeDefect:
            return (newSeq[float](5 * K), false)
        if oriented_link.crossings.len > 0 and oriented_link.signs[0] == 0:
            return (newSeq[float](5 * K), false)

    try:
        let (xings, phi) = pd_to_upright(oriented_link)
        for k in 0 ..< K:
            let (T1, T2) = points[k]
            let (d1, d2, d3, th) = theta_eval(xings, phi, T1, T2)
            let ddd = d1 * d2 * d3
            let theta0 = if abs(ddd) > 1e-15: th / ddd else: 0.0
            if classify(d1) in {fcInf, fcNegInf, fcNan} or
               classify(d2) in {fcInf, fcNegInf, fcNan} or
               classify(d3) in {fcInf, fcNegInf, fcNan} or
               classify(th) in {fcInf, fcNegInf, fcNan} or
               classify(theta0) in {fcInf, fcNegInf, fcNan}:
                return (newSeq[float](5 * K), false)
            values[5 * k]     = d1
            values[5 * k + 1] = d2
            values[5 * k + 2] = d3
            values[5 * k + 3] = th
            values[5 * k + 4] = theta0
    # pd_to_upright already turns vendored braid overflows into ValueError;
    # OverflowDefect stays as a direct guard anyway. IndexDefect and
    # RangeDefect cover degenerate diagrams, where a link component can still
    # point at a crossing that no longer exists.
    except ValueError, DivByZeroDefect, OverflowDefect, IndexDefect, RangeDefect:
        return (newSeq[float](5 * K), false)

    return (values, true)
