# Jones polynomial, ported from SageMath.
#
# Copyright (C) 2014 Miguel Angel Marco Buzunariz, Amit Jamadagni
#
# This file is a translation into Nim, with changes made in 2026, of
# Link._bracket and of the 'statesum' branch of Link.jones_polynomial in
# sage/knots/link.py, from SageMath. It is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 2 of
# the License, or (at your option) any later version. A translation is a
# derivative work, so anything this file is compiled into carries the same
# terms: features.nim imports it, which puts it inside the shared libraries
# this project builds. Ported against Sage 10.x.
#
# Both routines read the PD code only, so no diagram is rejected for want of a
# braid representative.

import "../spherogram-nim/src/links"

import alexander

proc invertVariable(p: Lpoly): Lpoly =
    discard """
    Substitutes t -> t^-1, that is Sage's f(A**-1).

    Args:
        p: the given Laurent polynomial

    Returns:
        The polynomial with every exponent negated.
    """
    if p.isZero: return p
    let hi = p.lo + p.coeffs.len - 1
    var c = newSeq[int](p.coeffs.len)
    for i in 0 ..< p.coeffs.len: c[p.coeffs.len - 1 - i] = p.coeffs[i]
    result = Lpoly(lo: -hi, coeffs: c)
    result.trim()

proc replaceFirst(cross: var seq[int], oldE, newE: int) =
    discard """
    Replaces the first occurrence of an edge label in a
    crossing, leaving any later occurrence alone. This is
    Sage's cross[cross.index(b)] = c.

    Args:
        cross: the four edge labels of a crossing, modified
            in place
        oldE: the label to replace
        newE: the label to replace it with
    """
    for i in 0 ..< cross.len:
        if cross[i] == oldE:
            cross[i] = newE
            return

proc bracketSage*(pdCode: seq[seq[int]]): Lpoly =
    discard """
    Calculates the Kauffman bracket of a diagram by recursion
    on its PD code, in Sage's bracket variable t. The
    recursion short circuits on kinks and on closed loops,
    which makes it near linear on structured diagrams where a
    plain state sum over 2^n states is not.

    The bracket is an invariant of the diagram, not of the
    link: it is not preserved by a Reidemeister I move.
    jonesPolySageInA applies the writhe correction that makes
    it one.

    Args:
        pdCode: the PD code of the diagram

    Returns:
        The Kauffman bracket as a Laurent polynomial in t.
    """
    if pdCode.len == 0: return lpConst(1)
    if pdCode.len == 1:
        if pdCode[0][0] == pdCode[0][3]: return lpMono(-1, -3)
        else: return lpMono(-1, 3)

    let cross = pdCode[0]
    var rest = newSeq[seq[int]](pdCode.len - 1)
    for i in 1 ..< pdCode.len: rest[i - 1] = pdCode[i]
    let (a, b, c, d) = (cross[0], cross[1], cross[2], cross[3])

    if a == d and c == b and rest.len > 0:
        # a closed loop, contributing -A^2 - A^-2 in Sage's variable
        return (lpMono(1, -1) + lpMono(1, -5)) * bracketSage(rest)
    elif a == b and c == d and rest.len > 0:
        return (lpMono(1, 1) + lpMono(1, 5)) * bracketSage(rest)
    elif a == d:
        for i in 0 ..< rest.len: replaceFirst(rest[i], b, c)
        return lpMono(-1, -3) * bracketSage(rest)
    elif a == b:
        for i in 0 ..< rest.len: replaceFirst(rest[i], c, d)
        return lpMono(-1, 3) * bracketSage(rest)
    elif c == d:
        for i in 0 ..< rest.len: replaceFirst(rest[i], b, a)
        return lpMono(-1, 3) * bracketSage(rest)
    elif c == b:
        for i in 0 ..< rest.len: replaceFirst(rest[i], d, a)
        return lpMono(-1, -3) * bracketSage(rest)
    else:
        var rest2 = newSeq[seq[int]](rest.len)
        for i in 0 ..< rest.len: rest2[i] = rest[i]
        for i in 0 ..< rest.len:
            replaceFirst(rest[i], b, a)
            replaceFirst(rest[i], c, d)
        for i in 0 ..< rest2.len:
            replaceFirst(rest2[i], b, c)
            replaceFirst(rest2[i], d, a)
        return lpMono(1, 1) * bracketSage(rest) + lpMono(1, -1) * bracketSage(rest2)

proc jonesPolySageInA*[T](link: Link[T]): Lpoly =
    discard """
    Given a link, calculates its skein normalised Jones
    polynomial as a Laurent polynomial in A: the Kauffman
    bracket multiplied by (-t)^(-3w) for w the writhe, then
    with t replaced by A^-1. Each additional split unknot
    contributes -A^2 - A^-2. The empty state is assigned 1,
    as is a single unknot.

    Args:
        link: the given link

    Returns:
        The Jones polynomial as a Laurent polynomial in A.
    """
    let pd = link.PD_code()
    var pdSeq = newSeq[seq[int]](pd.len)
    for i, cr in pd: pdSeq[i] = @[cr[0], cr[1], cr[2], cr[3]]
    result = lpConst(1)
    if pdSeq.len > 0:
        let bracket = bracketSage(pdSeq)
        let w = writhe(link)
        # (-t)^(-3w) = (-1)^w t^(-3w), since (-1)^(3w) = (-1)^w
        let sign = if (w and 1) == 0: 1 else: -1
        result = invertVariable(lpMono(sign, -3 * w) * bracket)
    let extraCircles = if pdSeq.len == 0:
                           max(0, link.unlinked_unknot_components - 1)
                       else:
                           link.unlinked_unknot_components
    let delta = lpMono(-1, 2) + lpMono(-1, -2)
    for _ in 0 ..< extraCircles:
        result = result * delta

proc jonesInSqrtT*(p: Lpoly): Lpoly =
    discard """
    Converts a polynomial in A to the Jones polynomial V_L in
    the variable x = sqrt(t).

    This project retains the inverse-variable convention
    relative to Sage: V_feature(t) = V_Sage(t^-1). It uses
    A = t^(-1/4), whereas Sage substitutes A = t^(1/4) into
    its skein-normalised result. Thus x = A^-2: an A-exponent
    e becomes -e/2 and the coefficient order reverses. For
    example A^4 + A^12 - A^16 becomes -x^-8 + x^-6 + x^-2.

    Halving is exact because the writhe-corrected polynomial
    of an oriented diagram has only even A-exponents.

    Args:
        p: the Jones polynomial as a Laurent polynomial in A

    Returns:
        The Jones polynomial as a Laurent polynomial in
        x = sqrt(t).

    Raises:
        ValueError: if p has a term at an odd A-exponent,
            which would mean the bracket recursion had gone
            wrong
    """
    if p.isZero: return p
    var q = p
    q.trim()
    for i in 0 ..< q.coeffs.len:
        if ((q.lo + i) and 1) != 0 and q.coeffs[i] != 0:
            raise newException(ValueError,
                "Jones polynomial has a term at the odd A-exponent " & $(q.lo + i) &
                ", so it cannot be written in x = sqrt(t)")
    # the highest A-exponent carries the lowest x-exponent
    let hi = q.lo + q.coeffs.len - 1
    var c = newSeq[int]((q.coeffs.len + 1) div 2)
    for i in 0 ..< q.coeffs.len:
        let e = q.lo + i
        if (e and 1) == 0:
            c[(hi - e) div 2] = q.coeffs[i]
    result = Lpoly(lo: -(hi div 2), coeffs: c)
    result.trim()
