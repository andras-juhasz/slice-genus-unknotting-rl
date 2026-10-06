# Alexander polynomial of a knot or link from its PD code, plus the Laurent
# polynomial arithmetic the other polynomial modules use.
#
# The PD convention is the one SnapPy, Spherogram and Regina use, so
# link.PD_code() can be fed straight in. Orientation, crossing signs and
# over-arcs are re-derived from the code alone; link.signs is never read.
# Reading only the PD code is what lets this run on diagrams that no braid
# algorithm will accept.
#
# ALGORITHM. The Alexander matrix of the Wirtinger presentation, via Fox's
# free differential calculus: one row per crossing, one column per over-arc,
# entries in Z[t]. At a crossing with under-in arc I, under-out arc J and
# over-arc K,
#
#     positive crossing:  I += t,  J += -1,  K += (1 - t)
#     negative crossing:  I += 1,  J += -t,  K += (t - 1)
#
# For a knot, any (n-1)x(n-1) minor is +- t^k Delta(t). For a link a single
# minor is not an invariant, so one column is deleted and the gcd of the n
# maximal minors is taken (the order of the first elementary ideal). Either
# way the result is normalised up to +- t^k.
#
# Minor determinants are exact: interpolated modulo several primes and
# reconstructed by the Chinese Remainder Theorem. A bound on all determinant
# coefficients certifies that signed reconstruction is unique; diagrams for
# which the fixed modulus is insufficient are refused. The integer polynomial
# gcd of the link branch uses a primitive Euclidean pseudo-remainder sequence,
# retaining the gcd of the integer contents.
#
# The Kauffman bracket state sum at the end of the file is a second,
# independent implementation of the Jones polynomial, kept for cross-checking.
# The observation features take Jones from jones_port instead, which is about
# ten times faster.
#
# Regression examples are in test/test_alexander.nim. The Alexander convention
# is the one-variable order of the first elementary ideal, up to +- t^k;
# a split link with more than one component has polynomial zero. A PD code
# cannot record crossingless components: callers must handle those separately.
#
# References:
#   [fox1953free]        R. H. Fox, "Free differential calculus I", Ann. of
#                        Math. 57 (1953) 547-560.
#   [crowellfox1963]     R. H. Crowell and R. H. Fox, "Introduction to Knot
#                        Theory", Ginn & Co. (1963), Ch. VII-VIII.
#   [kauffman1987state]  L. H. Kauffman, "State models and the Jones
#                        polynomial", Topology 26 (1987) 395-407.
#   [gathengerhard2013]  J. von zur Gathen and J. Gerhard, "Modern Computer
#                        Algebra", 3rd ed., CUP (2013).

import std/sequtils
import std/[algorithm, tables]

# Laurent polynomials  c[lo] t^lo + ... (integer coefficients)

discard """
A Laurent polynomial with integer coefficients, stored
densely: lo is the exponent of the first stored coefficient
and coeffs[i] multiplies t^(lo+i).
"""
type Lpoly* = object
    lo*: int
    coeffs*: seq[int]

proc trim*(p: var Lpoly) =
    discard """
    Drops leading and trailing zero coefficients, putting the
    polynomial in canonical form.

    Args:
        p: the polynomial to trim, modified in place
    """
    var a = 0
    while a < p.coeffs.len and p.coeffs[a] == 0: inc a
    if a == p.coeffs.len:
        p.lo = 0; p.coeffs = @[]; return
    var b = p.coeffs.len - 1
    while b >= 0 and p.coeffs[b] == 0: dec b
    p.lo += a
    p.coeffs = p.coeffs[a .. b]

proc lpZero*(): Lpoly = Lpoly(lo: 0, coeffs: @[])
proc lpConst*(c: int): Lpoly = Lpoly(lo: 0, coeffs: @[c])
proc lpMono*(c, e: int): Lpoly = Lpoly(lo: e, coeffs: @[c])
proc isZero*(p: Lpoly): bool = p.coeffs.len == 0

proc `+`*(a, b: Lpoly): Lpoly =
    if a.isZero: return b
    if b.isZero: return a
    let lo = min(a.lo, b.lo)
    let hi = max(a.lo + a.coeffs.len - 1, b.lo + b.coeffs.len - 1)
    var c = newSeq[int](hi - lo + 1)
    for i in 0 ..< a.coeffs.len: c[a.lo + i - lo] += a.coeffs[i]
    for i in 0 ..< b.coeffs.len: c[b.lo + i - lo] += b.coeffs[i]
    result = Lpoly(lo: lo, coeffs: c); result.trim()

proc `-`*(a: Lpoly): Lpoly =
    result = a
    for i in 0 ..< result.coeffs.len: result.coeffs[i] = -result.coeffs[i]

proc `-`*(a, b: Lpoly): Lpoly = a + (-b)

proc `*`*(a, b: Lpoly): Lpoly =
    if a.isZero or b.isZero: return lpZero()
    var c = newSeq[int](a.coeffs.len + b.coeffs.len - 1)
    for i in 0 ..< a.coeffs.len:
        if a.coeffs[i] != 0:
            for j in 0 ..< b.coeffs.len:
                c[i + j] += a.coeffs[i] * b.coeffs[j]
    result = Lpoly(lo: a.lo + b.lo, coeffs: c); result.trim()

proc normaliseUnit*(p: Lpoly): Lpoly =
    discard """
    Calculates the canonical representative of a polynomial up
    to multiplication by +- t^k: the lowest exponent is
    shifted to 0 and the leading coefficient made positive.

    Args:
        p: the given polynomial

    Returns:
        The canonical representative of p.
    """
    result = p; result.trim()
    if result.isZero: return
    result.lo = 0
    if result.coeffs[^1] < 0:
        for i in 0 ..< result.coeffs.len: result.coeffs[i] = -result.coeffs[i]

# From PD code to oriented diagram (signs, over-arcs, components)

discard """
An oriented diagram recovered from a PD code, where n is the
number of crossings, pos[c] the four edge labels of crossing
c counterclockwise from under-in, sign the sign of each
crossing, arcOf the over-arc attached to each of the four
ends of a crossing, nArcs the number of over-arcs (which are
the Wirtinger generators) and nComp the number of link
components.
"""
type Diagram = object
    n: int
    pos: seq[array[4, int]]
    sign: seq[int]
    arcOf: seq[array[4, int]]
    nArcs: int
    nComp: int

type PolyUF = object
    parent: seq[int]
proc initPolyUF(n: int): PolyUF =
    result.parent = newSeq[int](n)
    for i in 0 ..< n: result.parent[i] = i
proc find(u: var PolyUF, x: int): int =
    var r = x
    while u.parent[r] != r: r = u.parent[r]
    var y = x
    while u.parent[y] != y:
        let nxt = u.parent[y]; u.parent[y] = r; y = nxt
    r
proc union(u: var PolyUF, a, b: int) = u.parent[u.find(a)] = u.find(b)

proc parsePD(crossings: seq[seq[int]]): Diagram =
    let n = crossings.len
    result.n = n
    result.pos = newSeq[array[4, int]](n)
    # remap arbitrary edge labels to 0 .. 2n-1
    var labels: seq[int] = @[]
    for c in crossings:
        for x in c: labels.add x
    labels = labels.deduplicate()
    labels.sort()
    var remap = initOrderedTable[int, int]()
    for i, l in labels: remap[l] = i
    for ci in 0 ..< n:
        for k in 0 .. 3:
            result.pos[ci][k] = remap[crossings[ci][k]]
    let nEdges = labels.len   # = 2n

    # classify every edge-END as IN (entering a crossing) or OUT (leaving):
    # under-strand position 0 is IN, position 2 is OUT; the over-strand's two
    # ends (positions 1, 3) are one IN and one OUT, found by propagating two
    # constraints to a fixpoint: each edge joins one IN to one OUT, and the
    # over-strand passes straight through (positions 1 and 3 are opposite).
    var endIO = newSeq[array[4, int]](n)
    for c in 0 ..< n:
        endIO[c][0] = 1     # under-in
        endIO[c][2] = -1    # under-out
    var occ = newSeq[seq[(int, int)]](nEdges)  # edge -> list of (crossing, k)
    for c in 0 ..< n:
        for k in 0 .. 3:
            occ[result.pos[c][k]].add (c, k)
    var changed = true
    while changed:
        changed = false
        for e in 0 ..< nEdges:                  # (a) edge constraint
            let (c0, k0) = occ[e][0]
            let (c1, k1) = occ[e][1]
            if endIO[c0][k0] != 0 and endIO[c1][k1] == 0:
                endIO[c1][k1] = -endIO[c0][k0]; changed = true
            elif endIO[c1][k1] != 0 and endIO[c0][k0] == 0:
                endIO[c0][k0] = -endIO[c1][k1]; changed = true
        for c in 0 ..< n:                       # (b) over-strand pass-through
            if endIO[c][1] != 0 and endIO[c][3] == 0:
                endIO[c][3] = -endIO[c][1]; changed = true
            elif endIO[c][3] != 0 and endIO[c][1] == 0:
                endIO[c][1] = -endIO[c][3]; changed = true

    # crossing signs: positive iff the over-strand enters at position 3.
    result.sign = newSeq[int](n)
    for c in 0 ..< n:
        let overInAt3 = (endIO[c][3] == 1)
        result.sign[c] = (if overInAt3: 1 else: -1)

    # over-arcs (Wirtinger generators): a maximal strand that stays ON TOP,
    # broken only where it goes UNDER (positions 0, 2). Union the two over-ends
    # of every crossing (positions 1, 3) and the two ends of every edge.
    var uf = initPolyUF(4 * n)
    proc nd(c, k: int): int = 4 * c + k
    for c in 0 ..< n:
        uf.union(nd(c, 1), nd(c, 3))
    for e in 0 ..< nEdges:
        let (c0, k0) = occ[e][0]
        let (c1, k1) = occ[e][1]
        uf.union(nd(c0, k0), nd(c1, k1))
    var arcId = initOrderedTable[int, int]()
    result.arcOf = newSeq[array[4, int]](n)
    var na = 0
    for c in 0 ..< n:
        for k in 0 .. 3:
            let r = uf.find(nd(c, k))
            if not arcId.hasKey(r):
                arcId[r] = na; inc na
            result.arcOf[c][k] = arcId[r]
    result.nArcs = na

    # number of link components: each strand runs straight through every
    # crossing (k to (k+2) mod 4); count the cycles after gluing edges.
    var ufc = initPolyUF(4 * n)
    for c in 0 ..< n:
        ufc.union(4 * c + 0, 4 * c + 2)
        ufc.union(4 * c + 1, 4 * c + 3)
    for e in 0 ..< nEdges:
        let (c0, k0) = occ[e][0]
        let (c1, k1) = occ[e][1]
        ufc.union(4 * c0 + k0, 4 * c1 + k1)
    var seen = initOrderedTable[int, bool]()
    for c in 0 ..< n:
        for k in 0 .. 3:
            seen[ufc.find(4 * c + k)] = true
    result.nComp = (if n == 0: 1 else: seen.len)

# Alexander polynomial (Wirtinger matrix + exact CRT determinant)

const PRIMES = [1000003, 1000033, 1000037]   # product ~1e18 < 2^63
const CRT_MODULUS = PRIMES[0] * PRIMES[1] * PRIMES[2]

proc mmod(a, p: int): int = ((a mod p) + p) mod p
proc mpow(a, e, p: int): int =
    result = 1
    var b = mmod(a, p)
    var ee = e
    while ee > 0:
        if (ee and 1) == 1: result = (result * b) mod p
        b = (b * b) mod p
        ee = ee shr 1
proc minv(a, p: int): int = mpow(mmod(a, p), p - 2, p)  # p prime

proc detModP(mat: seq[seq[int]], p: int): int =
    discard """
    Calculates the determinant of an integer matrix modulo a
    prime, by Gaussian elimination.

    Args:
        mat: the given integer matrix
        p: the modulus, which must be prime

    Returns:
        The determinant of mat modulo p.
    """
    let m = mat.len
    if m == 0: return 1
    var a = mat
    var det = 1
    for col in 0 ..< m:
        var piv = -1
        for r in col ..< m:
            if mmod(a[r][col], p) != 0: piv = r; break
        if piv == -1: return 0
        if piv != col:
            swap(a[piv], a[col]); det = mmod(-det, p)
        let inv = minv(a[col][col], p)
        det = (det * mmod(a[col][col], p)) mod p
        for r in col + 1 ..< m:
            let f = (mmod(a[r][col], p) * inv) mod p
            if f != 0:
                for cc in col ..< m:
                    a[r][cc] = mmod(a[r][cc] - f * a[col][cc], p)
    mmod(det, p)

proc crtCoefficientBoundIsSafe(ent: seq[seq[(int, int)]]): bool =
    # The l1 norm of det(A(t)) is at most the product of the row sums
    # of |a_ij| + |b_ij|. Thus this also bounds each coefficient. Saturate
    # at limit+1, including before abs(), so even low(int) is safe.
    const limit = CRT_MODULUS div 2
    var bound = 1
    for row in ent:
        var rowBound = 0
        for (a, b) in row:
            for coefficient in [a, b]:
                if coefficient < -limit or coefficient > limit:
                    rowBound = limit + 1
                elif rowBound <= limit:
                    let magnitude = abs(coefficient)
                    rowBound = (if magnitude > limit - rowBound:
                                    limit + 1 else: rowBound + magnitude)
        if rowBound == 0: return true  # a zero row makes the determinant zero
        if bound <= limit:
            bound = (if rowBound > limit div bound:
                         limit + 1 else: bound * rowBound)
    bound <= limit

proc detPolyCRT*(ent: seq[seq[(int, int)]]): Lpoly =
    discard """
    Calculates the exact determinant of a square matrix whose
    entries are a + b t, as a polynomial in t. The
    determinant is evaluated modulo several primes, Newton
    interpolated modulo each, and the integer coefficients
    reconstructed by the Chinese Remainder Theorem, so that
    intermediate values never overflow a 64-bit integer.

    Args:
        ent: the matrix entries, each (constant, coefficient
            of t)

    Returns:
        The determinant as a polynomial in t.

    Raises:
        ValueError: if the input is not square, its degree is
            too large for interpolation, or the coefficient
            bound cannot certify reconstruction with the
            fixed CRT modulus (even if the true result fits)
    """
    let m = ent.len
    if m == 0: return lpConst(1)
    for row in ent:
        if row.len != m:
            raise newException(ValueError, "detPolyCRT: matrix must be square")
    if m >= PRIMES[0]:
        raise newException(ValueError, "detPolyCRT: too many interpolation points")
    if not crtCoefficientBoundIsSafe(ent):
        raise newException(ValueError,
            "detPolyCRT: coefficient bound exceeds the fixed CRT modulus; " &
            "exact signed reconstruction cannot be certified")
    let npts = m + 1                       # deg(det) <= m
    var coeffsModP = newSeq[seq[int]](PRIMES.len)
    for pi, p in PRIMES:
        var ys = newSeq[int](npts)
        for x in 0 ..< npts:
            var mat = newSeq[seq[int]](m)
            for i in 0 ..< m:
                mat[i] = newSeq[int](m)
                for j in 0 ..< m:
                    let (a0, b0) = ent[i][j]
                    mat[i][j] = mmod(mmod(a0, p) + mmod(b0, p) * x, p)
            ys[x] = detModP(mat, p)
        var coef = ys
        for j in 1 ..< npts:
            for i in countdown(npts - 1, j):
                let denom = minv(mmod(j, p), p)
                coef[i] = (mmod(coef[i] - coef[i - 1], p) * denom) mod p
        var std = newSeq[int](npts)
        std[0] = coef[npts - 1]
        var deg = 0
        for k in countdown(npts - 2, 0):
            var nw = newSeq[int](npts)
            for i in 0 .. deg:
                nw[i + 1] = mmod(nw[i + 1] + std[i], p)
                nw[i] = mmod(nw[i] - k * std[i], p)
            inc deg
            nw[0] = mmod(nw[0] + coef[k], p)
            std = nw
        coeffsModP[pi] = std
    var res = Lpoly(lo: 0, coeffs: newSeq[int](npts))
    var modProd = 1
    var crtAcc = newSeq[int](npts)
    for pi, p in PRIMES:
        if pi == 0:
            for i in 0 ..< npts: crtAcc[i] = coeffsModP[0][i]
            modProd = p
        else:
            let inv = minv(mmod(modProd, p), p)
            for i in 0 ..< npts:
                let diff = (mmod(coeffsModP[pi][i] - crtAcc[i], p) * inv) mod p
                crtAcc[i] = crtAcc[i] + modProd * diff
            modProd = modProd * p
    for i in 0 ..< npts:
        var v = crtAcc[i] mod modProd
        if v > modProd div 2: v -= modProd
        res.coeffs[i] = v
    res.trim()
    res

# integer polynomial gcd (primitive Euclidean PRS), for the link branch
proc igcd(a, b: int): int =
    var x = abs(a); var y = abs(b)
    while y != 0: (x, y) = (y, x mod y)
    x

proc content(p: Lpoly): int =
    result = 0
    for c in p.coeffs: result = igcd(result, c)
    if result == 0: result = 1

proc primitive(p: Lpoly): Lpoly =
    result = p; result.trim()
    if result.isZero: return
    let g = content(result)
    for i in 0 ..< result.coeffs.len: result.coeffs[i] = result.coeffs[i] div g
    if result.coeffs[^1] < 0:
        for i in 0 ..< result.coeffs.len: result.coeffs[i] = -result.coeffs[i]

proc degree*(p: Lpoly): int =
    if p.isZero: return -1
    p.lo + p.coeffs.len - 1

proc pseudoRem(a, b: Lpoly): Lpoly =
    var r = a; r.trim()
    var bb = b; bb.trim()
    let db = degree(bb)
    let lcb = bb.coeffs[^1]
    var dr = degree(r)
    var guard = 0
    while (not r.isZero) and dr >= db:
        let lcr = r.coeffs[^1]
        r = r * lpConst(lcb) - lpMono(lcr, dr - db) * bb
        r.trim()
        dr = degree(r)
        inc guard
        # The subtraction cancels the leading term by construction (both sides
        # have leading coefficient lcr*lcb at degree dr), so deg(r) strictly
        # decreases and the loop terminates. The guard only fires if that is
        # broken, by integer overflow in the coefficients say. It raises rather
        # than returning something that is not a remainder, which would corrupt
        # the gcd of the link branch.
        if guard > 10000:
            raise newException(ValueError,
                "pseudoRem: no degree decrease after " & $guard & " steps, so the " &
                "remainder sequence is not converging")
    r

proc polyGCD(a, b: Lpoly): Lpoly =
    if a.isZero: return b
    if b.isZero: return a
    let commonContent = igcd(content(a), content(b))
    var x = primitive(a)
    var y = primitive(b)
    while not y.isZero:
        let r = pseudoRem(x, y)
        x = y
        y = primitive(r)
    primitive(x) * lpConst(commonContent)

proc alexanderPoly*(crossings: seq[seq[int]]): Lpoly =
    discard """
    Given the PD code of a diagram, calculates its Alexander
    polynomial. For a knot this is any maximal minor of the
    Alexander matrix; for a link it is the gcd of all of them
    after deleting a column.

    Args:
        crossings: the PD code of the diagram

    Returns:
        The Alexander polynomial Delta(t), normalised up to
        +- t^k.
    """
    let D = parsePD(crossings)
    let n = D.n
    if n == 0: return lpConst(1)
    let na = D.nArcs
    # Each component with no undercrossings adds an extra Wirtinger arc.
    # Such an all-over component is a split unknot, so Delta is zero;
    # its orientation also cannot be recovered from under-in ends alone.
    if na > n: return lpZero()
    var A = newSeq[seq[(int, int)]](n)
    for r in 0 ..< n: A[r] = newSeq[(int, int)](na)
    for c in 0 ..< n:
        let I = D.arcOf[c][0]   # under-in
        let J = D.arcOf[c][2]   # under-out
        let K = D.arcOf[c][1]   # over (positions 1 and 3 share the same arc)
        if D.sign[c] == 1:
            A[c][I][1] += 1                       # + t
            A[c][J][0] += -1                      # - 1
            A[c][K][0] += 1; A[c][K][1] += -1     # + (1 - t)
        else:
            A[c][I][0] += 1                       # + 1
            A[c][J][1] += -1                      # - t
            A[c][K][0] += -1; A[c][K][1] += 1     # + (t - 1)
    let m = n - 1
    if m == 0: return lpConst(1)

    if D.nComp == 1:
        # KNOT: any (n-1)x(n-1) minor is +- t^k * Delta(t).
        var ent = newSeq[seq[(int, int)]](m)
        for i in 0 ..< m:
            ent[i] = newSeq[(int, int)](m)
            for j in 0 ..< m: ent[i][j] = A[i][j]
        return normaliseUnit(detPolyCRT(ent))
    else:
        # LINK: gcd of the n maximal minors after deleting the last column
        # (Crowell-Fox), a diagram invariant up to +- t^k.
        var g = lpZero()
        for skip in 0 ..< n:
            var ent = newSeq[seq[(int, int)]](m)
            var ri = 0
            for i in 0 ..< n:
                if i == skip: continue
                ent[ri] = newSeq[(int, int)](m)
                for j in 0 ..< m: ent[ri][j] = A[i][j]
                inc ri
            let minor = detPolyCRT(ent)
            g = (if g.isZero: minor else: polyGCD(g, minor))
        return normaliseUnit(g)



# Jones polynomial (Kauffman bracket state sum, 2^n)
# Independent implementation, slower than the one in 
# jones_port, so we use the other one for the 
# feature computation

proc jonesInA*(crossings: seq[seq[int]]): Lpoly =
    discard """
    Given the PD code of a diagram, calculates its Kauffman
    normalised Jones polynomial by a state sum over all 2^n
    smoothings.

    This legacy routine retains its existing smoothing and
    writhe conventions. Its jonesPoly wrapper converts an
    A-exponent e to an x-exponent e/2, giving the inverse
    variable of Sage's skein-normalised Jones convention.
    The observation feature uses jones_port instead. This
    PD-only routine omits crossingless components and does
    not resolve orientations of all-over components.

    Args:
        crossings: the PD code of the diagram

    Returns:
        The legacy normalised state sum as a Laurent polynomial in A.
    """
    let D = parsePD(crossings)
    let n = D.n
    let nEdges = 2 * n
    if n == 0: return lpConst(1)
    var w = 0
    for c in 0 ..< n: w += D.sign[c]
    let delta = Lpoly(lo: -2, coeffs: @[-1, 0, 0, 0, -1])   # -A^2 - A^-2
    var occ = newSeq[seq[int]](nEdges)
    for c in 0 ..< n:
        for k in 0 .. 3:
            occ[D.pos[c][k]].add (4 * c + k)
    var bracket = lpZero()
    for state in 0 ..< (1 shl n):
        var uf = initPolyUF(4 * n)
        for e in 0 ..< nEdges:
            uf.union(occ[e][0], occ[e][1])
        var aCount = 0
        for c in 0 ..< n:
            if ((state shr c) and 1) == 0:
                uf.union(4 * c + 0, 4 * c + 1)   # A-smoothing: (0,1) and (2,3)
                uf.union(4 * c + 2, 4 * c + 3)
                inc aCount
            else:
                uf.union(4 * c + 0, 4 * c + 3)   # B-smoothing: (0,3) and (1,2)
                uf.union(4 * c + 1, 4 * c + 2)
        let bCount = n - aCount
        var seen = initOrderedTable[int, bool]()
        var loops = 0
        for nodeId in 0 ..< 4 * n:
            let r = uf.find(nodeId)
            if not seen.hasKey(r): seen[r] = true; inc loops
        var term = lpMono(1, aCount - bCount)
        for _ in 0 ..< (loops - 1): term = term * delta
        bracket = bracket + term
    let signFac = (if (w and 1) == 0: 1 else: -1)
    result = lpMono(signFac, -3 * w) * bracket

proc jonesPoly*(crossings: seq[seq[int]]): Lpoly =
    discard """
    Given the PD code of a diagram, calculates its Jones
    polynomial V_L in the variable x = sqrt(t).

    The historical exponent map e_A -> e_A/2 is retained:
    the result uses the inverse variable of Sage's
    skein-normalised Jones polynomial. This is not a change
    of link orientation. See jonesInA for this PD-only
    routine's component/orientation limitations.

    Args:
        crossings: the PD code of the diagram

    Returns:
        The Jones polynomial as a Laurent polynomial in
        x = sqrt(t).
    """
    let inA = jonesInA(crossings)
    if inA.isZero: return lpZero()
    var terms: seq[(int, int)] = @[]
    for i in 0 ..< inA.coeffs.len:
        let cf = inA.coeffs[i]
        if cf == 0: continue
        let eA = inA.lo + i
        doAssert eA mod 2 == 0, "unexpected odd A-exponent " & $eA
        terms.add (eA div 2, cf)
    var minE = high(int)
    for (e, _) in terms: minE = min(minE, e)
    var maxE = low(int)
    for (e, _) in terms: maxE = max(maxE, e)
    var res = Lpoly(lo: minE, coeffs: newSeq[int](maxE - minE + 1))
    for (e, cf) in terms: res.coeffs[e - minE] += cf
    res.trim()
    res
