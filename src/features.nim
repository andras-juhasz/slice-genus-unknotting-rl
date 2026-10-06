import std/sequtils
import std/strformat
import std/math
import std/algorithm
import std/tables

import arraymancer
import "../spherogram-nim/src/links"
import "../spherogram-nim/src/invariants"
import "../spherogram-nim/src/seifert"

import utils
import theta
import alexander
import jones_port

discard """
Base class for the observation features. Each subclass turns
a link into one or more named tensors whose shapes depend
only on the shape parameter, so that the observation of the
network keeps a fixed size.
"""
type TensorFeature* = ref object of RootObj

method cls_name*(self: TensorFeature): string {.base.} =
    discard """
    Returns the name of the feature. This method should be
    overridden by subclasses of TensorFeature.

    Returns:
        The name of the feature.
    """
    return "Tensor Feature"
method need_unsimplified_diagram*(self: TensorFeature): bool {.base.} =
    discard """
    States which link compute_features passes to this
    feature: the unsimplified diagram, or the globally
    simplified one. A feature that reads the diagram itself
    needs the first, while a topological invariant gets the
    same answer from the second more cheaply. Getting this
    wrong is silent, so every feature answers explicitly
    rather than inheriting the default.

    Returns:
        Whether the feature needs the unsimplified diagram.
    """
    return false

proc `$`*(self: TensorFeature): string =
    return &"Tensor Feature: {self.cls_name()}"

method from_link*(
    self: TensorFeature,
    link: Link[int],
    shape: array[0..1, int]
): seq[(string, Tensor[float])] {.base.} =
    discard """
    Evaluates the feature on a link. This method must
    be overridden by subclasses of TensorFeature.

    Args:
        link: the link on which we are evaluating the
            feature
        shape: a pair of integers; shape[0] means the
            maximum number of crossings allowed and
            shape[1] means the maximum number of
            link components allowed

    Returns:
        A list of (feature name, feature value) pairs,
        where the feature name is a string and the
        feature value is an n-dimensional array, whose
        dimension must only depend on shape (in
        particular, it cannot depend on link).
    """
    raise newException(AssertionDefect, "not implemented")

discard """
Emits the determinant and the signature of the link as two
scalars, or the -1000 sentinel where they are undefined.
"""
type DeterminantAndSignature* = ref object of TensorFeature

method cls_name*(self: DeterminantAndSignature): string =
    return "Determinant and Signature"

method need_unsimplified_diagram*(self: DeterminantAndSignature): bool = false

method from_link*(self: DeterminantAndSignature,
                  link: Link[int],
                  shape: array[0..1, int]): seq[(string, Tensor[float])] =
    # Goeritz data can be undefined on degenerate diagrams. The
    # color determinant is a fallback; the signature has no such fallback.
    var det = -1000.0
    var sig = -1000.0
    try:
        det = float(link.determinant())
        sig = float(link.signature())
    except ValueError, IndexDefect:
        try:
            det = float(link.determinant("color"))
        except ValueError, IndexDefect, RangeDefect:
            det = -1000.0
        sig = -1000.0
    return @[("determinant", [det].toTensor()),
             ("signature", [sig].toTensor())]

discard """
Emits the matrix of pairwise linking numbers of the link
components, padded to a square of side shape[1].
"""
type LinkingMatrix* = ref object of TensorFeature

method cls_name*(self: LinkingMatrix): string =
    return "Linking Matrix"

method need_unsimplified_diagram*(self: LinkingMatrix): bool = true


method from_link*(self: LinkingMatrix,
                  link: Link[int],
                  shape: array[0..1, int]): seq[(string, Tensor[float])] =
    var lm: Tensor[int]
    if link.link_components.len == 0:
        lm = zeros[int]([0, 0])
    else:
        lm = link.linking_matrix().toTensor()
    return @[("linking_matrix",
              pad(lm,
                  [shape[1], shape[1]], -1000).astype(float))]

discard """
Emits the linking numbers of the link aggregated by
connected component label (CCL), padded to a square of side
shape[1].
"""
type CCLLinkingMatrix* = ref object of TensorFeature

method cls_name*(self: CCLLinkingMatrix): string =
    return "CCL Linking Matrix"

method need_unsimplified_diagram*(self: CCLLinkingMatrix): bool = true

method from_link*(self: CCLLinkingMatrix,
                  link: Link[int],
                  shape: array[0..1, int]): seq[(string, Tensor[float])] =
    # Get the number of CCLs from extra_info values
    var ccl_set: seq[int] = @[]
    for comp in link.link_components:
        if comp.extra_info notin ccl_set:
            ccl_set.add(comp.extra_info)
    let num_ccls = ccl_set.len

    if num_ccls == 0:
        return @[("ccl_linking_matrix",
                  pad(zeros[int]([0, 0]),
                      [shape[1], shape[1]], -1000).astype(float))]

    # Build strand_to_comp and strand_to_ccl mappings
    let num_crossings = link.crossings.len
    var strand_to_comp = newSeqWith(num_crossings, [-1, -1, -1, -1])
    var comp_to_ccl = newSeq[int](link.link_components.len)

    for comp_index in 0 ..< link.link_components.len:
        let comp = link.link_components[comp_index]
        comp_to_ccl[comp_index] = comp.extra_info
        var cur_c = comp.crossing
        var cur_s = comp.strand_index
        while strand_to_comp[cur_c][cur_s] == -1:
            strand_to_comp[cur_c][cur_s] = comp_index
            strand_to_comp[cur_c][(cur_s+2) mod 4] = comp_index
            (cur_c, cur_s) = link.next_strand((cur_c, cur_s))

    # First compute the standard linking matrix between components
    let num_components = link.link_components.len
    var comp_linking_matrix = newSeqWith(num_components, newSeqWith(num_components, 0))
    for c in 0 ..< num_crossings:
        let strands = strand_to_comp[c]
        let sign = link.signs[c]
        if strands[0] != strands[1]:
            comp_linking_matrix[strands[0]][strands[1]] += sign
            comp_linking_matrix[strands[1]][strands[0]] += sign
    for i in 0 ..< num_components:
        for j in 0 ..< num_components:
            comp_linking_matrix[i][j] = comp_linking_matrix[i][j] div 2

    # Map CCLs to consecutive indices 0..num_ccls-1
    var ccl_to_index: seq[(int, int)] = @[]
    for i in 0 ..< ccl_set.len:
        let ccl = ccl_set[i]
        ccl_to_index.add((ccl, i))

    proc get_ccl_index(ccl: int): int =
        for (c, idx) in ccl_to_index:
            if c == ccl:
                return idx
        return -1

    # Aggregate linking numbers by CCL pairs
    var ccl_linking_matrix = newSeqWith(num_ccls, newSeqWith(num_ccls, 0))
    for i in 0 ..< num_components:
        for j in 0 ..< num_components:
            if i != j:  # Skip self-linking of same component
                let ccl_i = get_ccl_index(comp_to_ccl[i])
                let ccl_j = get_ccl_index(comp_to_ccl[j])
                if ccl_i <= ccl_j:
                    ccl_linking_matrix[ccl_i][ccl_j] += comp_linking_matrix[i][j]
                # Added once only: comp_linking_matrix is already symmetric

    # Make the matrix symmetric (copy upper triangle to lower)
    for i in 0 ..< num_ccls:
        for j in i+1 ..< num_ccls:
            ccl_linking_matrix[j][i] = ccl_linking_matrix[i][j]

    return @[("ccl_linking_matrix",
              pad(ccl_linking_matrix.toTensor(),
                  [shape[1], shape[1]], -1000).astype(float))]

discard """
Emits the map sending each crossing strand to the index of
the link component it belongs to.
"""
type ComponentsMatrix* = ref object of TensorFeature

method cls_name*(self: ComponentsMatrix): string =
    return "Link Components Matrix"

method need_unsimplified_diagram*(self: ComponentsMatrix): bool = true

method from_link*(self: ComponentsMatrix,
                  link: Link[int],
                  shape: array[0..1, int]): seq[(string, Tensor[float])] =
    let num_crossings = link.crossings.len
    # Preserve the fixed observation shape on a crossingless diagram.
    if num_crossings == 0:
        return @[("component_matrix",
                  pad(zeros[int]([0, 4]),
                      [shape[0], 4], -1000).astype(float))]
    var component_matrix = newSeqWith(num_crossings, [-1, -1, -1, -1])

    for comp_index in 0 ..< link.link_components.len:
        let comp = link.link_components[comp_index]
        var cur_c = comp.crossing
        var cur_s = comp.strand_index
        if cur_c < 0 or cur_c >= num_crossings:
            continue
        while component_matrix[cur_c][cur_s] == -1:
            component_matrix[cur_c][cur_s] = comp_index
            component_matrix[cur_c][(cur_s + 2) mod 4] = comp_index
            (cur_c, cur_s) = link.next_strand((cur_c, cur_s))
            if cur_c < 0 or cur_c >= num_crossings:
                break

    return @[("component_matrix",
              pad(component_matrix.toTensor(),
                  [shape[0], 4], -1000).astype(float))]

discard """
Emits the Murasugi-Tristram bound at lambda = -1 as a
scalar, or the -1000 sentinel where it is undefined.
"""
type MTBound* = ref object of TensorFeature

method cls_name*(self: MTBound): string =
    return "Murasugi-Tristram bound"

method need_unsimplified_diagram*(self: MTBound): bool = false

method from_link*(self: MTBound,
                  link: Link[int],
                  shape: array[0..1, int]): seq[(string, Tensor[float])] =
    # Check if link is split
    if link.split_link_diagram().len > 1:
        return @[("mt_bound", [-1000.0].toTensor())]

    let num_comps = link.link_components.len + link.unlinked_unknot_components

    # The vendored braid/Seifert path can reject degenerate episode diagrams.
    var V: seq[seq[int]]
    try:
        V = seifert_matrix(link)[0]
    except ValueError, IndexDefect, RangeDefect, OverflowDefect:
        return @[("mt_bound", [-1000.0].toTensor())]
    let n = V.len

    if n == 0:
        # Unknot case: signature = 0, nullity = 0
        let bound = (0 + 0 - num_comps + 1) div 2
        return @[("mt_bound", [float(bound)].toTensor())]

    # Compute M = 2V + 2V^T
    var M = newSeqWith(n, newSeq[float](n))
    for i in 0 ..< n:
        for j in 0 ..< n:
            M[i][j] = float(2 * V[i][j] + 2 * V[j][i])

    # arraymancer 0.7.33 segfaults here unless eigenvectors are requested:
    # its LAPACK path dereferences an unallocated eigenvector buffer.
    let M_tensor = M.toTensor()
    # LAPACK non-convergence also makes this feature undefined.
    var eigenvalues: Tensor[float]
    try:
        eigenvalues = M_tensor.symeig(return_eigenvectors = true).eigenval
    except ValueError:
        return @[("mt_bound", [-1000.0].toTensor())]

    # Compute signature and nullity from eigenvalues
    var signature = 0
    var nullity = 0
    let epsilon = 1e-9
    for ev in eigenvalues:
        if abs(ev) < epsilon:
            nullity += 1
        elif ev > 0:
            signature += 1
        else:
            signature -= 1

    let bound = (abs(signature) + nullity - num_comps + 1) div 2
    return @[("mt_bound", [float(bound)].toTensor())]

discard """
Emits the total genus of the diagram's canonical Seifert surface,
(2*r(D) - l - k(D) + c(D)) / 2, where r(D) counts connected
components of the diagram, including crossingless unknots.
"""
type SeifertGenusBound* = ref object of TensorFeature

method cls_name*(self: SeifertGenusBound): string =
    return "Seifert genus bound"

method need_unsimplified_diagram*(self: SeifertGenusBound): bool = false

method from_link*(self: SeifertGenusBound,
                  link: Link[int],
                  shape: array[0..1, int]): seq[(string, Tensor[float])] =
    # The vendored Seifert-circle path can reject degenerate episode diagrams.
    var num_circles: int
    try:
        num_circles = num_seifert_circles(link)
    except ValueError, IndexDefect, RangeDefect, OverflowDefect:
        return @[("seifert_genus_bound", [-1000.0].toTensor())]
    let num_crossings = link.crossings.len
    # Count crossingless unknots in r and l, as num_seifert_circles does in k.
    let num_components = link.link_components.len + link.unlinked_unknot_components
    let num_diagram_components = link.split_link_diagram().len + link.unlinked_unknot_components

    let bound = (2 * num_diagram_components - num_components - num_circles + num_crossings) div 2
    return @[("seifert_genus_bound", [float(bound)].toTensor())]

discard """
Emit the numerical Theta invariant of the link, signed-log
compressed, together with a flag saying whether it is
undefined.

Two classes share one implementation and differ only in how
many evaluation points they use: OnePoint takes the first
point alone (5 floats), ThreePoint all three (15). Cost is
linear in the number of points. They emit the same
observation keys, so they are alternatives rather than
additions.
"""
type ThetaEvaluationsOnePoint* = ref object of TensorFeature
type ThetaEvaluationsThreePoint* = ref object of TensorFeature

proc signed_log*(x: float): float =
    discard """
    Compresses a value monotonically to keep the observation
    in a range comparable to the other features.

    Args:
        x: the value to compress

    Returns:
        sign(x) ln(1 + |x|).
    """
    return float(sgn(x)) * ln(1.0 + abs(x))

method cls_name*(self: ThetaEvaluationsOnePoint): string =
    return "Theta Evaluations (one point)"

method need_unsimplified_diagram*(self: ThetaEvaluationsOnePoint): bool = false


method cls_name*(self: ThetaEvaluationsThreePoint): string =
    return "Theta Evaluations (three points)"

method need_unsimplified_diagram*(self: ThetaEvaluationsThreePoint): bool = false


proc theta_observation(
    points: openArray[(float, float)],
    link: Link[int]
): seq[(string, Tensor[float])] =
    discard """
    Packs the Theta evaluations of a link into the observation
    tensors. The invariant itself is computed by
    theta.theta_at_points; this only compresses the values and
    flags the undefined case.

    Args:
        points: the evaluation points (T1, T2)
        link: the given link

    Returns:
        theta_evals, holding five signed-log floats per point
        (Delta1, Delta2, Delta3, theta and theta0), and
        theta_undefined, which is 1 when Theta is undefined at
        any of the points. The tensor is zeroed in that case,
        so a partial evaluation is never emitted.
    """
    var out_tensor = zeros[float]([5 * points.len])
    let (values, defined) = theta_at_points(link, points)
    if defined:
        for i, v in values:
            out_tensor[i] = signed_log(v)
    return @[("theta_evals", out_tensor),
             ("theta_undefined", [(if defined: 0.0 else: 1.0)].toTensor())]

method from_link*(self: ThetaEvaluationsOnePoint,
                  link: Link[int],
                  shape: array[0..1, int]): seq[(string, Tensor[float])] =
    theta_observation([THETA_PAPER_POINT], link)

method from_link*(self: ThetaEvaluationsThreePoint,
                  link: Link[int],
                  shape: array[0..1, int]): seq[(string, Tensor[float])] =
    theta_observation(THETA_EVAL_POINTS, link)

discard """
Emits the coefficients of the Alexander polynomial of the
link in a fixed window, all -1000 where it is undefined or
does not fit.
"""
type AlexanderPolynomial* = ref object of TensorFeature

method cls_name*(self: AlexanderPolynomial): string =
    return "Alexander Polynomial"

method need_unsimplified_diagram*(self: AlexanderPolynomial): bool = false

# The coefficient window both polynomial features pad to, as a function of the
# observation shape alone. Defined once so that the two cannot drift apart.
proc poly_window_len(shape: array[0..1, int]): int = 2 * shape[0] + 1

proc pd_to_seqseq(pd: seq[array[0..3, int]]): seq[seq[int]] =
    discard """
    Adapts the fixed-size PD tuples of spherogram-nim to the
    sequences the polynomial code expects.

    Args:
        pd: the PD code as fixed-size tuples

    Returns:
        The same PD code as sequences.
    """
    result = newSeq[seq[int]](pd.len)
    for i, c in pd:
        result[i] = @[c[0], c[1], c[2], c[3]]

proc lpoly_to_window(p: Lpoly, win: int): Tensor[float] =
    discard """
    Writes the coefficients of a polynomial into a fixed
    window, padding with the -1000 sentinel.

    Args:
        p: the given polynomial
        win: the length of the window

    Returns:
        The window, filled entirely with the sentinel if the
        polynomial has more coefficients than fit. The zero
        polynomial is encoded as 0 followed by padding.
    """
    var w = newSeqWith(win, -1000.0)
    var q = p
    q.trim()
    if q.coeffs.len > win:
        return w.toTensor()   # does not fit: emit all-pad sentinel
    if q.isZero and win > 0:
        w[0] = 0.0
    for i in 0 ..< q.coeffs.len:
        w[i] = float(q.coeffs[i])
    return w.toTensor()

method from_link*(self: AlexanderPolynomial,
                  link: Link[int],
                  shape: array[0..1, int]): seq[(string, Tensor[float])] =
    let win = poly_window_len(shape)
    if link.unlinked_unknot_components > 0 and
            (link.crossings.len > 0 or link.unlinked_unknot_components > 1):
        # A free circle together with another component is a split link.
        return @[("alexander_coeffs", lpoly_to_window(lpZero(), win))]
    if link.crossings.len == 0:
        # Delta = 1 for a single unknot and, by convention, the empty state.
        return @[("alexander_coeffs", lpoly_to_window(lpConst(1), win))]
    # The Wirtinger route in alexander.nim reads the PD code alone, so it also
    # works on diagrams no braid algorithm accepts. Its result is normalised to
    # lowest exponent 0 with a positive leading coefficient, so there is no
    # separate exponent to emit.
    var poly: Lpoly
    try:
        poly = alexanderPoly(pd_to_seqseq(link.PD_code()))
    except ValueError, IndexDefect, RangeDefect, OverflowDefect:
        return @[("alexander_coeffs", newSeqWith(win, -1000.0).toTensor())]
    return @[("alexander_coeffs", lpoly_to_window(poly, win))]

discard """
Emits the coefficients of the Jones polynomial of the link
in a fixed window, together with their lowest exponent, all
-1000 where it is undefined or does not fit.
"""
type JonesPolynomial* = ref object of TensorFeature

# Above this many crossings JonesPolynomial emits the
# sentinel instead of computing. The bracket recursion short
# circuits on kinks and on closed loops, so it is near linear
# on structured diagrams, but a generic one has no such
# structure and the cost roughly doubles per crossing: about
# 1.6 ms at 16 crossings, 15 ms at 18, 100 ms at 22, 800 ms
# at 26. Override at compile time without editing this file,
# for example nim c -d:JONES_MAX_CROSSINGS=22 ...
const JONES_MAX_CROSSINGS* {.intdefine.}: int = 16

method cls_name*(self: JonesPolynomial): string =
    return "Jones Polynomial"

method need_unsimplified_diagram*(self: JonesPolynomial): bool = false

method from_link*(self: JonesPolynomial,
                  link: Link[int],
                  shape: array[0..1, int]): seq[(string, Tensor[float])] =
    let win = poly_window_len(shape)
    if link.crossings.len > JONES_MAX_CROSSINGS:
        # Skip the exponential state sum above the configured cap.
        return @[("jones_lo", [-1000.0].toTensor()),
                 ("jones_coeffs", newSeqWith(win, -1000.0).toTensor())]
    # jones_port returns the skein-normalised polynomial in A; jonesInSqrtT
    # converts it to x = sqrt(t), the variable this observation uses.
    var poly: Lpoly
    try:
        poly = jonesInSqrtT(jonesPolySageInA(link))
    except ValueError, IndexDefect, RangeDefect, OverflowDefect, AssertionDefect:
        return @[("jones_lo", [-1000.0].toTensor()),
                 ("jones_coeffs", newSeqWith(win, -1000.0).toTensor())]
    poly.trim()
    if poly.coeffs.len > win:
        return @[("jones_lo", [-1000.0].toTensor()),
                 ("jones_coeffs", newSeqWith(win, -1000.0).toTensor())]
    let lo = (if poly.isZero: 0 else: poly.lo)
    return @[("jones_lo", [float(lo)].toTensor()),
             ("jones_coeffs", lpoly_to_window(poly, win))]

proc decode_cls_name*(feature_cls_name: string): TensorFeature =
    if feature_cls_name == "AlexanderPolynomial":
        return AlexanderPolynomial()
    if feature_cls_name == "JonesPolynomial":
        return JonesPolynomial()
    if feature_cls_name == "DeterminantAndSignature":
        return DeterminantAndSignature()
    if feature_cls_name == "LinkingMatrix":
        return LinkingMatrix()
    if feature_cls_name == "CCLLinkingMatrix":
        return CCLLinkingMatrix()
    if feature_cls_name == "ComponentsMatrix":
        return ComponentsMatrix()
    if feature_cls_name == "MTBound":
        return MTBound()
    if feature_cls_name == "SeifertGenusBound":
        return SeifertGenusBound()
    if feature_cls_name == "ThetaEvaluationsOnePoint":
        return ThetaEvaluationsOnePoint()
    if feature_cls_name == "ThetaEvaluationsThreePoint":
        return ThetaEvaluationsThreePoint()
    raise newException(ValueError, &"feature {feature_cls_name} does not exist")
