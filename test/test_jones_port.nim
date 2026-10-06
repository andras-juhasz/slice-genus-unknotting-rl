import std/unittest

import "../spherogram-nim/src/links"
import ../src/alexander
import ../src/jones_port

proc trefoil(): Link0 =
    link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])

proc hopf(): Link0 =
    link_from_PD_code(@[[2, 1, 3, 0], [0, 3, 1, 2]])

proc unlink(count: int): Link0 =
    link_from_PD_code(@[], unlinked_unknot_components = count)

proc pdSeq(link: Link0): seq[seq[int]] =
    for crossing in link.PD_code():
        result.add(@[crossing[0], crossing[1], crossing[2], crossing[3]])

proc power(p: Lpoly, n: int): Lpoly =
    result = lpConst(1)
    for _ in 0 ..< n:
        result = result * p

proc inverted(p: Lpoly): Lpoly =
    result = lpZero()
    for i, coefficient in p.coeffs:
        result = result + lpMono(coefficient, -(p.lo + i))

let deltaA = lpMono(-1, -2) + lpMono(-1, 2)
let deltaX = lpMono(-1, -1) + lpMono(-1, 1)

test "Jones retains the empty terminal state's value one":
    check jonesPolySageInA(unlink(0)) == lpConst(1)
    check jonesInSqrtT(jonesPolySageInA(unlink(0))) == lpConst(1)

test "Jones of a crossingless n-component unlink is delta^(n-1)":
    for count in 1 .. 5:
        let polynomial = jonesPolySageInA(unlink(count))
        check polynomial == power(deltaA, count - 1)
        check jonesInSqrtT(polynomial) == power(deltaX, count - 1)

test "Jones includes each free unknot beside a nonempty diagram":
    for link in [trefoil(), hopf(), trefoil().disjoint_union(trefoil())]:
        let original = jonesPolySageInA(link)
        for count in 1 .. 4:
            let withUnknots = link.disjoint_union(unlink(count))
            check jonesPolySageInA(withUnknots) == original * power(deltaA, count)
            check jonesInSqrtT(jonesPolySageInA(withUnknots)) ==
                jonesInSqrtT(original) * power(deltaX, count)

test "Jones preserves the existing inverse-Sage sqrt(t) convention":
    # The skein polynomial agrees with Sage; the observation substitutes
    # A = t^(-1/4), the inverse of Sage's ordinary-variable convention.
    let trefoilA = lpMono(-1, -16) + lpMono(1, -12) + lpMono(1, -4)
    let trefoilX = lpMono(1, 2) + lpMono(1, 6) + lpMono(-1, 8)
    check jonesPolySageInA(trefoil()) == trefoilA
    check jonesInSqrtT(trefoilA) == trefoilX
    let hopfA = lpMono(-1, -10) + lpMono(-1, -2)
    let hopfX = lpMono(-1, 1) + lpMono(-1, 5)
    check jonesPolySageInA(hopf()) == hopfA
    check jonesInSqrtT(hopfA) == hopfX

test "Jones agrees on crossingful and crossingless presentations of unlinks":
    let riiUnlink = link_from_PD_code(@[[0, 2, 1, 3], [1, 2, 0, 3]], @[1, -1])
    check jonesPolySageInA(riiUnlink) == jonesPolySageInA(unlink(2))
    for kink in [link_from_PD_code(@[[0, 0, 1, 1]]),
                 link_from_PD_code(@[[0, 1, 1, 0]])]:
        check jonesPolySageInA(kink) == jonesPolySageInA(unlink(1))
        check jonesPolySageInA(kink.disjoint_union(unlink(1))) ==
            jonesPolySageInA(unlink(2))

test "Jones of a mirror inverts the variable":
    for link in [trefoil(), hopf(), trefoil().disjoint_union(unlink(2))]:
        let original = jonesPolySageInA(link)
        let mirrored = jonesPolySageInA(link.mirror())
        check mirrored == inverted(original)
        check jonesInSqrtT(mirrored) == inverted(jonesInSqrtT(original))

test "Jones of disjoint nonempty diagrams has one additional delta factor":
    for first in [trefoil(), hopf()]:
        for second in [trefoil(), trefoil().mirror(), hopf()]:
            check jonesPolySageInA(first.disjoint_union(second)) ==
                deltaA * jonesPolySageInA(first) * jonesPolySageInA(second)

test "Jones agrees with the independent state sum on supported PD examples":
    # Do not use the PD-only reference for an all-over component: its PD
    # parser cannot recover that component's orientation from labels alone.
    for link in [trefoil(), trefoil().mirror(), hopf(), hopf().mirror(),
                 trefoil().disjoint_union(trefoil())]:
        check jonesInSqrtT(jonesPolySageInA(link)) == jonesPoly(pdSeq(link))

test "Jones recursion leaves its PD and link inputs unchanged":
    let link = trefoil().disjoint_union(hopf())
    let pd = pdSeq(link)
    let beforePD = $pd
    let beforeCrossings = $link.crossings
    let beforeSigns = $link.signs
    discard bracketSage(pd)
    discard jonesPolySageInA(link)
    check $pd == beforePD
    check $link.crossings == beforeCrossings
    check $link.signs == beforeSigns
