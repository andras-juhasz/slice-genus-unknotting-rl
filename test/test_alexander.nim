import std/unittest
import ../src/alexander

const trefoil = @[@[5, 2, 0, 3], @[3, 0, 4, 1], @[1, 4, 2, 5]]
const hopf = @[@[2, 1, 3, 0], @[0, 3, 1, 2]]
const l6a5 = @[@[4, 0, 5, 3], @[0, 4, 1, 7], @[8, 2, 9, 1],
              @[2, 8, 3, 11], @[10, 6, 11, 5], @[6, 10, 7, 9]]
const l7a4 = @[@[4, 0, 5, 3], @[0, 4, 1, 13], @[8, 1, 9, 2],
              @[2, 9, 3, 10], @[12, 5, 13, 6], @[6, 11, 7, 12],
              @[10, 7, 11, 8]]

proc disjointUnion(a, b: seq[seq[int]]): seq[seq[int]] =
    result = a
    let offset = 2 * a.len
    for crossing in b:
        var shifted: seq[int] = @[]
        for label in crossing: shifted.add(label + offset)
        result.add shifted

suite "Alexander polynomial":
    test "unknot, trefoil and Hopf link":
        check alexanderPoly(@[]) == lpConst(1)
        check alexanderPoly(@[@[0, 1, 1, 0]]) == lpConst(1)
        check alexanderPoly(trefoil) == Lpoly(lo: 0, coeffs: @[1, -1, 1])
        check alexanderPoly(hopf) == Lpoly(lo: 0, coeffs: @[-1, 1])

    test "integer content is retained for links":
        # One-variable Alexander polynomials, normalised up to +-t^k;
        # these numerical factors are not units in Z[t, t^-1].
        check alexanderPoly(l6a5) == Lpoly(lo: 0, coeffs: @[3, -6, 3])
        check alexanderPoly(l7a4) == Lpoly(lo: 0, coeffs: @[-2, 6, -6, 2])

    test "all-over components give zero, not undersized minors":
        let rii = @[@[0, 2, 1, 3], @[1, 2, 0, 3]]
        let fourCrossings = @[@[4, 2, 5, 1], @[5, 2, 6, 3],
                             @[6, 0, 7, 3], @[7, 0, 4, 1]]
        let sixCrossings = @[@[6, 2, 7, 1], @[7, 2, 8, 3],
                            @[8, 4, 9, 3], @[9, 4, 10, 5],
                            @[10, 0, 11, 5], @[11, 0, 6, 1]]
        for pd in [rii, fourCrossings, sixCrossings]:
            check alexanderPoly(pd).isZero
        check alexanderPoly(disjointUnion(trefoil, rii)).isZero

    test "zero minors of disconnected nontrivial diagrams":
        check alexanderPoly(disjointUnion(trefoil, trefoil)).isZero
        check alexanderPoly(disjointUnion(l6a5, hopf)).isZero

suite "Certified CRT polynomial determinant":
    const modulus = 1000003 * 1000033 * 1000037
    const limit = modulus div 2

    test "small determinants":
        check detPolyCRT(@[]) == lpConst(1)
        check detPolyCRT(@[@[(1, 1), (2, 0)], @[(0, 1), (3, -1)]]) ==
            Lpoly(lo: 0, coeffs: @[3, 0, -1])

    test "signed reconstruction boundary is inclusive":
        check detPolyCRT(@[@[(limit, 0)]]) == lpConst(limit)
        check detPolyCRT(@[@[(-limit, 0)]]) == lpConst(-limit)
        check detPolyCRT(@[@[(0, limit)]]) == lpMono(limit, 1)
        expect ValueError:
            discard detPolyCRT(@[@[(limit + 1, 0)]])
        expect ValueError:
            discard detPolyCRT(@[@[(-limit - 1, 0)]])

    test "uncertifiable sums and products are refused":
        expect ValueError:
            discard detPolyCRT(@[@[(limit, 1)]])
        expect ValueError:
            discard detPolyCRT(@[@[(limit, 0), (0, 0)], @[(0, 0), (2, 0)]])
        expect ValueError:
            discard detPolyCRT(@[@[(high(int), low(int))]])

    test "former deterministic witness false acceptance is refused":
        # The old CRT result was zero and its sole witness x=3 also
        # vanished, although the true determinant is modulus*(t-3).
        expect ValueError:
            discard detPolyCRT(@[@[(-3 * modulus, modulus)]])

    test "zero row and modular evaluation cannot overflow":
        # The zero row certifies det=0 even with large entries elsewhere.
        check detPolyCRT(@[@[(high(int), low(int)), (limit, limit)],
                          @[(0, 0), (0, 0)]]).isZero

    test "matrix must be square":
        expect ValueError:
            discard detPolyCRT(@[@[(1, 0), (0, 0)]])
