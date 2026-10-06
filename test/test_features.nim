import std/[unittest, sequtils]

import "../spherogram-nim/src/links"
import arraymancer

import ../src/features

test "DeterminantAndSignature":
    # the unknot
    var unknot = link_from_PD_code(@[], @[], 1)
    check (DeterminantAndSignature().from_link(unknot, [0, 0]) ==
           @[("determinant", [1.0].toTensor()),
             ("signature", [0.0].toTensor())])
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    check (DeterminantAndSignature().from_link(trefoil, [3, 1]) ==
           @[("determinant", [3.0].toTensor()),
             ("signature", [2.0].toTensor())])

test "LinkingMatrix":
    var unknot = link_from_PD_code(@[], @[], 1)
    check (LinkingMatrix().from_link(unknot, [3, 1]) ==
           @[("linking_matrix", [[-1000.0]].toTensor())])
    var l2a1 = link_from_PD_code(@[[2, 1, 3, 0], [0, 3, 1, 2]])
    check (LinkingMatrix().from_link(l2a1, [3, 2]) ==
           @[("linking_matrix", [[0.0, -1.0], [-1.0, 0.0]].toTensor())])

test "SeifertGenusBound on connected diagrams":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    let hopf = link_from_PD_code(@[[2, 1, 3, 0], [0, 3, 1, 2]])
    let rii_unlink = link_from_PD_code(@[[0, 2, 1, 3], [1, 2, 0, 3]], @[1, -1])
    check (SeifertGenusBound().from_link(trefoil, [3, 1]) ==
           @[("seifert_genus_bound", [1.0].toTensor())])
    # Diagram components are not link components, nor topological split factors.
    for link in [hopf, rii_unlink]:
        check link.link_components.len == 2
        check link.split_link_diagram().len == 1
        check (SeifertGenusBound().from_link(link, [2, 2]) ==
               @[("seifert_genus_bound", [0.0].toTensor())])

test "SeifertGenusBound is additive on disjoint diagrams":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    let two_trefoils = trefoil.disjoint_union(trefoil)
    let three_trefoils = two_trefoils.disjoint_union(trefoil)
    check (SeifertGenusBound().from_link(two_trefoils, [6, 2]) ==
           @[("seifert_genus_bound", [2.0].toTensor())])
    check (SeifertGenusBound().from_link(three_trefoils, [9, 3]) ==
           @[("seifert_genus_bound", [3.0].toTensor())])

test "SeifertGenusBound on crossingless diagrams":
    for count in 0..4:
        let unlink = link_from_PD_code(@[], unlinked_unknot_components = count)
        check (SeifertGenusBound().from_link(unlink, [0, 4]) ==
               @[("seifert_genus_bound", [0.0].toTensor())])

test "SeifertGenusBound ignores added crossingless disks":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    let two_trefoils = trefoil.disjoint_union(trefoil)
    for count in 1..4:
        let unlink = link_from_PD_code(@[], unlinked_unknot_components = count)
        check (SeifertGenusBound().from_link(trefoil.disjoint_union(unlink), [3, 5]) ==
               @[("seifert_genus_bound", [1.0].toTensor())])
        check (SeifertGenusBound().from_link(two_trefoils.disjoint_union(unlink), [6, 6]) ==
               @[("seifert_genus_bound", [2.0].toTensor())])

proc polynomialWindow(values: openArray[float], size: int): Tensor[float] =
    var padded = newSeqWith(size, -1000.0)
    for i, value in values:
        padded[i] = value
    padded.toTensor()

test "Polynomial features retain the empty-state and unknot conventions":
    for count in 0..1:
        let link = link_from_PD_code(@[], unlinked_unknot_components = count)
        check (AlexanderPolynomial().from_link(link, [0, 1]) ==
               @[("alexander_coeffs", [1.0].toTensor())])
        check (JonesPolynomial().from_link(link, [0, 1]) ==
               @[("jones_lo", [0.0].toTensor()),
                 ("jones_coeffs", [1.0].toTensor())])

test "Polynomial features count crossingless unlink components":
    let cases = @[
        (2, -1.0, @[-1.0, 0.0, -1.0]),
        (3, -2.0, @[1.0, 0.0, 2.0, 0.0, 1.0]),
        (4, -3.0, @[-1.0, 0.0, -3.0, 0.0, -3.0, 0.0, -1.0])]
    for (count, lo, coeffs) in cases:
        let link = link_from_PD_code(@[], unlinked_unknot_components = count)
        check (AlexanderPolynomial().from_link(link, [4, 4]) ==
               @[("alexander_coeffs", polynomialWindow([0.0], 9))])
        check (JonesPolynomial().from_link(link, [4, 4]) ==
               @[("jones_lo", [lo].toTensor()),
                 ("jones_coeffs", polynomialWindow(coeffs, 9))])

test "Polynomial features include free unknots beside a drawn knot":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    for count in 1..3:
        let unlink = link_from_PD_code(@[], unlinked_unknot_components = count)
        check (AlexanderPolynomial().from_link(trefoil.disjoint_union(unlink), [5, 4]) ==
               @[("alexander_coeffs", polynomialWindow([0.0], 11))])
    let unknot = link_from_PD_code(@[], unlinked_unknot_components = 1)
    check (JonesPolynomial().from_link(trefoil.disjoint_union(unknot), [5, 2]) ==
           @[("jones_lo", [1.0].toTensor()),
             ("jones_coeffs", polynomialWindow([-1.0, 0.0, -1.0, 0.0, -1.0,
                                               0.0, 0.0, 0.0, 1.0], 11))])

test "Alexander zero is distinct from unavailable for drawn split links":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    let rii = link_from_PD_code(@[[0, 2, 1, 3], [1, 2, 0, 3]], @[1, -1])
    for link in [rii, trefoil.disjoint_union(trefoil)]:
        check (AlexanderPolynomial().from_link(link, [6, 2]) ==
               @[("alexander_coeffs", polynomialWindow([0.0], 13))])
    check (AlexanderPolynomial().from_link(trefoil, [0, 1]) ==
           @[("alexander_coeffs", [-1000.0].toTensor())])

test "Alexander feature preserves integer content":
    let l6a5 = link_from_PD_code(@[[4, 0, 5, 3], [0, 4, 1, 7], [8, 2, 9, 1],
                                  [2, 8, 3, 11], [10, 6, 11, 5], [6, 10, 7, 9]])
    check (AlexanderPolynomial().from_link(l6a5, [6, 3]) ==
           @[("alexander_coeffs", polynomialWindow([3.0, -6.0, 3.0], 13))])

test "Alexander feature marks an uncertifiable CRT result unavailable":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var link = trefoil
    for _ in 1..<11:
        link = link.connected_sum(trefoil)
    # The conservative determinant bound exceeds the fixed CRT modulus,
    # even though this knot's actual polynomial would fit the window.
    check (AlexanderPolynomial().from_link(link, [33, 1]) ==
           @[("alexander_coeffs", polynomialWindow([], 67))])

test "Jones feature preserves its inverse-Sage convention":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    check (JonesPolynomial().from_link(trefoil, [3, 1]) ==
           @[("jones_lo", [2.0].toTensor()),
             ("jones_coeffs", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0].toTensor())])

test "Jones feature marks both outputs unavailable when the window is too short":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    let twoTrefoils = trefoil.disjoint_union(trefoil)
    check (JonesPolynomial().from_link(twoTrefoils, [6, 2]) ==
           @[("jones_lo", [-1000.0].toTensor()),
             ("jones_coeffs", polynomialWindow([], 13))])
    let unlink = link_from_PD_code(@[], unlinked_unknot_components = 2)
    check (JonesPolynomial().from_link(unlink, [0, 2]) ==
           @[("jones_lo", [-1000.0].toTensor()),
             ("jones_coeffs", [-1000.0].toTensor())])

test "Jones feature retains the crossing cap":
    let kink = link_from_PD_code(@[[0, 0, 1, 1]])
    var link = kink
    for _ in 0..<JONES_MAX_CROSSINGS:
        link = link.disjoint_union(kink)
    check (JonesPolynomial().from_link(link, [JONES_MAX_CROSSINGS + 1, 1]) ==
           @[("jones_lo", [-1000.0].toTensor()),
             ("jones_coeffs", polynomialWindow([], 2 * (JONES_MAX_CROSSINGS + 1) + 1))])
