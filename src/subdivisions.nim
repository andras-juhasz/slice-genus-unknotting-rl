import std/strformat
import std/options
import std/sugar

proc split_face_into_two*(
    strands: seq[(int, int)],
    strand1: (int, int),
    strand2: (int, int)
): (seq[(int, int)], seq[(int, int)]) =
    discard """
    After the band enters a subdivision of a face through a
    strand and exits through another strand, splits the
    subdivision (excluding the two strands that the band
    passed through) into two and returns the strands in the
    two new subdivisions.

    Args:
        strands: the list of crossing strands in the
            original subdivision
        strand1: the strand (counterclockwise w.r.t. a point
            in the interior of the face) that the band
            entered the face from
        strand2: the strand (counterclockwise w.r.t. a point
            in the interior of the face) that the band exit
            the face from

    Returns:
        A tuple of two lists -- the lists of crossing
        strands in each of the two new subdivisions

    Raises:
        ValueError: if strand1 and strand2 are equal, or if
            strand1 or strand2 does not appear in strands
    """
    if strand1 == strand2:
        raise newException(ValueError, &"strand1 = {strand1} = strand2, but in order for the face to be subdivided, strand1 and strand2 need to be different")
    if strand1 notin strands:
        raise newException(ValueError, &"strand1 = {strand1} is not in face with strands {strands}")
    if strand2 notin strands:
        raise newException(ValueError, &"strand2 = {strand2} is not in face with strands {strands}")
    var index1 = strands.find(strand1)
    var index2 = strands.find(strand2)
    if index1 > index2:
        swap(index1, index2)
    # the second line below should not be changed to
    # strands[index2+1 - strands.len ..< index1] because
    # this would result in an IndexDefect
    return (strands[index1+1 ..< index2],
            strands[index2+1 ..< strands.len] & strands[0 ..< index1])

discard """
A data structure maintaining the subdivisions of a face
resulting from the band going through it, possibly
multiple times. When the band attempts to enter the
face, the data structure returns a list of crossing
strands that the band can exit the face from without
intersecting itself or crossing already-visited edges.
This list is then checked to be non-empty in
band.BandTwist to ensure, before entering the face, that
we can exit it. When we need to decide the next over /
under / end action, the data structure again gives a
list of possible exit strands.
"""
type SubdividedFace* = object of RootObj
    subdivisions*: seq[seq[(int, int)]]
    initial_strand*: Option[(int, int)]

proc newSubdividedFace*(strands: seq[(int, int)]): SubdividedFace =
    discard """
    Initializes an undivided face with a list of
    crossing strands (all counterclockwise w.r.t. a
    point in the interior of the face).

    Args:
        strands: the list of crossing strands that bound
            the face
    """
    return SubdividedFace(subdivisions: @[strands], initial_strand: none((int, int)))

proc set_initial_strand*(self: var SubdividedFace, initial_strand: (int, int)): void =
    discard """
    After the initial strand is established, marks it as
    the initial strand on the face the band is not
    entering, so that the band would not be able to
    cross it later.

    Args:
        initial_strand: the established initial strand
    """
    self.initial_strand = some(initial_strand)

proc feasible_next_strands*(self: SubdividedFace, strand: (int, int)): seq[(int, int)] =
    discard """
    Given an uncrossed strand in a face, finds the list
    of other strands in its subdivision, excluding the
    starting strand.

    Args:
        strand: the strand that the band entered, or is
            trying to enter, the face from

    Returns:
        A list of strands that the band may exit the
        face from without intersecting itself or going
        through already-visited edges.

    Raises:
        ValueError: if the strand is not found in any
            subdivision of the face
    """
    for subdivision in self.subdivisions:
        if strand in subdivision:
            return collect:
                for next_strand in subdivision:
                    if next_strand != strand and (self.initial_strand.isNone or next_strand != self.initial_strand.get()):
                        next_strand
    raise newException(ValueError, &"strand {strand} cannot be found in any subdivision of {self.subdivisions}")

proc cut_through*(self: var SubdividedFace, strand1: (int, int), strand2: (int, int)): void =
    discard """
    Given two strands that the band entered and exited
    (a subdivision of) the face from, splits the
    subdivision into two and replaces the list of
    strands of the subdivision with the lists of strands
    in each of the two new subdivisions (empty lists are
    not included).

    Args:
        strand1: the strand that the band entered the
            face from
        strand2: the strand that the band exited the
            face from

    Raises:
        AssertionError: if strand1 is in some
            subdivision but strand2 is not in the same
            subdivision
        ValueError: if strand1 cannot be found in any
            subdivision
    """
    var i = 0
    while i < self.subdivisions.len and strand1 notin self.subdivisions[i]:
        i += 1
    if i == self.subdivisions.len:
        raise newException(ValueError, &"strand {strand1} cannot be found in any subdivision of {self.subdivisions}")
    let subdivision = self.subdivisions[i]
    assert strand2 in subdivision
    var parts = split_face_into_two(subdivision, strand1, strand2)
    var new_subdivisions = self.subdivisions[0 ..< i]
    if parts[0].len > 0:
        new_subdivisions.add(parts[0])
    if parts[1].len > 0:
        new_subdivisions.add(parts[1])
    for j in i+1 ..< self.subdivisions.len:
        new_subdivisions.add(self.subdivisions[j])
    self.subdivisions = new_subdivisions
