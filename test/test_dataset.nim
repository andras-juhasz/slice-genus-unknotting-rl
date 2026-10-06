import std/unittest
import std/options
import std/random
import std/sequtils

import "../spherogram-nim/src/links"
import datamancer

import ../src/dataset

test "parse_invariant":
    # check that rep is stripped of whitespaces and that
    # ";" is replaced by ","
    check parse_invariant(" [1;2] ") == some(@[1, 2])
    # check that the empty string leads to the return
    # value none(seq[int])
    check parse_invariant("") == none(seq[int])
    # check that a single integer is supported
    for i in 0 ..< 10:
        check parse_invariant($i) == some(@[i])
    # check that an illegal range leads to a ValueError
    expect ValueError:
        discard parse_invariant("[2,1]")
    # check that a range of integers is supported
    check parse_invariant("[2,4]") == some(@[2, 3, 4])
    # check that a set of integers is supported and
    # automatically sorted
    check parse_invariant("{3,1}") == some(@[1, 3])
    # check that illegal input leads to a ValueError
    expect ValueError:
        discard parse_invariant("<1,2>")

test "invariant_to_str":
    check invariant_to_str(none(seq[int])) == ""
    check invariant_to_str(some(@[1])) == "1"
    check invariant_to_str(some(@[1, 2])) == "[1;2]"
    check invariant_to_str(some(@[1, 3])) == "{1;3}"

test "newDataset":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]], name = "3_1")
    expect AssertionDefect:
        discard newDataset(@[trefoil], @[some(@[1])], @[some(@[1])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[some("3_1"), some("4_1")])
    expect AssertionDefect:
        discard newDataset(@[trefoil], @[some(newSeq[int]())], @[some(@[1])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[some("3_1")])
    expect AssertionDefect:
        discard newDataset(@[trefoil], @[some(@[1])], @[some(newSeq[int]())], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[some("3_1")])
    expect AssertionDefect:
        discard newDataset(@[trefoil], @[some(@[1, 0])], @[some(@[1])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[some("3_1")])
    expect AssertionDefect:
        discard newDataset(@[trefoil], @[some(@[1])], @[some(@[1, 0])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[some("3_1")])
    var dataset = newDataset(@[trefoil], @[some(@[1])], @[some(@[1])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[some("3_1")])
    check dataset.links[0].name == "3_1"
    check dataset.unknotting_nums[0] == some(@[1])
    check dataset.slice_genera[0] == some(@[1])
    check dataset.ribbon_genera[0] == none(seq[int])
    check dataset.strong_ribbon_genera[0] == none(seq[int])
    check dataset.names[0] == some("3_1")
    # invariant args passed as empty seqs are none-filled
    var dataset_defaults = newDataset(@[trefoil], @[some(@[1])], @[some(@[1])], @[none(seq[int])])
    check dataset_defaults.ribbon_genera == @[none(seq[int])]
    check dataset_defaults.strong_ribbon_genera == @[none(seq[int])]
    check dataset_defaults.names == @[none(string)]

test "copy":
    # check that self.links has indeed been (deep) copied
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var dataset = newDataset(@[trefoil], @[some(@[1])], @[some(@[1])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[none(seq[int])], @[some("3_1")])
    let dataset_copy = dataset.copy()
    dataset.links[0].name = "3_1"
    check dataset_copy.links[0].name == ""

test "dataset_from_dataframe":
    # check that deep copies of the DataFrame are made
    var df = toTab({"Name": ["3_1"], "PD Notation": ["[[5;2;0;3];[3;0;4;1];[1;4;2;5]]"]})
    discard dataset_from_dataframe(df)
    check df.getKeys()[0] == "Name"
    # check that the lack of a "name" column results in none values in ds.names
    df = toTab({"PD Notation": ["[[5;2;0;3];[3;0;4;1];[1;4;2;5]]"]})
    var ds = dataset_from_dataframe(df)
    check ds.names[0] == none(string)
    # check that multiple "name" columns result in a ValueError
    df = toTab({"Name": ["3_1"], "PD Notation": ["[[5;2;0;3];[3;0;4;1];[1;4;2;5]]"], "PD Code": ["[[5;2;0;3];[3;0;4;1];[1;4;2;5]]"]})
    expect ValueError:
        discard dataset_from_dataframe(df)
    # check that a ValueError is raised when no links are specified
    df = toTab({"Unknotting Number": ["1"]})
    expect ValueError:
        discard dataset_from_dataframe(df)
    # check that the PD code is correctly interpreted
    df = toTab({"Name": ["3_1"], "PD Notation": ["[[5;2;0;3];[3;0;4;1];[1;4;2;5]]"]})
    ds = dataset_from_dataframe(df)
    check ds.links[0].crossings == @[[11, 10, 5, 4], [3, 2, 9, 8], [7, 6, 1, 0]]
    # check that the name column is correctly interpreted
    check ds.names[0] == some("3_1")
    # check that a sample dataset with all columns filled in is correctly loaded
    df = toTab({"Name": ["L5a1{0}"],
                "PD Notation": ["[[5;0;6;1];[9;6;4;7];[3;4;0;5];[1;9;2;8];[7;3;8;2]]"],
                "Crossing Signs": ["[-1;-1;-1;1;1]"],
                "Unknotting Number": ["1"], "Genus-4D": ["1"],
                "Strong Slice Genus-4D": ["1"], "Ribbon Genus-4D": ["1"],
                "Strong Ribbon Genus-4D": ["1"],
                "Splitting Number": ["2"], "Weak Splitting Number": ["1"]})
    ds = dataset_from_dataframe(df)
    check ds.names[0] == some("L5a1{0}")
    check ds.links[0].crossings == @[[11, 10, 5, 12], [13, 2, 9, 16],
                                     [17, 6, 1, 0], [3, 4, 19, 18],
                                     [7, 8, 15, 14]]
    check ds.links[0].signs == @[-1, -1, -1, 1, 1]
    check ds.unknotting_nums[0] == some(@[1])
    check ds.slice_genera[0] == some(@[1])
    check ds.strong_slice_genera[0] == some(@[1])
    check ds.ribbon_genera[0] == some(@[1])
    check ds.strong_ribbon_genera[0] == some(@[1])
    check ds.splitting_nums[0] == some(@[2])
    check ds.weaksplitting_nums[0] == some(@[1])
    # the PD code already determines the crossing signs, so
    # the column is a consistency check rather than an
    # override: signs contradicting the diagram's orientation
    # are rejected, and so is a column of the wrong length
    df = toTab({"PD Notation": ["[[5;2;0;3];[3;0;4;1];[1;4;2;5]]"],
                "Crossing Signs": ["[1;1;1]"]})
    expect ValueError:
        discard dataset_from_dataframe(df)
    df = toTab({"PD Notation": ["[[4;1;3;2];[2;3;1;4]]"],
                "Crossing Signs": ["[1;1;1]"]})
    expect ValueError:
        discard dataset_from_dataframe(df)
    # the four genus columns must land in four distinct
    # fields, so in particular the ribbon regexes must not
    # collide with "Genus-4D" or "Strong Slice Genus-4D".
    # These values are synthetic and pairwise distinct
    # precisely so that such a collision is visible; the row
    # is left unnamed because no link has them.
    df = toTab({"PD Notation": ["[[5;2;0;3];[3;0;4;1];[1;4;2;5]]"],
                "Genus-4D": ["0"], "Strong Slice Genus-4D": ["1"],
                "Ribbon Genus-4D": ["2"], "Strong Ribbon Genus-4D": ["3"]})
    ds = dataset_from_dataframe(df)
    check ds.slice_genera[0] == some(@[0])
    check ds.strong_slice_genera[0] == some(@[1])
    check ds.ribbon_genera[0] == some(@[2])
    check ds.strong_ribbon_genera[0] == some(@[3])

test "`[]`":
    var ds = dataset_read_csv("../datasets/rolfsen.csv")
    # rolfsen.csv has no ribbon columns: the fields are none-filled
    check ds.ribbon_genera.allIt(it == none(seq[int]))
    check ds.strong_ribbon_genera.allIt(it == none(seq[int]))
    for i in 0 .. 10:
        var sub_ds = ds[0 ..< i]
        check sub_ds.links.len == i

test "evaluate_invariant":
    var ds = newDataset(newSeq[Link[int]](),
                        newSeq[Option[seq[int]]](),
                        newSeq[Option[seq[int]]](),
                        newSeq[Option[seq[int]]]())
    expect ValueError:
        discard ds.evaluate_invariant("unknotting", newSeq[(int, seq[(string, float)], seq[Link[int]], seq[Rand])]())
    ds = dataset_read_csv("../datasets/rolfsen.csv")[0 ..< 1]
    expect ValueError:
        discard ds.evaluate_invariant("unknotting", newSeq[(int, seq[(string, float)], seq[Link[int]], seq[Rand])]())
    ds = dataset_read_csv("../datasets/rolfsen.csv")[0 ..< 1]
    expect ValueError:
        discard ds.evaluate_invariant("unknotting", @[(1, @[("C0S1_start", 1.0), ("C0S0_end1", 150.0)], @[ds.links[0], link_from_PD_code(@[], @[], 1)], newSeq[Rand]())], false, "../outputs/evaluation.txt")
    check ds.evaluate_invariant("unknotting", @[(2, @[("C0S1_start", 1.0), ("C0S0_end1", 150.0)], @[ds.links[0], link_from_PD_code(@[], @[], 1)], newSeq[Rand]())]).get() == 0.0
    check ds.evaluate_invariant("unknotting", @[(1, @[("C0S1_start", 1.0), ("C0S0_end1", 150.0)], @[ds.links[0], link_from_PD_code(@[], @[], 1)], newSeq[Rand]())]).get() == 1.0
