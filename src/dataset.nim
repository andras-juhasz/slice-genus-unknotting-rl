import std/options
import std/os
import std/strutils
import std/re
import std/strformat
import std/sequtils
import std/sugar
from std/json import parseJson, getElems, getInt
import std/algorithm
import std/random
import std/math
import std/syncio
import std/terminal
import std/streams
import std/marshal
import std/enumerate
import std/tables

import datamancer
import "../spherogram-nim/src/links"

proc parse_invariant*(rep0: string): Option[seq[int]] =
    discard """
    Converts a string representation of the value of a link
    invariant to a sorted list of possible values for the
    link invariant or none (if the value is not given).

    Args:
        rep: the string representation of the value of a
            link invariant; may be an empty string (meaning
            that the value is not given), a single integer,
            a range of integers, or a set of integers

    Returns:
        A sorted list of the possible values for the link
        invariant.

    Raises:
        ValueError: if the string represents an illegal
            range or is not of one of the recognized forms
    """
    var rep = rep0.strip().replace(';', ',')
    if rep == "":
        return none(seq[int])
    if match(rep, re"^\d+$"):
        return some(@[parseInt(rep)])
    if match(rep, re"\[\d+,\d+]$"):
        var captured = ["", ""]
        discard match(rep, re"\[(\d+),(\d+)]$", captured)
        let low_range = parseInt(captured[0])
        let high_range = parseInt(captured[1])
        if low_range > high_range:
            raise newException(ValueError, &"the range {rep} is illegal")
        return some((low_range .. high_range).toSeq())
    if match(rep, re"^\{(\d+,)*\d+}$"):
        var int_arr = collect:
            for node in parseJson(&"[{rep[1 ..< rep.len-1]}]").getElems():
                node.getInt()
        return some(sorted(int_arr))
    raise newException(ValueError, &"representation {rep} cannot be parsed because it is not empty, a single integer, a range of integers, or a set of integers")

proc invariant_to_str*(v0: Option[seq[int]]): string =
    discard """
    Converts the value of some link invariant to a string.

    Args:
        v0: the value of a link invariant

    Returns:
        The string representation of v. If v is none,
        returns the empty string. A single integer is
        preferred to a range of integers, which in turn is
        preferred to a set of integers.
    """
    if v0.isNone:
        return ""
    let v = v0.get()
    if v.len == 1:
        return $v[0]
    if (0 ..< v.len-1).toSeq().allIt(v[it+1] == v[it]+1):
        return &"[{v[0]};{v[v.len-1]}]"
    let str_arr = collect:
        for n in v:
            $n
    return "{" & str_arr.join(";") & "}"

proc known_value_to_display_str*(v0: Option[seq[int]],
                                 max_listed: int = 10): string =
    discard """
    Formats a known invariant value for display in evaluation
    output.
    Avoids dumping e.g. the whole [0, 1, ..., 9999] placeholder range
    for invariants whose value is effectively unconstrained.

    Args:
        v0: the known invariant value (none if unknown)
        max_listed: the largest number of possible values that
            is still printed in full; above this, the compact
            range form is used

    Returns:
        The display string. none becomes the empty string; at
        most max_listed values are listed verbatim, otherwise
        the compact invariant_to_str form is returned.
    """
    if v0.isNone:
        return ""
    if v0.get().len <= max_listed:
        return $(v0.get())
    return invariant_to_str(v0)

proc seq_to_string*[T](arr: seq[T]): string =
    var ans_str = ""
    for item in arr:
        ans_str &= $item & "\n"
    return ans_str

proc dump_random_states*(random_states_list: seq[seq[Rand]],
                         random_states_list_file: string): bool =
    discard """
    Marshals the random states to file, creating 
    the parent directory if needed.

    This is a best-effort diagnostic dump: failures 
    are ignored so they do not invalidate the run.

    Returns: 
        Whether the file was written
    """
    var s = newStringStream("")
    store(s, random_states_list)
    s.setPosition(0)
    try:
        let parent = random_states_list_file.parentDir()
        if parent.len > 0:
            createDir(parent)
        writeFile(random_states_list_file, s.readAll())
        return true
    except OSError, IOError:
        stderr.writeLine(&"WARNING: could not save the random states to {random_states_list_file}: {getCurrentExceptionMsg()}")
        return false

proc evaluate_invariant*[T](
    invariant_name: string,
    include_crossing_signs: bool,
    links: seq[Link[int]],
    link_names: seq[Option[string]],
    values: seq[Option[seq[int]]],
    output: seq[(int, seq[T], seq[Link[int]], seq[Rand])],
    verbose: bool,
    evaluation_file: string,
    random_states_list_file: string,
    detailed_info_for_unknown_values: bool
): Option[float] =
    discard """
    Evaluates the output of a link invariant calculation on
    the known invariant values, distinguishing all the
    possible cases, writes the evaluation to the console or
    a file, and returns the accuracy rate.

    Args:
        invariant_name: the name of the invariant
        include_crossing_signs: whether to include the
            crossing signs along with the PD code
        links: the list of links on which the calculation
            was performed
        link_names: the names of the links on which the
            calculation was performed
        values: the known values of the invariant
        output: a list of answers output by some agent
            calculating the invariant
        verbose: whether to print a table of results and
            analyses
        evaluation_file: the file the evaluation should be
            written to; if set to '', the evaluation is
            printed to the standard output
        random_states_list_file: the file that the list of
            lists of random states just before each end
            action for the links with detailed information
            should be written to
        detailed_info_for_unknown_values: whether detailed
            information should be output for links whose
            invariant values are not given

    Returns:
        The proportion of correctly solved links.

    Raises:
        ValueError: if the length of output does not equal
            the number of links in values, or verbose is
            False but save_file is not '', or values has no
            links with the invariant given
    """
    if values.len == 0:
        raise newException(ValueError, "given invariant values cannot be empty")
    if output.len != values.len:
        raise newException(ValueError, &"output {output} has length {output.len} which does not equal values.len = {values.len}")
    if not verbose and evaluation_file != "":
        raise newException(ValueError, &"verbose must be set to True in order for the {invariant_name} evaluation to be saved to the file {evaluation_file}")
    var link_reps = newSeq[string]()
    for (name, link) in zip(link_names, links):
        link_reps.add(if name.isSome: name.get() else: $(link.PD_code()))
    var result_analyses = newSeq[string]()
    var num_total = links.len
    var detailed_info = newSeq[(string, string, string, string, string)]()
    var random_states_list = newSeq[seq[Rand]]()
    var outcome_counts = [0, 0, 0, 0, 0, 0]
    var suboptimal_diff = newSeq[int]()
    for i in 0 ..< output.len:
        let (answer, actions, link_history, random_states) = output[i]
        let value = values[i]
        let link_rep = link_reps[i]
        var need_detailed_info = false
        var outcome: int
        if answer == -1:
            result_analyses.add(&"Agent failed to find {invariant_name}")
            outcome = 0
        elif value.isNone:
            result_analyses.add(&"{invariant_name} not given")
            need_detailed_info = detailed_info_for_unknown_values
            num_total -= 1
            outcome = 1
        elif answer > value.get()[value.get().len-1]:
            result_analyses.add(&"Agent found suboptimal {invariant_name}")
            suboptimal_diff.add(answer - value.get()[value.get().len-1])
            outcome = 2
        elif answer notin value.get():
            result_analyses.add(&"Agent found illegal {invariant_name} -- BUG IN AGENT!")
            need_detailed_info = true
            outcome = 3
        elif answer == value.get()[value.get().len-1]:
            result_analyses.add(&"Agent verified the current best {invariant_name}")
            outcome = 4
        else:
            result_analyses.add(&"Agent found lower {invariant_name} than current best -- NEW RESULT!")
            need_detailed_info = true
            outcome = 5
        outcome_counts[outcome] += 1
        if need_detailed_info:
            var link_history_info = ""
            for link in link_history:
                var link_info = &"{link.PD_code()}, {link.unlinked_unknot_components}"
                if include_crossing_signs:
                    link_info &= &", {link.signs}"
                link_history_info &= link_info & "\n"
            detailed_info.add((link_rep, known_value_to_display_str(value), $answer, $actions, link_history_info))
            random_states_list.add(random_states)
    var accuracy_rate: Option[float]
    if num_total == 0:
        accuracy_rate = none(float)
    else:
        accuracy_rate = some((outcome_counts[4] +
                              outcome_counts[5]) / num_total)
    var info = ""
    var capitalized_invariant_name = invariant_name.capitalizeAscii()
    if verbose:
        var answers = collect:
            for t in output:
                t[0]
        info &= $["Link Representation", capitalized_invariant_name, "Agent's answer", "Commentary"] & "\n"
        var zipped_results_1 = collect:
            for i in 0 ..< output.len:
                (link_reps[i], known_value_to_display_str(values[i]), answers[i], result_analyses[i])
        info &= seq_to_string(zipped_results_1) & "\n"
        var zipped_results_2 = collect:
            for i in 0 ..< 6:
                (["Failed", "Not given", "Suboptimal",
                  "Illegal", "Current best", "New result"][i],
                 outcome_counts[i],
                 outcome_counts[i] / values.len,
                 if i != 2 or suboptimal_diff.len == 0: "" else: &"Maximum (found - known): {suboptimal_diff[suboptimal_diff.maxIndex()]}\nMean    (found - known): {sum(suboptimal_diff) / suboptimal_diff.len}")
        info &= $["Outcome Label", "Count", "Proportion", "Remark"] & "\n" & seq_to_string(zipped_results_2) & $("Total", values.len, 1, "") & "\n\n"
        info &= &"Accuracy rate: {accuracy_rate}\n"
        if evaluation_file == "":
            stdout.write(info)
    if outcome_counts[3] > 0:
        # An illegal answer is below the known lower bound: 
        # either the agent or the dataset is wrong
        stderr.writeLine(&"WARNING: {outcome_counts[3]} link(s) got an illegal {invariant_name}, below the known lower bound -- BUG IN AGENT (see the detailed information)")
    if detailed_info.len > 0:
        # diagnostic information and new results should
        # always be printed even if verbose is not set
        # marshal the list of the lists of random states first, so the line
        # below reports what actually happened. A lost diagnostic dump must
        # not lose the run with it: the parent directory of the (relative)
        # default path need not exist under a cluster scheduler's cwd.
        let dumped = dump_random_states(random_states_list,
                                        random_states_list_file)
        var text = $["Link Representation",
                     capitalized_invariant_name,
                     "Agent's answer",
                     "Actions",
                     "Link history"] & "\n" & seq_to_string(detailed_info)
        # must keep starting with "The list of random state": results_collector
        # uses that prefix to skip the line when parsing the detailed table.
        text &= &"The list of random state lists has {(if dumped: \"been saved to\" else: \"NOT been saved to\")} file {random_states_list_file}\n"
        if evaluation_file == "":
            styledEcho(fgRed, "BEGINS DETAILED INFORMATION")
            stdout.write(text)
            styledEcho(fgRed, "ENDS DETAILED INFORMATION")
        else:
            info &= "BEGINS DETAILED INFORMATION\n"
            info &= text
            info &= "ENDS DETAILED INFORMATION\n"
    if evaluation_file != "":
        # append to save_file instead of overwriting it
        var f = open(evaluation_file, fmAppend)
        write(f, info)
        close(f)
    return accuracy_rate

discard """
A dataset of links, possibly with associated unknotting
numbers, slice genera, strong slice genera, ribbon genera,
strong ribbon genera, splitting numbers, weak splitting
numbers and names. The data can be read from and written to
Datamancer DataFrames and csv files, and results of link
invariant calculations can be evaluated, with certificates
for new results automatically printed out. The dataset can
be iterated over, sliced and concatenated with other
datasets.
"""
type Dataset* = ref object of RootObj
    links*: seq[Link[int]]
    unknotting_nums*: seq[Option[seq[int]]]
    slice_genera*: seq[Option[seq[int]]]
    strong_slice_genera*: seq[Option[seq[int]]]
    ribbon_genera*: seq[Option[seq[int]]]
    strong_ribbon_genera*: seq[Option[seq[int]]]
    splitting_nums*: seq[Option[seq[int]]]
    weak_splitting_nums*: seq[Option[seq[int]]]
    names*: seq[Option[string]]

proc newDataset*(links: seq[Link[int]],
                 unknotting_nums: seq[Option[seq[int]]],
                 slice_genera: seq[Option[seq[int]]],
                 strong_slice_genera: seq[Option[seq[int]]],
                 ribbon_genera: seq[Option[seq[int]]] = @[],
                 strong_ribbon_genera: seq[Option[seq[int]]] = @[],
                 splitting_nums: seq[Option[seq[int]]] = @[],
                 weak_splitting_nums: seq[Option[seq[int]]] = @[],
                 names: seq[Option[string]] = @[]): Dataset =
    discard """
    Creates a dataset with nine lists: links, unknotting
    numbers, slice genera, strong slice genera, ribbon
    genera, strong ribbon genera, splitting numbers, weak
    splitting numbers, and names. The links describe the
    links in the dataset, and each element of the link
    invariant arrays is a list of the possible values of
    the link invariant of the link at the corresponding
    index. If the list of possible values is unknown, then
    the element is set to none. The elements of the names
    array are the names of the links at the corresponding
    indices. An invariant argument passed as an empty seq
    is filled with none values (the names argument with
    none strings). In general, this method should not be
    invoked directly.

    Args:
        links: list of links in the dataset
        unknotting_nums: list of unknotting numbers of links,
            where a none element means the set of possible
            unknotting numbers is unknown or not provided for
            the corresponding link.
        slice_genera: list of slice genera of links, whose
            elements have the same meaning as above
        strong_slice_genera: list of strong slice genera of
            links, whose elements have the same meaning as
            above
        ribbon_genera: list of ribbon genera of links, whose
            elements have the same meaning as above
        strong_ribbon_genera: list of strong ribbon genera
            of links, whose elements have the same meaning
            as above
        splitting_nums: list of splitting numbers of links
        weak_splitting_nums: list of weak splitting numbers
            of links
        names: list of names of links

    Raises:
        AssertionDefect: if the nine arguments do not
            have the same length, or an empty array (as
            opposed to none) or an unsorted array is
            provided as the link invariant of some link
    """
    proc filled(arr: seq[Option[seq[int]]]): seq[Option[seq[int]]] =
        if arr.len == 0 and links.len > 0:
            return newSeqWith(links.len, none(seq[int]))
        return arr
    let ribbon_genera = filled(ribbon_genera)
    let strong_ribbon_genera = filled(strong_ribbon_genera)
    let splitting_nums = filled(splitting_nums)
    let weak_splitting_nums = filled(weak_splitting_nums)
    var names = names
    if names.len == 0 and links.len > 0:
        names = newSeqWith(links.len, none(string))
    assert (names.len == links.len and
            links.len == unknotting_nums.len and
            unknotting_nums.len == slice_genera.len and
            slice_genera.len == strong_slice_genera.len and
            strong_slice_genera.len == ribbon_genera.len and
            ribbon_genera.len == strong_ribbon_genera.len and
            strong_ribbon_genera.len == splitting_nums.len and
            splitting_nums.len == weak_splitting_nums.len)
    for arr in [unknotting_nums, slice_genera, strong_slice_genera,
                ribbon_genera, strong_ribbon_genera,
                splitting_nums, weak_splitting_nums]:
        for val in arr:
            if val.isSome:
                assert val.get().len > 0 and isSorted(val.get())
    return Dataset(links: links,
                   unknotting_nums: unknotting_nums,
                   slice_genera: slice_genera,
                   strong_slice_genera: strong_slice_genera,
                   ribbon_genera: ribbon_genera,
                   strong_ribbon_genera: strong_ribbon_genera,
                   splitting_nums: splitting_nums,
                   weak_splitting_nums: weak_splitting_nums,
                   names: names)

proc copy*(self: Dataset): Dataset =
    let links_copy = collect:
        for link in self.links:
            link.copy()
    return newDataset(links_copy,
                      self.unknotting_nums,
                      self.slice_genera,
                      self.strong_slice_genera,
                      self.ribbon_genera,
                      self.strong_ribbon_genera,
                      self.splitting_nums,
                      self.weak_splitting_nums,
                      self.names)

proc dataset_from_dataframe*(df0: DataFrame): Dataset =
    discard """
    Created a dataset from a Datamancer dataframe with links
    described using PD code (which is preferred) or name and
    link invariants described as a single number, a range of
    numbers or a set of numbers, if known.

    Args:
        df: the given dataframe with links stored in the "PD
            notation" (or similar names matching the regex,
            same below) column or the "Name" column, and
            possibly the unknotting numbers, slice genera,
            strong slice genera, ribbon genera, strong ribbon
            genera, splitting numbers, and weak splitting
            numbers stored in the "Unknotting Number",
            "Genus-4D", "Strong Slice Genus-4D",
            "Ribbon Genus-4D", "Strong Ribbon Genus-4D",
            "Splitting Number", and "Weak Splitting Number"
            columns.

    Returns:
        A dataset created from the dataframe, with as much
        information filled in as possible.
    """
    var df = df0
    let orig_col_names = df.getKeys()
    for orig_col_name in orig_col_names:
        df = df.rename(f{orig_col_name.toLowerAscii().strip() <- orig_col_name})
    var list_columns = df.getKeys()

    proc choose_column_matching_regex(
        regex: Regex
    ): Option[string] =
        var can_choose = collect:
            for col_name in list_columns:
                if match(col_name, regex): 1 else: 0
        if not anyIt(can_choose, it == 1):
            return none(string)
        if sum(can_choose) > 1:
            raise newException(ValueError, &"multiple columns names in {list_columns} match regex")
        return some(list_columns[can_choose.find(1)])

    let pd_code_col_name = choose_column_matching_regex(re"^pd([ -_]?(code|notation)s?)?( \(vector\))?$")
    let name_col_name = choose_column_matching_regex(re"^names?$")
    let signs_col_name = choose_column_matching_regex(re"^crossing[ -_]?signs?$")
    if pd_code_col_name.isNone and name_col_name.isNone:
        raise newException(ValueError, "no list of links present in the dataframe")
    var links = newSeq[Option[Link[int]]]()
    if pd_code_col_name.isSome:
        # When a "Crossing Signs" column is present, parse the per-link
        # signs in parallel with the PD code and pass them to
        # link_from_PD_code so the orientation is taken from the
        # dataframe rather than auto-deduced. (Mirrors dataset.py.)
        var signs_tensor: seq[string]
        if signs_col_name.isSome:
            signs_tensor = collect:
                for s in df[signs_col_name.get()].toTensor(string): s
        var row_index = 0
        for pd_code_str0 in df[pd_code_col_name.get()].toTensor(string):
            if pd_code_str0.strip() != "":
                var pd_code_str = pd_code_str0.multiReplace([(";", ","), ("{", "["), ("}", "]")])
                var pd_code_arr = newSeq[array[0..3, int]]()
                for item_node in parseJson(pd_code_str).getElems():
                    var item = [-1, -1, -1, -1]
                    for (i, num_node) in enumerate(item_node.getElems()):
                        item[i] = num_node.getInt()
                    pd_code_arr.add(item)
                var signs_arr = newSeq[int]()
                if signs_col_name.isSome and signs_tensor[row_index].strip() != "":
                    let signs_str = signs_tensor[row_index].multiReplace([(";", ","), ("{", "["), ("}", "]")])
                    for num_node in parseJson(signs_str).getElems():
                        signs_arr.add(num_node.getInt())
                    if signs_arr.len != pd_code_arr.len:
                        raise newException(ValueError, &"crossing signs length {signs_arr.len} != PD code length {pd_code_arr.len} at row {row_index}")
                links.add(some(link_from_PD_code(pd_code_arr, signs_arr)))
            else:
                links.add(none(Link[int]))
            row_index += 1
    else:
        links = newSeqWith(df.len, none(Link[int]))
    var names: seq[Option[string]]
    if name_col_name.isSome:
        names = collect:
            for name in df[name_col_name.get()].toTensor(string):
                if name == "": none(string) else: some(name)
        for (i, name) in enumerate(names):
            if links[i].isNone:
                raise newException(AssertionDefect, "name_to_link is not implemented")
    else:
        names = newSeqWith(links.len, none(string))
    for link in links:
        if link.isNone:
            raise newException(ValueError, "dataframe does not specify every link with PD code or name")
    var links_unwrapped = collect:
        for l in links:
            l.get()

    proc load_invariant_col(
        regex: Regex
    ): seq[Option[seq[int]]] =
        var invariant_col_name = choose_column_matching_regex(regex)
        if invariant_col_name.isSome:
            return collect:
                for c in df[invariant_col_name.get()].toTensor(string):
                    if c == "":
                        none(seq[int])
                    else:
                        parse_invariant(c)
        return newSeqWith(links.len, none(seq[int]))

    var unknotting_nums = load_invariant_col(re"^un(knot(ting)?|link(ing)?)[ -_]?num(ber)?s?$")
    var slice_genera = load_invariant_col(re"^(smooth[ -_]?)?(slice[ -_]?)?(genus|genera)([ -_]?4d)?$")
    var strong_slice_genera = load_invariant_col(re"^strong[ -_]?slice([ -_]?(genus|genera)([ -_]?4d)?)?$")
    var ribbon_genera = load_invariant_col(re"^(smooth[ -_]?)?ribbon[ -_]?(genus|genera)([ -_]?4d)?$")
    var strong_ribbon_genera = load_invariant_col(re"^strong[ -_]?ribbon([ -_]?(genus|genera)([ -_]?4d)?)?$")
    var splitting_nums = load_invariant_col(re"^split(ting)?[ -_]?num(ber)?s?$")
    var weak_splitting_nums = load_invariant_col(re"^weak[ -_]?split(ting)?[ -_]?num(ber)?s?$")
    return newDataset(links_unwrapped,
                      unknotting_nums, slice_genera,
                      strong_slice_genera,
                      ribbon_genera, strong_ribbon_genera,
                      splitting_nums, weak_splitting_nums,
                      names)

proc dataset_read_csv*(filename: string): Dataset =
    discard """
    Constructs a Dataset object by reading in a csv
    file.

    Args:
        filename: path to the csv file

    Returns:
        The constructed dataset.
    """
    return dataset_from_dataframe(readCsv(filename))

proc `[]`*(self: Dataset, x: HSlice[int, int]): Dataset =
    discard """
    Constructs a new dataset from a HSlice object.

    Args:
        key: given slice

    Returns:
        New dataset containing data specified by the slice
        object
    """
    return newDataset(self.links[x],
                      self.unknotting_nums[x],
                      self.slice_genera[x],
                      self.strong_slice_genera[x],
                      self.ribbon_genera[x],
                      self.strong_ribbon_genera[x],
                      self.splitting_nums[x],
                      self.weak_splitting_nums[x],
                      self.names[x])

proc evaluate_invariant*[T](
    self: Dataset,
    framework: string,
    output: seq[(int, seq[T], seq[Link[int]], seq[Rand])],
    verbose: bool = true,
    evaluation_file: string = "",
    random_states_list_file: string = "../outputs/random-states.json",
    detailed_info_for_unknown_values: bool = false
): Option[float] =
    var (inv_n, inc_cross_sgns, vals) = (
        {"unknotting": ("unknotting number",
                       true,
                       self.unknotting_nums),
         "slice": ("slice genus",
                   true,
                   self.slice_genera),
         "strong slice": ("strong slice genus",
                          true,
                          self.strong_slice_genera),
         "ribbon": ("ribbon genus",
                    true,
                    self.ribbon_genera),
         "strong ribbon": ("strong ribbon genus",
                           true,
                           self.strong_ribbon_genera),
         "splitting": ("splitting number",
                       false,
                       self.splitting_nums),
         "weak splitting": ("weak splitting number",
                            false,
                            self.weak_splitting_nums)
        }.toTable()[framework]
    )
    return evaluate_invariant(
        inv_n, inc_cross_sgns, self.links, self.names,
        vals, output, verbose, evaluation_file,
        random_states_list_file,
        detailed_info_for_unknown_values
    )
