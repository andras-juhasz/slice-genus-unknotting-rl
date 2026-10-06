from typing import List, Optional, Tuple
from copy import deepcopy
import math
import os
import re
import json
import random
import statistics
import warnings

import pandas as pd
from spherogram import Link
from tabulate import tabulate, SEPARATING_LINE
from colorist import red

from utils import is_sorted, link_from_crossings


def _normalize_pd_labels(
        pd_list: List[List[int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Remap a PD code's edge labels to the contiguous 
    range 0..2n-1 without changing the diagram.
    """
    label_map: dict = {}
    normalized: List[Tuple[int, int, int, int]] = []
    for crossing in pd_list:
        new_crossing = []
        for edge in crossing:
            if edge not in label_map:
                label_map[edge] = len(label_map)
            new_crossing.append(label_map[edge])
        normalized.append(tuple(new_crossing))
    return normalized


def name_to_link(name: str) -> Link:
    """
    Converts the name of a link to a Link object
    representing the link. Names like 13n_3370 are
    automatically stripped of the underline, and the name
    0_1 is taken to represent the unknot.

    Args:
        name: the name of a link

    Returns:
        A Link object representing the link.

    Raises:
        ValueError: if name is empty
    """
    name = name.strip()
    if name == '':
        raise ValueError('name is empty')
    if re.match(r'^\d+[a-z]_\d+$', name) is not None:
        name = name.replace('_', '')
    if name == '0_1':
        unknot = Link([])
        unknot.unlinked_unknot_components = 1
        return unknot
    return Link(name)


def convert_float_invariant(c: float | str | int) -> str:
    if isinstance(c, float):
        if math.isinf(c):
            return 'inf'
        c = int(c)
    return str(c)


def parse_invariant(rep: str) -> Optional[List[int | float]]:
    """
    Converts a string representation of the value of a link
    invariant to a sorted list of possible values for the
    link invariant or None (if the value is not given).

    Args:
        rep: the string representation of the value of a
            link invariant; may be an empty string (meaning
            that the value is not given), the literal 'inf'
            (meaning the invariant is infinite), a
            single integer, a range of integers, or a set of
            integers

    Returns:
        A sorted list of the possible values for the link
        invariant. For 'inf', returns [math.inf].

    Raises:
        ValueError: if the string represents an illegal
            range or is not of one of the recognized forms
    """
    rep = rep.strip().replace(';', ',')
    if rep == '':
        return None
    if rep.lower() in ('inf', 'infty', 'infinity', '∞'):
        return [math.inf]
    if re.match(r'^\d+$', rep) is not None:
        return [int(rep)]
    if re.match(r'^\[\d+,\d+]$', rep) is not None:
        match = re.match(r'\[(\d+),(\d+)]$', rep)
        low_range = int(match.group(1))
        high_range = int(match.group(2))
        if low_range > high_range:
            raise ValueError(f'the range {rep} is illegal')
        return list(range(low_range, high_range+1))
    if re.match(r'^\{(\d+,)*\d+}$', rep) is not None:
        return sorted(json.loads(f'[{rep[1:-1]}]'))
    raise ValueError(f'representation {rep} cannot be '
                     f'parsed because it is not empty, a '
                     f'single integer, a range of '
                     f'integers, or a set of integers')


def convert_invariant(
        v: int | float | Tuple[int, int] | List[int | float] | None
) -> List[int | float] | None:
    """
    Converts a link invariant to the standard form, namely a
    sorted list of integers (or math.inf) or None.

    Args:
        v: the link invariant, which can be a single
            integer, math.inf (the invariant is infinite),
            a pair representing a range of integers, a list
            of integers (possibly including math.inf), or
            None

    Returns:
        If v is a single integer or math.inf, returns a
        singleton list containing v. If v is a pair of
        integers, returns a list of elements of the range
        represented by v. If v is a list, returns the sorted
        list of v. If v is None, returns None.

    Raises:
        ValueError: if v is a pair of integers representing
            an illegal range
    """
    if isinstance(v, int):
        return [v]
    if isinstance(v, float):
        # Only math.inf is a valid float invariant value.
        if math.isinf(v):
            return [v]
        raise ValueError(f'float invariant {v} is not math.inf')
    if isinstance(v, tuple):
        low_range, high_range = v
        if low_range > high_range:
            raise ValueError(f'the range {v} is illegal')
        return list(range(low_range, high_range+1))
    if isinstance(v, list):
        return sorted(v)
    return v


def invariant_to_str(v: Optional[List[int | float]]) -> str:
    """
    Converts the value of some link invariant to a string.

    Args:
        v: the value of a link invariant

    Returns:
        The string representation of v. If v is None,
        returns the empty string. A single value is
        preferred to a range of integers, which in turn is
        preferred to a set of integers. The literal 'inf'
        is emitted for math.inf (parsed back by
        parse_invariant).
    """
    if v is None:
        return ''
    if len(v) == 1:
        x = v[0]
        if isinstance(x, float) and math.isinf(x):
            return 'inf'
        return str(x)
    if all(isinstance(v[i], int) for i in range(len(v))) and \
            all(v[i+1] == v[i]+1 for i in range(len(v)-1)):
        return f'[{v[0]};{v[-1]}]'
    return '{' + ';'.join(
        ['inf' if isinstance(n, float) and math.isinf(n) else str(n)
         for n in v]) + '}'


def known_value_to_display_str(
        v: Optional[List[int | float]],
        max_listed: int = 10) -> str:
    """
    Formats a known invariant value for display in
    evaluation output, it avoids dumping e.g. the whole
    ``[0, 1, ..., 9999]`` placeholder range for invariants
    whose value is effectively unconstrained.

    Args:
        v: the known invariant value, as a list of the
            possible values (or None if unknown)
        max_listed: the largest number of possible values
            that is still printed in full; above this, the
            compact range form is used

    Returns:
        The display string. None becomes the empty string;
        at most ``max_listed`` values are listed verbatim,
        otherwise the compact ``invariant_to_str`` form is
        returned.
    """
    if v is None:
        return ''
    if len(v) <= max_listed:
        return str(v)
    return invariant_to_str(v)


def _dump_random_states(
        random_states_list: List[List[str]],
        random_states_list_file: str
) -> bool:
    """
    Marshals the random states to file, creating 
    the parent directory if needed.

    This is a best-effort diagnostic dump: failures 
    are ignored so they do not invalidate the run.
    The default ``random_states_list_file`` is a 
    relative path.

    Returns:
        Whether the file was written.
    """
    parent = os.path.dirname(os.path.abspath(random_states_list_file))
    try:
        os.makedirs(parent, exist_ok=True)
        with open(random_states_list_file, 'w') as f:
            print(json.dumps(random_states_list), file=f)
        return True
    except OSError as e:
        warnings.warn(f'could not save the random states to '
                      f'{random_states_list_file}: {e}')
        return False


def evaluate_invariant(
        invariant_name: str,
        include_crossing_signs: bool,
        links: List[Link],
        link_names: List[Optional[str]],
        values: List[Optional[List[int]]],
        output: List[Tuple[int, List[str], List[Link], List[str]]] |
                List[Tuple[int, List[Tuple[str, float]], List[Link], List[str]]],
        verbose: bool,
        evaluation_file: str,
        random_states_list_file: str,
        detailed_info_for_unknown_values: bool
) -> float | None:
    """
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
    if not values:
        raise ValueError(f'given invariant values cannot '
                         f'be empty')
    if len(output) != len(values):
        raise ValueError(f'output {output} has length '
                         f'{len(output)} which does not '
                         f'equal len(values) = '
                         f'{len(values)}')
    if not verbose and evaluation_file != '':
        raise ValueError(f'verbose must be set to True in '
                         f'order for the {invariant_name} '
                         f'evaluation to be saved to the '
                         f'file {evaluation_file}')
    link_reps: List[str] = []
    for name, link in zip(link_names, links):
        link_reps.append(name if name is not None else str(link.PD_code()))
    result_analyses = []
    num_total = len(links)
    detailed_info: List[Tuple[str, str, int, str, str]] = []
    random_states_list: List[List[str]] = []
    outcome_counts = [0] * 6
    suboptimal_diff = []
    for (answer, actions, link_history, random_states), value, link_rep in zip(output, values, link_reps):
        need_detailed_info = False
        if answer == -1:
            result_analyses.append(f'Agent failed to find '
                                   f'{invariant_name}')
            outcome = 0
        elif value is None:
            result_analyses.append(f'{invariant_name} not '
                                   f'given')
            need_detailed_info = detailed_info_for_unknown_values
            num_total -= 1
            outcome = 1
        elif answer > value[-1]:
            result_analyses.append(f'Agent found '
                                   f'suboptimal '
                                   f'{invariant_name}')
            suboptimal_diff.append(answer - value[-1])
            outcome = 2
        elif answer not in value:
            result_analyses.append(f'Agent found illegal '
                                   f'{invariant_name} -- '
                                   f'BUG IN AGENT!')
            need_detailed_info = True
            outcome = 3
        elif answer == value[-1]:
            result_analyses.append(f'Agent verified the '
                                   f'current best '
                                   f'{invariant_name}')
            outcome = 4
        else:
            result_analyses.append(f'Agent found lower '
                                   f'{invariant_name} than '
                                   f'current best -- NEW '
                                   f'RESULT!')
            need_detailed_info = True
            outcome = 5
        outcome_counts[outcome] += 1
        if need_detailed_info:
            link_history_info = ''
            for link in link_history:
                link_info = (f'{link.PD_code()}, '
                             f'{link.unlinked_unknot_components}')
                if include_crossing_signs:
                    link_info += f', {[c.sign for c in link.crossings]}'
                link_history_info += link_info + '\n'
            detailed_info.append((link_rep,
                                  known_value_to_display_str(value),
                                  answer,
                                  str(actions),
                                  link_history_info))
            random_states_list.append(random_states)
    if num_total == 0:
        accuracy_rate = None
    else:
        accuracy_rate = (outcome_counts[4] +
                         outcome_counts[5]) / num_total
    info = ''
    capitalized_invariant_name = invariant_name[0].upper() + invariant_name[1:]
    if verbose:
        answers = [t[0] for t in output]
        values_display = [known_value_to_display_str(value)
                          for value in values]
        info += tabulate(zip(link_reps, values_display, answers,
                             result_analyses),
                         headers=['Link Representation',
                                  capitalized_invariant_name,
                                  'Agent\'s answer',
                                  'Commentary']) + '\n\n'
        outcome_summary = list(filter(
            lambda t: t[1] > 0,
            zip(['Failed', 'Not given', 'Suboptimal',
                 'Illegal', 'Current best', 'New result'],
                outcome_counts,
                [c / len(values) for c in outcome_counts],
                ['', '',
                 f'Maximum (found - known): '
                 f'{max(suboptimal_diff)}\n'
                 f'Mean    (found - known): '
                 f'{statistics.mean(suboptimal_diff)}'
                 if suboptimal_diff else '',
                 '', '', ''])
        )) + [SEPARATING_LINE,
              ('Total', len(values), 1, '')]
        info += tabulate(outcome_summary,
                         headers=['Outcome Label',
                                  'Count',
                                  'Proportion',
                                  'Remark']) + '\n\n'
        info += f'Accuracy rate: {accuracy_rate}\n'
        if evaluation_file == '':
            print(info, end='')
    if outcome_counts[3]:
        # An illegal answer is below the known lower bound: 
        # either the agent or the dataset is wrong..
        warnings.warn(f'{outcome_counts[3]} link(s) got an illegal '
                      f'{invariant_name}, below the known lower bound '
                      f'-- BUG IN AGENT (see the detailed information)')
    if detailed_info:
        # diagnostic information and new results should
        # always be printed even if verbose is not set
        # dump the list of the lists of random states first, so the line
        # below reports what actually happened. It must keep starting with
        # "The list of random state", which results_collector uses to skip
        # it when parsing the detailed table back out of the .out file.
        dumped = _dump_random_states(random_states_list,
                                     random_states_list_file)
        text = tabulate(detailed_info,
                        headers=['Link Representation',
                                 capitalized_invariant_name,
                                 'Agent\'s answer',
                                 'Actions',
                                 'Link history'])
        text += (f'\nThe list of random state lists has '
                 f'{"been saved to" if dumped else "NOT been saved to"} file '
                 f'{random_states_list_file}')
        if evaluation_file == '':
            red('BEGINS DETAILED INFORMATION')
            print(text)
            red('ENDS DETAILED INFORMATION')
        else:
            info += 'BEGINS DETAILED INFORMATION\n'
            info += text + '\n'
            info += 'ENDS DETAILED INFORMATION\n'
    if evaluation_file != '':
        # append to save_file instead of overwriting it
        with open(evaluation_file, 'a') as f:
            print(info, end='', file=f)
    return accuracy_rate


class Dataset:
    """
    A dataset of links, possibly with associated unknotting
    numbers, splitting numbers, weak splitting numbers,
    slice genera, strong slice genera, ribbon genera,
    strong ribbon genera and names. The four genus
    invariants are distinct: the (strong) slice genera are
    the targets of the band-move frameworks with the
    unknot-birth move, while the (strong) ribbon genera are
    the targets of the band-move-only frameworks. The data
    can be read from and written to pandas DataFrames and
    csv files, and results of link invariant calculations
    can be evaluated, with certificates for new results
    automatically printed out. The dataset can be iterated
    over, sliced and concatenated with other datasets.
    """
    def __init__(
            self,
            links: List[Link],
            unknotting_nums: List[Optional[List[int]]],
            slice_genera: List[Optional[List[int]]],
            strong_slice_genera: List[Optional[List[int]]],
            ribbon_genera: List[Optional[List[int]]],
            strong_ribbon_genera: List[Optional[List[int]]],
            splitting_nums: List[Optional[List[int]]],
            weak_splitting_nums: List[Optional[List[int]]],
            names: List[Optional[str]]
    ):
        """
        Creates a dataset with nine lists: links, unknotting
        numbers, slice genera, strong slice genera, ribbon
        genera, strong ribbon genera, splitting numbers,
        weak splitting numbers, and names. The links
        describe the links in the dataset, and each element of
        the link invariant arrays is a list of the possible
        values of the link invariant of the link at the
        corresponding index. If the list of possible values
        is unknown, then the element is set to None. The
        elements of the names array are the names of the
        links at the corresponding indices. In general, this
        method should not be invoked directly.

        Args:
            links: list of links in the dataset
            unknotting_nums: list of unknotting numbers of
                links, where a None element means the set of
                possible unknotting numbers is unknown or
                not provided for the corresponding link.
            slice_genera: list of slice genera of links,
                whose elements have the same meaning as
                above
            strong_slice_genera: list of strong slice genera
                of links, whose elements have the same meaning
                as above
            ribbon_genera: list of ribbon genera of links,
                whose elements have the same meaning as
                above
            strong_ribbon_genera: list of strong ribbon
                genera of links, whose elements have the
                same meaning as above
            splitting_nums: list of splitting numbers of
                links
            weak_splitting_nums: list of weak splitting
                numbers of links
            names: list of names of links

        Raises:
            AssertionError: if the nine arguments do not
                have the same length, or an empty array (as
                opposed to None) or an unsorted array is
                provided as the link invariant of some link
        """
        assert (len(names) == len(links) ==
                len(unknotting_nums) ==
                len(slice_genera) == len(strong_slice_genera) ==
                len(ribbon_genera) == len(strong_ribbon_genera) ==
                len(splitting_nums) == len(weak_splitting_nums))
        for arr in [unknotting_nums, slice_genera, strong_slice_genera,
                    ribbon_genera, strong_ribbon_genera,
                    splitting_nums, weak_splitting_nums]:
            for val in arr:
                if val is not None:
                    assert val != [] and is_sorted(val)
        self.links = links
        self.unknotting_nums = unknotting_nums
        self.slice_genera = slice_genera
        self.strong_slice_genera = strong_slice_genera
        self.ribbon_genera = ribbon_genera
        self.strong_ribbon_genera = strong_ribbon_genera
        self.splitting_nums = splitting_nums
        self.weak_splitting_nums = weak_splitting_nums
        self.names = names

    def copy(self) -> 'Dataset':
        return Dataset(deepcopy(self.links),
                       deepcopy(self.unknotting_nums),
                       deepcopy(self.slice_genera),
                       deepcopy(self.strong_slice_genera),
                       deepcopy(self.ribbon_genera),
                       deepcopy(self.strong_ribbon_genera),
                       deepcopy(self.splitting_nums),
                       deepcopy(self.weak_splitting_nums),
                       deepcopy(self.names))

    @staticmethod
    def from_dataframe(df: pd.DataFrame,
                       log_deduced_signs: bool = False) -> 'Dataset':
        """
        Created a dataset from a pandas dataframe with links
        described using PD code (which is preferred) or name
        and link invariants described as a single number, a
        range of numbers or a set of numbers, if known.

        If a "Crossing Signs" column is present, it is parsed
        alongside the PD code so that the link's orientation is
        taken from the dataframe rather than deduced. When it is
        absent, spherogram deduces the crossing signs from the
        bare PD code (keeping the PD verbatim) and a one-line
        summary is printed; set log_deduced_signs to also print
        the PD code and deduced signs of every such link.

        Args:
            df: the given dataframe with links stored in the
                "PD notation" (or similar names matching the
                regex, same below) column or the "Name"
                column, and possibly the unknotting numbers,
                slice genera, strong slice genera, ribbon
                genera, strong ribbon genera, splitting
                numbers, and weak splitting numbers stored
                in the "Unknotting Number", "Genus-4D",
                "Strong Slice Genus-4D", "Ribbon Genus-4D",
                "Strong Ribbon Genus-4D", "Splitting
                Number", and "Weak Splitting Number"
                columns.
                The optional "Crossing Signs" column gives, 
                per link, the list of +-1 crossing signs in 
                the same order as the PD code.
            log_deduced_signs: if True, print the PD code and
                deduced signs of every link whose signs were not
                provided in the dataframe (otherwise only a
                summary count is printed).

        Returns:
            A dataset created from the dataframe, with as
            much information filled in as possible.
        """
        df = df.copy(deep=True)
        df.columns = [x.lower().strip() for x in df.columns]
        list_columns: List[str] = list(df.columns)

        def choose_column_matching_regex(
                regex: str
        ) -> Optional[str]:
            can_choose = [re.match(regex, col_name) is not None
                          for col_name in list_columns]
            if not any(can_choose):
                return None
            if sum(can_choose) > 1:
                raise ValueError(f'multiple columns names '
                                 f'in {list_columns} match '
                                 f'regex {regex}')
            return list_columns[can_choose.index(True)]
        pd_code_col_name = choose_column_matching_regex(r'^pd([ -_]?(code|notation)s?)?( \(vector\))?$')
        name_col_name = choose_column_matching_regex(r'^names?$')
        signs_col_name = choose_column_matching_regex(r'^crossing[ -_]?signs?$')
        if pd_code_col_name is None and name_col_name is None:
            raise ValueError('no list of links present in '
                             'the dataframe')
        # Number of links whose crossing signs were deduced by
        # spherogram because the dataframe did not provide them
        # (or provided them only partially). See the summary log
        # printed after the loop below.
        num_signs_deduced = 0
        deduced_details: List[str] = []
        links = []
        if pd_code_col_name is not None:
            signs_col = (df[signs_col_name]
                         if signs_col_name is not None else None)
            for row_index, PD_code in enumerate(df[pd_code_col_name]):
                if not pd.isna(PD_code) and PD_code.strip() != '':
                    pd_list = json.loads(PD_code.replace(';', ',').replace('{', '[').replace('}', ']'))
                    # Default: let spherogram deduce signs from a bare
                    # PD. This keeps the PD code verbatim and assigns
                    # deterministic, all-nonzero signs.
                    link = Link(pd_list)
                    deduced_signs = [c.sign for c in link.crossings]
                    stored_signs = None
                    if signs_col is not None:
                        cell = signs_col.iloc[row_index]
                        if not pd.isna(cell) and str(cell).strip() != '':
                            stored_signs = json.loads(str(cell).replace(';', ',').replace('{', '[').replace('}', ']'))
                    if stored_signs is None:
                        # Signs not provided: record that spherogram
                        # deduced them, so the caller can verify / freeze
                        # the orientation on the next write.
                        num_signs_deduced += 1
                        if name_col_name is not None:
                            row_name = df[name_col_name].iloc[row_index]
                            link_id = (f'row {row_index}'
                                       if pd.isna(row_name)
                                       else str(row_name))
                        else:
                            link_id = f'row {row_index}'
                        deduced_details.append(
                            f'  {link_id}:\n'
                            f'    PD:    {pd_list}\n'
                            f'    signs: {deduced_signs}')
                    elif stored_signs != deduced_signs:
                        # A deliberately non-default orientation is
                        # stored, we rebuild from explicit signs
                        if len(stored_signs) != len(pd_list):
                            raise ValueError(
                                f'crossing signs length {len(stored_signs)} '
                                f'!= PD code length {len(pd_list)} at row '
                                f'{row_index}')
                        link = link_from_crossings(
                            _normalize_pd_labels(pd_list), stored_signs)
                    # else: stored_signs == deduced_signs -> keep the
                    # verbatim-PD Link built above.
                    links.append(link)
                else:
                    links.append(None)
            if num_signs_deduced > 0:
                print(f'[Dataset] crossing signs not in CSV -- deduced by '
                      f'spherogram for {num_signs_deduced}/{len(links)} links')
                if log_deduced_signs:
                    for detail in deduced_details:
                        print(detail)
        else:
            links = [None] * len(df)
        if name_col_name is not None:
            names = [None if pd.isna(name) else str(name)
                     for name in df[name_col_name]]
            for i, name in enumerate(names):
                if links[i] is None:
                    links[i] = name_to_link(name)
        else:
            names = [None] * len(links)
        for link in links:
            if link is None:
                raise ValueError('dataframe does not '
                                 'specify every link with '
                                 'PD code or name')

        def load_invariant_col(
                regex: str
        ) -> List[Optional[List[int]]]:
            invariant_col_name = choose_column_matching_regex(regex)
            if invariant_col_name is not None:
                # we need to convert float invariants
                # because if a column contains both
                # single-integer invariant values and empty
                # strings, then the corresponding column of
                # its pandas DataFrame would be of type
                # float, with empty strings represented by
                # NaNs
                return [None if pd.isna(c) else
                        parse_invariant(convert_float_invariant(c))
                        for c in df[invariant_col_name]]
            return [None] * len(links)
        unknotting_nums = load_invariant_col(r'^un(knot(ting)?|link(ing)?)[ -_]?num(ber)?s?$')
        slice_genera = load_invariant_col(r'^(smooth[ -_]?)?(slice[ -_]?)?(genus|genera)([ -_]?4d)?$')
        strong_slice_genera = load_invariant_col(r'^strong[ -_]?slice([ -_]?(genus|genera)([ -_]?4d)?)?$')
        ribbon_genera = load_invariant_col(r'^(smooth[ -_]?)?ribbon[ -_]?(genus|genera)([ -_]?4d)?$')
        strong_ribbon_genera = load_invariant_col(r'^strong[ -_]?ribbon([ -_]?(genus|genera)([ -_]?4d)?)?$')
        splitting_nums = load_invariant_col(r'^split(ting)?[ -_]?num(ber)?s?$')
        weak_splitting_nums = load_invariant_col(r'^weak[ -_]?split(ting)?[ -_]?num(ber)?s?$')
        return Dataset(links, unknotting_nums,
                       slice_genera, strong_slice_genera,
                       ribbon_genera, strong_ribbon_genera,
                       splitting_nums, weak_splitting_nums,
                       names)

    @staticmethod
    def from_links(
            links: List[Link | str | List[Tuple[int, int, int, int]]],
            unknotting_nums: List[int | Tuple[int, int] | List[int] | None] = None,
            slice_genera: List[int | Tuple[int, int] | List[int] | None] = None,
            strong_slice_genera: List[int | Tuple[int, int] | List[int] | None] = None,
            ribbon_genera: List[int | Tuple[int, int] | List[int] | None] = None,
            strong_ribbon_genera: List[int | Tuple[int, int] | List[int] | None] = None,
            splitting_nums: List[int | Tuple[int, int] | List[int] | None] = None,
            weak_splitting_nums: List[int | Tuple[int, int] | List[int] | None] = None,
            names: List[Optional[str]] = None,
            crossing_signs: List[Optional[List[int]]] = None
    ) -> 'Dataset':
        """
        Constructs a Dataset object from a list of links.

        Args:
            links: the list of links; each link can be
                specified by a Link object, the link's name
                or its PD code
            crossing_signs: optionally, the crossing signs to
                use for the links given as a PD code. When given
                for a link, that link is built from the explicit
                signs (uniquely fixing its orientation); when
                None, spherogram deduces the signs from the bare
                PD code. Ignored for links given as a Link object
                or a name.
            unknotting_nums: the links' unknotting numbers;
                each unknotting number can be specified by
                a single integer, a pair of integers
                indicating a range, a list of integers or
                None indicating that the unknotting number
                is not given
            slice_genera: the links' slice genera; each
                slice genus is specified in the same manner
                as the unknotting number
            strong_slice_genera: the links' strong slice
                genera
            ribbon_genera: the links' ribbon genera
            strong_ribbon_genera: the links' strong ribbon
                genera
            splitting_nums: the links' splitting numbers
            weak_splitting_nums: the links' weak splitting
                numbers
            names: the links' names; each name can be a
                string or None

        Returns:
            A Dataset object containing the list of links.
        """
        if unknotting_nums is None:
            unknotting_nums = [None] * len(links)
        if slice_genera is None:
            slice_genera = [None] * len(links)
        if strong_slice_genera is None:
            strong_slice_genera = [None] * len(links)
        if ribbon_genera is None:
            ribbon_genera = [None] * len(links)
        if strong_ribbon_genera is None:
            strong_ribbon_genera = [None] * len(links)
        if names is None:
            names = [None] * len(links)
        if splitting_nums is None:
            splitting_nums = [None] * len(links)
        if weak_splitting_nums is None:
            weak_splitting_nums = [None] * len(links)
        if crossing_signs is None:
            crossing_signs = [None] * len(links)
        links = deepcopy(links)
        unknotting_nums = deepcopy(unknotting_nums)
        slice_genera = deepcopy(slice_genera)
        strong_slice_genera = deepcopy(strong_slice_genera)
        ribbon_genera = deepcopy(ribbon_genera)
        strong_ribbon_genera = deepcopy(strong_ribbon_genera)
        splitting_nums = deepcopy(splitting_nums)
        weak_splitting_nums = deepcopy(weak_splitting_nums)
        names = deepcopy(names)
        for i in range(len(links)):
            if names[i] is None and isinstance(links[i], str):
                names[i] = links[i]
            if isinstance(links[i], str):
                links[i] = name_to_link(links[i])
            elif isinstance(links[i], list):
                if crossing_signs[i] is not None:
                    if len(crossing_signs[i]) != len(links[i]):
                        raise ValueError(
                            f'crossing signs length '
                            f'{len(crossing_signs[i])} != PD code length '
                            f'{len(links[i])} for link {i}')
                    links[i] = link_from_crossings(
                        _normalize_pd_labels(links[i]), crossing_signs[i])
                else:
                    links[i] = Link(links[i])
            unknotting_nums[i] = convert_invariant(unknotting_nums[i])
            slice_genera[i] = convert_invariant(slice_genera[i])
            strong_slice_genera[i] = convert_invariant(strong_slice_genera[i])
            ribbon_genera[i] = convert_invariant(ribbon_genera[i])
            strong_ribbon_genera[i] = convert_invariant(strong_ribbon_genera[i])
            splitting_nums[i] = convert_invariant(splitting_nums[i])
            weak_splitting_nums[i] = convert_invariant(weak_splitting_nums[i])
        return Dataset(links, unknotting_nums,
                       slice_genera, strong_slice_genera,
                       ribbon_genera, strong_ribbon_genera,
                       splitting_nums, weak_splitting_nums,
                       names)

    @staticmethod
    def from_link(
            link: Link | str | List[Tuple[int, int, int, int]],
            unknotting_num: int | Tuple[int, int] | List[int] | None = None,
            slice_genus: int | Tuple[int, int] | List[int] | None = None,
            strong_slice_genus: int | Tuple[int, int] | List[int] | None = None,
            ribbon_genus: int | Tuple[int, int] | List[int] | None = None,
            strong_ribbon_genus: int | Tuple[int, int] | List[int] | None = None,
            splitting_number: int | Tuple[int, int] | List[int] | None = None,
            weak_splitting_number: int | Tuple[int, int] | List[int] | None = None,
            name: Optional[str] = None
    ) -> "Dataset":
        """
        Constructs a Dataset object from a single link.

        Args:
            link: the link; can be specified with a Link
                object, its name or PD code
            unknotting_num: the link's unknotting number;
                can be specified with a single integer, a
                pair of integers indicating a range, a list
                of integers or None indicating that the
                unknotting number is not given
            slice_genus: the link's slice genus; specified
                in the same manner as its unknotting number
            strong_slice_genus: the link's strong slice genus
            ribbon_genus: the link's ribbon genus
            strong_ribbon_genus: the link's strong ribbon
                genus
            splitting_number: the link's splitting number
            weak_splitting_number: the link's weak splitting
                number
            name: the link's name

        Returns:
            A Dataset object containing the given link.
        """
        return Dataset.from_links(
            [link], [unknotting_num],
            [slice_genus], [strong_slice_genus],
            [ribbon_genus], [strong_ribbon_genus],
            [splitting_number], [weak_splitting_number],
            [name]
        )

    @staticmethod
    def read_csv(filename: str) -> "Dataset":
        """
        Constructs a Dataset object by reading in a csv
        file.

        Args:
            filename: path to the csv file

        Returns:
            The constructed dataset.
        """
        return Dataset.from_dataframe(pd.read_csv(filename))

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts self to a pandas DataFrame. The PD code is
        always included in the "PD Notation" column. If some
        name is not None, a "Name" column is prepended. If
        some unknotting number is not None, an "Unknotting
        Number" column is appended. The same goes for the
        "Genus-4D" (slice genus), "Strong Slice Genus-4D",
        "Ribbon Genus-4D", "Strong Ribbon Genus-4D",
        "Splitting Number" and "Weak Splitting Number"
        columns.

        Returns:
            A pandas DataFrame object representing the links
            in self.
        """
        df = pd.DataFrame({'PD Notation': [str([list(t) for t in link.PD_code()]).replace(' ', '').replace(',', ';') for link in self.links]})
        # Always persist the crossing signs next to the PD code so the
        # link's full oriented diagram round-trips exactly: (PD, signs)
        # together define the diagram uniquely.
        df.insert(1, 'Crossing Signs',
                  [str([c.sign for c in link.crossings]).replace(' ', '').replace(',', ';')
                   for link in self.links])
        if any(name is not None for name in self.names):
            df.insert(0, 'Name',
                      ['' if name is None else name
                       for name in self.names])

        def ins_inv_col(
                inv: List[Optional[List[int]]],
                col_name: str
        ) -> None:
            if any(n is not None for n in inv):
                df.insert(len(df.columns), col_name,
                          [invariant_to_str(v)
                           for v in inv])

        ins_inv_col(self.unknotting_nums, 'Unknotting Number')
        ins_inv_col(self.slice_genera, 'Genus-4D')
        ins_inv_col(self.strong_slice_genera, 'Strong Slice Genus-4D')
        ins_inv_col(self.ribbon_genera, 'Ribbon Genus-4D')
        ins_inv_col(self.strong_ribbon_genera, 'Strong Ribbon Genus-4D')
        ins_inv_col(self.splitting_nums, 'Splitting Number')
        ins_inv_col(self.weak_splitting_nums, 'Weak Splitting Number')
        return df

    def write_csv(self, filename: str) -> None:
        """
        Writes self to a csv file.

        Args:
            filename: path to the csv file
        """
        self.to_dataframe().to_csv(filename, index=False)

    def to_list(self) -> List[Tuple[Link,
                                    Optional[List[int]],
                                    Optional[List[int]],
                                    Optional[List[int]],
                                    Optional[List[int]],
                                    Optional[List[int]],
                                    Optional[List[int]],
                                    Optional[List[int]],
                                    Optional[str]]]:
        """
        Converts self to a list of (link, unknotting number,
        slice genus, strong slice genus, ribbon genus,
        strong ribbon genus, splitting number, weak
        splitting number, name) 9-tuples.

        Returns:
            A list of (link, unknotting number, slice genus,
            strong slice genus, ribbon genus, strong ribbon
            genus, splitting number, weak splitting number,
            name) 9-tuples representing the links in self.
        """
        return list(zip(self.links,
                        self.unknotting_nums,
                        self.slice_genera,
                        self.strong_slice_genera,
                        self.ribbon_genera,
                        self.strong_ribbon_genera,
                        self.splitting_nums,
                        self.weak_splitting_nums,
                        self.names))

    @staticmethod
    def from_list(
            lst: List[Tuple[Link,
                            Optional[List[int]],
                            Optional[List[int]],
                            Optional[List[int]],
                            Optional[List[int]],
                            Optional[List[int]],
                            Optional[List[int]],
                            Optional[List[int]],
                            Optional[str]]]
    ) -> 'Dataset':
        """
        Constructs a Dataset object from a list of (link,
        unknotting number, slice genus, strong slice genus,
        ribbon genus, strong ribbon genus, splitting number,
        weak splitting number, name) 9-tuples, as produced
        by to_list. Tuples of any other length raise.

        Args:
            lst: the given list of (link, unknotting number,
                slice genus, strong slice genus, ribbon
                genus, strong ribbon genus, splitting
                number, weak splitting number, name)
                9-tuples.

        Returns:
            The constructed dataset.
        """
        if not lst:
            return Dataset([], [], [], [], [], [], [], [], [])
        return Dataset(*[deepcopy(list(c))
                         for c in zip(*lst)])

    def __getitem__(self,
                    key: int | slice
                    ) -> Tuple[Link,
                               Optional[List[int]],
                               Optional[List[int]],
                               Optional[List[int]],
                               Optional[List[int]],
                               Optional[List[int]],
                               Optional[List[int]],
                               Optional[List[int]],
                               Optional[str]] | 'Dataset':
        """
        Returns (link, unknotting number, slice genus,
        strong slice genus, ribbon genus, strong ribbon
        genus, splitting number, weak splitting number,
        name) 9-tuple at given index in dataset or
        constructs a new dataset from a slice object.

        Args:
            key: given index or slice

        Returns:
            (link, unknotting number, slice genus, strong
            slice genus, ribbon genus, strong ribbon genus,
            splitting number, weak splitting number, name)
            9-tuple at index or new dataset containing data
            specified by the slice object
        """
        if isinstance(key, int):
            return (self.links[key],
                    self.unknotting_nums[key],
                    self.slice_genera[key],
                    self.strong_slice_genera[key],
                    self.ribbon_genera[key],
                    self.strong_ribbon_genera[key],
                    self.splitting_nums[key],
                    self.weak_splitting_nums[key],
                    self.names[key])
        return Dataset(self.links[key],
                       self.unknotting_nums[key],
                       self.slice_genera[key],
                       self.strong_slice_genera[key],
                       self.ribbon_genera[key],
                       self.strong_ribbon_genera[key],
                       self.splitting_nums[key],
                       self.weak_splitting_nums[key],
                       self.names[key])

    def __len__(self) -> int:
        """
        Returns the number of links in self.
        """
        return len(self.links)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.links):
            tup = self.__getitem__(self.index)
            self.index += 1
            return tup
        else:
            raise StopIteration

    def __add__(self, other: 'Dataset') -> 'Dataset':
        return Dataset.from_list(self.to_list() +
                                 other.to_list())

    def __iadd__(self, other: 'Dataset') -> 'Dataset':
        self.links += other.links
        self.unknotting_nums += other.unknotting_nums
        self.slice_genera += other.slice_genera
        self.strong_slice_genera += other.strong_slice_genera
        self.ribbon_genera += other.ribbon_genera
        self.strong_ribbon_genera += other.strong_ribbon_genera
        self.splitting_nums += other.splitting_nums
        self.weak_splitting_nums += other.weak_splitting_nums
        self.names += other.names
        return self

    def __repr__(self) -> str:
        return (f'Dataset with {len(self.links)} links:\n'
                f'{self.to_list()}')

    def __str__(self) -> str:
        return f'Dataset with {len(self.links)} links'

    def links_with_exact_invariant_value(
            self,
            framework: str,
            value: int
    ) -> 'Dataset':
        index = {'unknotting': 1, 'slice': 2, 'strong slice': 3,
                 'ribbon': 4, 'strong ribbon': 5,
                 'splitting': 6, 'weak splitting': 7}[framework]
        return Dataset.from_list(list(filter(lambda t: t[index] is not None and len(t[index]) == 1 and t[index][0] == value, self.to_list())))
    
    def links_with_known_invariant_value(
            self,
            framework: str
    ) -> 'Dataset':
        """
        Constructs a new Dataset object from self's links
        whose specified link invariant has a single possible
        value.

        Args:
            framework: the type of link invariant;
                "unknotting" for the unknotting number,
                "slice" for the slice genus, "strong slice"
                for the strong slice genus, "ribbon" for the
                ribbon genus, "strong ribbon" for the strong
                ribbon genus, "splitting" for the splitting
                number, and "weak splitting" for the weak
                splitting number

        Returns:
            A new Dataset object consisting of self's links
            whose link invariant specified by framework has
            a single possible value.
        """
        index = {'unknotting': 1, 'slice': 2, 'strong slice': 3,
                 'ribbon': 4, 'strong ribbon': 5,
                 'splitting': 6, 'weak splitting': 7}[framework]
        return Dataset.from_list(list(filter(lambda t: t[index] is not None and len(t[index]) == 1, self.to_list())))

    def links_with_unknown_invariant_value(
            self,
            framework: str
    ) -> 'Dataset':
        """
        Constructs a new Dataset object from self's links
        whose specified link invariant is either unknown or
        has multiple possible values.

        Args:
            framework: the type of link invariant;
                "unknotting" for the unknotting number,
                "slice" for the slice genus, "strong slice"
                for the strong slice genus, "ribbon" for the
                ribbon genus, "strong ribbon" for the strong
                ribbon genus, "splitting" for the splitting
                number, and "weak splitting" for the weak
                splitting number

        Returns:
            A new Dataset object consisting of self's links
            whose link invariant specified by framework is
            either None or has multiple possible values.
        """
        index = {'unknotting': 1, 'slice': 2, 'strong slice': 3,
                 'ribbon': 4, 'strong ribbon': 5,
                 'splitting': 6, 'weak splitting': 7}[framework]
        return Dataset.from_list(list(filter(lambda t: t[index] is None or len(t[index]) > 1, self.to_list())))

    @staticmethod
    def has_finite_strong_ribbon_genus(link: Link) -> bool:
        """
        Tells whether the strong ribbon genus of a link is
        finite, i.e. iff the link is a knot (a single component) 
        or all off-diagonal entries of the linking
        matrix vanish.
        Args:
            link: the link to test.

        Returns:
            True if the link's strong ribbon genus is finite.
        """
        if len(link.link_components) <= 1:
            return True
        lm = link.linking_matrix()
        n = len(lm)
        return all(lm[i][j] == 0
                   for i in range(n) for j in range(i+1, n))

    @staticmethod
    def has_finite_strong_slice_genus(link: Link) -> bool:
        """
        Tells whether the strong slice genus of a link is
        finite. The criterion is the same as for the strong 
        ribbon genus, so this delegates to 
        has_finite_strong_ribbon_genus.

        Args:
            link: the link to test.

        Returns:
            True if the link's strong slice genus is finite.
        """
        return Dataset.has_finite_strong_ribbon_genus(link)

    def links_with_finite_strong_ribbon_genus(self) -> 'Dataset':
        """
        Constructs a new Dataset object containing only self's
        links whose strong ribbon genus is finite.
        Returns:
            A new Dataset object with self's
            strong-ribbon-computable links (the "link0" subset).
        """
        return Dataset.from_list(
            [t for t in self.to_list()
             if Dataset.has_finite_strong_ribbon_genus(t[0])])

    def links_with_finite_strong_slice_genus(self) -> 'Dataset':
        """
        Constructs a new Dataset object containing only self's
        links whose strong slice genus is finite. The criterion
        is the same as for the strong ribbon genus, so this
        delegates to links_with_finite_strong_ribbon_genus.

        Returns:
            A new Dataset object with self's
            strong-slice-computable links (the "link0" subset).
        """
        return self.links_with_finite_strong_ribbon_genus()

    def split_by_number_of_crossings(self) -> List['Dataset']:
        """
        Splits self into a list of datasets, whose element
        at index i consists of self's links whose number of
        crossings equals i.

        Returns:
            A list of datasets, whose element at index i
            consists of self's links whose number of
            crossings equals i.
        """
        max_crossings = max([len(lnk.crossings)
                             for lnk in self.links] + [-1])
        lst = [[] for _ in range(max_crossings+1)]
        for t in self.to_list():
            lst[len(t[0].crossings)].append(t)
        return [Dataset.from_list(c) for c in lst]

    def sample(self, sample_size: int,
               seed: Optional[int] = None) -> 'Dataset':
        """
        Takes a sample of self's links with specified size.

        Args:
            sample_size: the specified sample size
            seed: the seed of the sample; if None, the
                sample is unseeded

        Returns:
            A new dataset consisting of a random sample of
            self's links with size equal to sample_size.

        Raises:
            ValueError: if sample_size is not between 0 and
                len(self.links)
        """
        if not (0 <= sample_size <= len(self.links)):
            raise ValueError(f'{sample_size} is not '
                             f'between 0 and '
                             f'{len(self.links)}')
        indices = sorted(random.Random(seed).sample(
            range(len(self)), sample_size))
        lst = self.to_list()
        return Dataset.from_list([lst[index]
                                  for index in indices])

    def simplify(self, mode: str = 'global') -> int:
        """
        Attempts to simplify the links in self.links.

        Args:
            mode: the simplification strategy to use in
                snappy.simplify

        Returns:
            The number of links that have been successfully
            simplified.
        """
        num_simplified = 0
        for link in self.links:
            if link.simplify(mode):
                num_simplified += 1
        return num_simplified

    def evaluate_invariant(
            self,
            framework: str,
            output: List[Tuple[int, List[str], List[Link], List[str]]] |
                    List[Tuple[int, List[Tuple[str, float]], List[Link], List[str]]],
            verbose: bool = True,
            evaluation_file: str = '',
            random_states_list_file: str = '../outputs/random-states.json',
            detailed_info_for_unknown_values: bool = False
    ) -> float | None:
        inv_n, inc_cross_signs, vals = (
            {'unknotting': ('unknotting number',
                           True,
                           self.unknotting_nums),
             'slice': ('slice genus',
                       True,
                       self.slice_genera),
             'strong slice': ('strong slice genus',
                               True,
                               self.strong_slice_genera),
             'ribbon': ('ribbon genus',
                        True,
                        self.ribbon_genera),
             'strong ribbon': ('strong ribbon genus',
                               True,
                               self.strong_ribbon_genera),
             'splitting': ('splitting number',
                           False,
                           self.splitting_nums),
             'weak splitting': ('weak splitting number',
                                False,
                                self.weak_splitting_nums)}
            [framework])
        return evaluate_invariant(
            inv_n, inc_cross_signs, self.links, self.names,
            vals, output, verbose, evaluation_file,
            random_states_list_file,
            detailed_info_for_unknown_values
        )
