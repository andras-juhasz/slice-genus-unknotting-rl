"""
Wrapper around KnotJob, a Java program computing the Rasmussen
s-invariant from Khovanov homology.  Needs Java 23 or later and
KnotJob.jar in a KnotJob directory beside this package, which
unzipping KnotJob.zip from
https://www.maths.dur.ac.uk/users/dirk.schuetz/knotjob.html
into src/ creates; the java binary is KNOTJOB_JAVA from the
environment, else java from the PATH.  Links are written in KnotJob's plain PD text format,
one "Name = PD[a,b,c,d],..." line each, and the s-invariants are
parsed from its output file.

Usage:
    from bounds_pipeline.knotjob_wrapper import (
        compute_s_invariant, compute_s_invariants_batch)
    from spherogram import Link

    compute_s_invariant(Link('3_1'))
    compute_s_invariants_batch([Link('3_1'), Link('4_1')],
                               names=['3_1', '4_1'])
"""

import subprocess
import tempfile
import threading
import os
import sys
from pathlib import Path
from time import time
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# The KnotJob jar, which needs Java 23 or later.
KNOTJOB_DIR = Path(__file__).parent.parent / "KnotJob"
KNOTJOB_JAR = KNOTJOB_DIR / "KnotJob.jar"

# Where KnotJob is distributed, as a zip holding the KnotJob directory.
KNOTJOB_URL = "https://www.maths.dur.ac.uk/users/dirk.schuetz/KnotJob.zip"

# The java binary, from the environment variable KNOTJOB_JAVA; "java" from
# the PATH by default, which must then be Java 23 or later.
JAVA23_PATH = os.environ.get("KNOTJOB_JAVA", "java")

# JVM heap cap.  KnotJob's README recommends 16g for large computations;
# the JVM default causes GC thrash and out-of-memory errors on big links.
# A cap is not a reservation, but with parallel workers size it so that
# num_workers x heap fits the machine.  Override with the environment
# variable KNOTJOB_JAVA_XMX (e.g. KNOTJOB_JAVA_XMX=16g).
KNOTJOB_JAVA_XMX = os.environ.get("KNOTJOB_JAVA_XMX", "8g")


def _girth_profile(pd_code, order) -> Tuple[int, int]:
    """
    (max_girth, total_girth) of consuming the rows of pd_code in
    order: an edge is open while exactly one of its two occurrences
    has been consumed, and the girth after each crossing is the
    number of open edges.  KnotJob's cost is exponential in the
    girth of the ordering it is given.
    """
    seen: Dict[int, int] = {}
    open_edges = 0
    mx = tot = 0
    for idx in order:
        for e in pd_code[idx]:
            seen[e] = seen.get(e, 0) + 1
            if seen[e] == 1:
                open_edges += 1
            else:
                open_edges -= 1
        if open_edges > mx:
            mx = open_edges
        tot += open_edges
    return mx, tot


def _girth_minimized_pd(pd_code: List[List[int]],
                        max_restarts: int = 64) -> List[List[int]]:
    """
    Reorders the rows of a PD code, a permutation that changes no
    link, so that KnotJob sees a small-girth crossing ordering; its
    command line consumes the row order as given.  Greedy: from
    each of up to max_restarts start crossings, append the
    crossing that keeps the open boundary smallest, preferring
    crossings touching it.  Returns the better of the original
    and the greedy order by (max_girth, total_girth).
    """
    n = len(pd_code)
    if n <= 2:
        return pd_code
    edges_of = [list(row) for row in pd_code]

    def run_from(start):
        used = [False] * n
        open_edges = set()
        order = []
        mx = tot = 0

        def place(i):
            nonlocal mx, tot
            used[i] = True
            order.append(i)
            for e in edges_of[i]:
                if e in open_edges:
                    open_edges.remove(e)
                else:
                    open_edges.add(e)
            if len(open_edges) > mx:
                mx = len(open_edges)
            tot += len(open_edges)

        place(start)
        while len(order) < n:
            best = None
            best_key = None
            for i in range(n):
                if used[i]:
                    continue
                closes = sum(1 for e in edges_of[i] if e in open_edges)
                new_open = len(open_edges) + 4 - 2 * closes
                key = (0 if closes > 0 else 1, new_open)
                if best_key is None or key < best_key:
                    best_key = key
                    best = i
            place(best)
        return order, mx, tot

    best_order, best_mx, best_tot = None, None, None
    for s in range(min(n, max_restarts)):
        order, mx, tot = run_from(s)
        if best_mx is None or (mx, tot) < (best_mx, best_tot):
            best_order, best_mx, best_tot = order, mx, tot

    if (best_mx, best_tot) < _girth_profile(pd_code, range(n)):
        return [pd_code[i] for i in best_order]
    return pd_code


def _link_to_pd_string(link, name: str = "Link",
                       girth_reorder: bool = True) -> str:
    """
    Convert a spherogram Link to KnotJob PD format string.

    Args:
        link: spherogram Link object
        name: Name to give the link in the output
        girth_reorder: reorder the PD rows for small girth before export
            (see _girth_minimized_pd; large speedups on big diagrams)

    Returns:
        String in format: "Name = PD[a,b,c,d],[e,f,g,h],..."
    """
    return _pd_code_to_string(link.PD_code(), name, girth_reorder)


def _pd_code_to_string(pd_code: List[List[int]], name: str = "Link",
                       girth_reorder: bool = True) -> str:
    """
    Convert a raw PD code to KnotJob PD format string.

    Args:
        pd_code: List of 4-element lists representing crossings
        name: Name to give the link
        girth_reorder: reorder the PD rows for small girth before export
            (see _girth_minimized_pd; large speedups on big diagrams)

    Returns:
        String in format: "Name = PD[a,b,c,d],[e,f,g,h],..."
    """
    if girth_reorder:
        pd_code = _girth_minimized_pd(pd_code)
    crossings_str = ",".join(
        f"[{','.join(str(x) for x in crossing)}]"
        for crossing in pd_code
    )
    return f"{name} = PD{crossings_str}"


def compute_s_invariant(
    link,
    characteristic: int = 0,
    name: Optional[str] = None,
    timeout: int = 15
) -> Optional[int]:
    """
    Computes the Rasmussen s-invariant of one link with KnotJob.

    Args:
        link: a spherogram Link.
        characteristic: the field characteristic, 0 for the
            rationals.
        name: a name for the link.
        timeout: seconds to wait for KnotJob.

    Returns:
        The s-invariant, or None when the computation failed or
        timed out.

    Raises:
        FileNotFoundError: without KnotJob.jar.
    """
    name = name or "Link"
    return compute_s_invariants_batch(
        links=[link], names=[name], characteristic=characteristic,
        timeout=timeout, per_link_timeout=timeout).get(name)


def _parse_s(content: str, characteristic: int = 0) -> Optional[int]:
    """
    Parse the s-invariant out of one KnotJob output record.

    The pattern is anchored to the start of a line, so a record carrying
    invariants this module did not ask for cannot be mistaken for the
    s-invariant: KnotJob names each one at the head of its own line.
    """
    s_match = re.search(
        rf"^\s*S-Invariant mod {characteristic}\s*:\s*(-?\d+)",
        content, re.MULTILINE
    )
    if s_match:
        return int(s_match.group(1))
    # Fallback: try without "mod N"
    s_match = re.search(r"^\s*S-Invariant[^:]*:\s*(-?\d+)", content, re.MULTILINE)
    return int(s_match.group(1)) if s_match else None


# Prefix of the identifiers written into the KnotJob input file in batch mode.
# The caller's link names are not used there: they may repeat, one may be a
# substring of another, and they may carry characters with a meaning in a
# regular expression.  A generated identifier is unique, is matched as a whole
# line, and is mapped back to the caller's name only after parsing.
_RECORD_ID_PREFIX = "KJBREC"


def _record_ids(count: int) -> List[str]:
    """Identifiers for one batch, in the order the links are written."""
    return [f"{_RECORD_ID_PREFIX}{n:07d}" for n in range(count)]


def _relabel_pd_string(pd_string: str, record_id: str) -> str:
    """
    Replaces the name in a "Name = PD[...],[...]" line with record_id.

    Raises:
        ValueError: if the string is not in that form, since silently sending
            an unlabelled line to KnotJob would produce a record no parser
            could attribute.
    """
    marker = " = PD"
    at = pd_string.find(marker)
    if at < 0:
        raise ValueError(f"not a KnotJob PD line: {pd_string[:60]!r}")
    return record_id + pd_string[at:]


def _split_knotjob_records(content: str, record_ids: List[str]) -> Dict[str, str]:
    """
    Splits KnotJob output into one block of text per record identifier.

    KnotJob writes the name of each link on a line of its own,
    followed by its invariant lines.  A record whose computation
    produced no output has an empty block, so its invariants stay
    None instead of taking the next record's.

    Args:
        content: the whole output file.
        record_ids: the identifiers written into the input file.

    Returns:
        {record id: the lines belonging to it}, with an entry for every id,
        possibly empty.  Text before the first identifier is discarded.
    """
    wanted = set(record_ids)
    blocks: Dict[str, List[str]] = {rid: [] for rid in record_ids}
    current = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped in wanted:
            current = stripped
            continue
        if current is not None:
            blocks[current].append(line)
    return {rid: "\n".join(lines) for rid, lines in blocks.items()}


def _run_knotjob_batch_single(
    pd_strings: List[str],
    names: List[str],
    characteristic: int,
    timeout: int,
    verbose: bool,
    worker_id: Optional[int] = None
) -> Tuple[Dict[str, Optional[int]], bool]:
    """
    Runs one KnotJob call on a batch of links.

    Args:
        pd_strings: KnotJob PD lines, one per link.
        names: the names of the links, unique within the call.
        characteristic: the field characteristic (the -s flag).
        timeout: seconds to wait for the Java process.
        verbose: print progress.
        worker_id: shown in the progress lines of a parallel worker.

    Returns:
        (results, retry): {name: s or None}, and whether the call
        is worth bisecting and retrying, i.e. the JVM was killed
        at the timeout, died on its own (an OutOfMemoryError at
        the heap cap, typically) or exited without writing its
        output file.  KnotJob writes that file only when the whole
        call finishes, so in each case every link is lost.

    Raises:
        ValueError: if a name repeats.
    """
    results: Dict[str, Optional[int]] = {name: None for name in names}
    if not names:
        # An empty input file makes the KnotJob JVM sit idle forever; never
        # launch it.
        return results, False
    prefix = f"[Worker {worker_id}] " if worker_id is not None else ""

    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        # results is keyed by name.
        raise ValueError(
            f"KnotJob batch: duplicate link names {sorted(duplicates)[:5]}; "
            f"names must identify a link uniquely within one call.")

    record_ids = _record_ids(len(names))
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.txt',
        delete=False
    ) as f:
        for pd_string, record_id in zip(pd_strings, record_ids):
            f.write(_relabel_pd_string(pd_string, record_id) + "\n")
        input_file = f.name

    output_file = f"{input_file}_s{characteristic}"

    try:
        cmd = [
            JAVA23_PATH, f"-Xmx{KNOTJOB_JAVA_XMX}", "-jar", str(KNOTJOB_JAR),
            input_file,
            f"-s{characteristic}",
        ]
        cmd.append("-ns")

        t0 = time()

        # Use Popen to stream stdout and show progress
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(KNOTJOB_DIR)
        )

        # The stdout loop below blocks until KnotJob exits, so a watchdog
        # timer kills the JVM at the deadline and flags the timeout.
        watchdog_fired = threading.Event()

        def _kill_on_timeout():
            watchdog_fired.set()
            proc.kill()

        watchdog = threading.Timer(timeout, _kill_on_timeout)
        watchdog.daemon = True
        watchdog.start()

        # KnotJob prints one "finished" line per link.
        total_links = len(names)
        total_expected = total_links
        finished_count = 0
        next_milestone_pct = 10
        stdout_lines = []

        try:
            for line in proc.stdout:
                stdout_lines.append(line)
                if 'finished' in line.lower():
                    finished_count += 1
                    if verbose and total_expected > 0:
                        pct = 100 * finished_count / total_expected
                        if pct >= next_milestone_pct:
                            elapsed = time() - t0
                            print(f"    {prefix}KnotJob progress: {int(pct)}% "
                                  f"({finished_count}/{total_expected} invariants) "
                                  f"in {elapsed:.1f}s",
                                  flush=True)
                            next_milestone_pct = (int(pct) // 10 + 1) * 10

            proc.wait()
        finally:
            watchdog.cancel()

        if watchdog_fired.is_set():
            if verbose:
                print(f"    {prefix}KnotJob timed out after {timeout}s "
                      f"({total_links} links in call)", flush=True)
            return results, True

        kj_elapsed = time() - t0
        if verbose:
            print(f"    {prefix}KnotJob completed in {kj_elapsed:.1f}s "
                  f"({total_links} links)")

        if proc.returncode != 0:
            if verbose:
                print(f"    {prefix}KnotJob exited abnormally "
                      f"(rc={proc.returncode}, likely OutOfMemoryError at "
                      f"-Xmx{KNOTJOB_JAVA_XMX}) ({total_links} links in call)",
                      flush=True)
            return results, True

        if not os.path.exists(output_file):
            if verbose:
                print(f"    {prefix}KnotJob finished without writing its "
                      f"output file ({total_links} links in call)", flush=True)
            return results, True

        t_parse = time()
        with open(output_file, 'r') as f:
            content = f.read()

        # One block per record: the record id on its own line, then
        # "S-Invariant mod 0 : <value>".
        blocks = _split_knotjob_records(content, record_ids)
        for name, record_id in zip(names, record_ids):
            results[name] = _parse_s(blocks.get(record_id, ""), characteristic)

        if verbose:
            parse_elapsed = time() - t_parse
            print(f"    {prefix}Parsed results in {parse_elapsed:.1f}s")

        return results, False

    except Exception:
        return results, False
    finally:
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(output_file):
            os.unlink(output_file)


def _run_knotjob_batch_with_retry(
    pd_strings: List[str],
    names: List[str],
    characteristic: int,
    timeout: int,
    verbose: bool,
    worker_id: Optional[int] = None,
    per_link_timeout: int = 15
) -> Dict[str, Optional[int]]:
    """
    Runs a KnotJob batch and, on a failed call, bisects it and
    retries the halves, so that one over-budget link loses only
    its own result; a link failing on its own budget keeps None.

    Args:
        pd_strings: KnotJob PD lines, one per link.
        names: the names of the links.
        characteristic: the field characteristic.
        timeout: budget in seconds for this call.
        verbose: print progress.
        worker_id: shown in the progress lines of a parallel worker.
        per_link_timeout: sizes the budget of each retry half,
            max(60, per_link_timeout * len(half)).

    Returns:
        {name: s or None}.
    """
    if not names:
        return {}

    results, timed_out = _run_knotjob_batch_single(
        pd_strings, names, characteristic, timeout, verbose, worker_id)
    if not timed_out:
        return results

    prefix = f"[Worker {worker_id}] " if worker_id is not None else ""
    if len(names) == 1:
        if verbose:
            print(f"    {prefix}link {names[0]} exceeded its own budget "
                  f"({timeout}s) or crashed; s-invariant skipped", flush=True)
        return results

    mid = len(names) // 2
    if verbose:
        print(f"    {prefix}retrying timed-out batch as two halves "
              f"({mid} + {len(names) - mid} links)", flush=True)
    half_budget_l = max(60, per_link_timeout * mid)
    half_budget_r = max(60, per_link_timeout * (len(names) - mid))
    merged = _run_knotjob_batch_with_retry(
        pd_strings[:mid], names[:mid], characteristic, half_budget_l,
        verbose, worker_id, per_link_timeout)
    merged.update(_run_knotjob_batch_with_retry(
        pd_strings[mid:], names[mid:], characteristic, half_budget_r,
        verbose, worker_id, per_link_timeout))
    return merged


def compute_s_invariants_batch(
    links: list = None,
    pd_codes: List[List[List[int]]] = None,
    names: Optional[List[str]] = None,
    characteristic: int = 0,
    timeout: int = 600,
    verbose: bool = False,
    num_workers: int = 1,
    per_link_timeout: int = 15
) -> Dict[str, Optional[int]]:
    """
    Computes the Rasmussen s-invariant of several links in one
    KnotJob call, or in num_workers parallel Java processes, the
    links spread over them by crossing count.  A failed call is
    bisected and retried (_run_knotjob_batch_with_retry).

    Args:
        links: spherogram Links, or
        pd_codes: their PD codes.
        names: one name per link; generated when omitted.
        characteristic: the field characteristic, 0 for the
            rationals.
        timeout: seconds for each Java call.
        verbose: print progress.
        num_workers: parallel Java processes; each JVM takes its
            own heap, KNOTJOB_JAVA_XMX.
        per_link_timeout: seconds per link, sizing the budgets of
            the retry halves.

    Returns:
        {name: s or None}.

    Raises:
        FileNotFoundError: without KnotJob.jar.
        ValueError: with neither links nor pd_codes.
    """
    if not KNOTJOB_JAR.exists():
        raise FileNotFoundError(
            f"KnotJob JAR not found at {KNOTJOB_JAR}. Download "
            f"{KNOTJOB_URL} and unzip it into {KNOTJOB_DIR.parent}; "
            f"readme.md has the steps."
        )

    if links is not None:
        if names is None:
            names = [f"Link_{i}" for i in range(len(links))]
        pd_strings = [_link_to_pd_string(link, name) for link, name in zip(links, names)]
    elif pd_codes is not None:
        if names is None:
            names = [f"Link_{i}" for i in range(len(pd_codes))]
        pd_strings = [_pd_code_to_string(pd, name) for pd, name in zip(pd_codes, names)]
    else:
        raise ValueError("Either 'links' or 'pd_codes' must be provided")

    results: Dict[str, Optional[int]] = {name: None for name in names}
    if not names:
        return results

    # Single-worker path (with bisect-retry on timeout).
    if num_workers <= 1:
        return _run_knotjob_batch_with_retry(
            pd_strings, names, characteristic, timeout, verbose,
            per_link_timeout=per_link_timeout
        )

    # Multi-worker path. Use greedy LPT scheduling by crossing
    # count, carrying original indices through to result mapping.
    n = len(pd_strings)
    actual_workers = min(num_workers, n)  # cap at number of links

    def _crossing_count(idx):
        # Prefer the spherogram Link's crossing list; fall back to the PD length
        # (one PD tuple per crossing); 0 if neither is available.
        if links is not None:
            try:
                return len(links[idx].crossings)
            except Exception:
                pass
        if pd_codes is not None:
            try:
                return len(pd_codes[idx])
            except Exception:
                pass
        return 0

    order = sorted(range(n), key=_crossing_count, reverse=True)
    # worker_bins[w] = list of original indices assigned to worker w
    worker_bins = [[] for _ in range(actual_workers)]
    worker_load = [0] * actual_workers  # running sum of crossings per worker
    for idx in order:
        w = min(range(actual_workers), key=lambda k: worker_load[k])
        worker_bins[w].append(idx)
        worker_load[w] += _crossing_count(idx)

    chunks_pd = [[pd_strings[i] for i in bin_] for bin_ in worker_bins]
    chunks_names = [[names[i] for i in bin_] for bin_ in worker_bins]

    if verbose:
        loads = ", ".join(str(l) for l in worker_load)
        print(f"    KnotJob: {actual_workers} parallel workers, "
              f"{n} links total "
              f"(~{n // actual_workers} links/worker; "
              f"crossings/worker balanced via LPT: [{loads}])",
              flush=True)

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_id = {
            executor.submit(
                _run_knotjob_batch_with_retry,
                chunk_pd, chunk_names,
                characteristic, timeout,
                verbose, worker_id, per_link_timeout
            ): worker_id
            for worker_id, (chunk_pd, chunk_names) in enumerate(
                zip(chunks_pd, chunks_names))
        }
        for future in as_completed(future_to_id):
            results.update(future.result())

    return results


if __name__ == "__main__":
    # Installation check: the tests never start a JVM, so this is what
    # says whether the jar is reachable.  The assertions are chirality
    # independent.
    import sys

    from spherogram import Link

    single = compute_s_invariant(Link('3_1'), name='3_1')
    batch = compute_s_invariants_batch(
        links=[Link('3_1'), Link('4_1'), Link('L2a1')],
        names=['3_1', '4_1', 'L2a1'],
    )
    print(f"single call: s(3_1) = {single}")
    for name, s in batch.items():
        print(f"batch: s({name}) = {s}")

    failures = []
    if single is None or abs(single) != 2:
        failures.append(f"|s(3_1)| should be 2, got {single}")
    if batch.get('3_1') is None or abs(batch['3_1']) != 2:
        failures.append(f"batch |s(3_1)| should be 2, got {batch.get('3_1')}")
    if batch.get('4_1') != 0:
        failures.append(f"s(4_1) should be 0, got {batch.get('4_1')}")
    if batch.get('L2a1') is None:
        failures.append("s(L2a1) came back empty")

    if failures:
        for line in failures:
            print(f"FAILED: {line}")
        sys.exit(1)
    print("OK")
