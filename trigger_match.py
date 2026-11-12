"""
Merge trigger_acs.py outputs: cluster triggers within a time window and keep
the most significant (highest sigma) per cluster. Output matches triggers.npy
columns: time, algorithm index, sigma.

Cross-match merged triggers to events.csv by start time (within a time window).

Merged triggers with no simulated event start within that window are **spurious**
(no corresponding input event); they are listed in ``trigger-spurious.csv``.
That list is **identical** for every flux band and replicate (``a``, ``b``, ...) variant; it is
computed once from the **lowest-flux ``a``** catalog folder (see
``trigger_outputs.canonical_spurious_source_dataset_name``) and copied to all
other expected dataset folders after matching.

Expected ACS flux folders (those with ``events.csv`` under the ACS root,
default ``sgrbs-scaled-background-binned-acs``) define the suite. **Fast mode**
(default) only includes ``flux-1-10`` and ``flux-10-100`` replicas (``a``, ``b``, ...);
pass ``--complete`` to require every ACS flux band. Pass ``--acs-root`` to use
a different ACS tree (for example the unscaled ``sgrbs-background-binned-acs-1-100``).
Each dataset in scope must have ``triggers.npy`` in the
run before ``trigger_qa`` treats the run as complete for that scope.

Run this file from the IDE or with optional ``--complete`` / ``--acs-root``.
"""

import math
import os
import shutil

import numpy as np

import trigger_outputs

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACS_ROOT = trigger_outputs.default_acs_root(_SCRIPT_DIR)

TRIGGERS_IN = 'triggers.npy'
MERGED_OUT = 'trigger-merged.npy'
MERGE_WINDOW_S = 5.0

EVENTS_OUT = 'events-matched.csv'
SPURIOUS_CSV = 'trigger-spurious.csv'
CROSSMATCH_WINDOW_S = 5.0
START_TIME_COL = 'Start time (s)'


def merge_triggers(triggers: np.ndarray, merge_window_s: float = MERGE_WINDOW_S) -> np.ndarray:
    """
    Single-linkage clustering on sorted times: consecutive triggers with
    (t[i] - t[i-1]) <= merge_window_s belong to the same event. Each event
    keeps the row with maximum sigma (column 2); ties break on earlier time.

    Same schema as triggers.npy: shape (n, 3) with columns [time, algorithm, sigma].
    """
    triggers = np.asarray(triggers)
    if triggers.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if triggers.ndim == 1:
        if len(triggers) == 0:
            return np.empty((0, 3), dtype=np.float64)
        triggers = triggers.reshape(1, -1)
    if triggers.shape[1] != 3:
        raise ValueError(f'Expected triggers with shape (n, 3), got {triggers.shape}')

    order = np.argsort(triggers[:, 0])
    t = triggers[order]

    merged = []
    cluster_start = 0
    for i in range(1, len(t)):
        if t[i, 0] - t[i - 1, 0] > merge_window_s:
            block = t[cluster_start:i]
            j = int(np.argmax(block[:, 2]))
            merged.append(block[j])
            cluster_start = i
    block = t[cluster_start:]
    j = int(np.argmax(block[:, 2]))
    merged.append(block[j])

    out = np.array(merged, dtype=triggers.dtype)
    return out


def find_best_trigger_match(
    event_start_s: float,
    merged: np.ndarray,
    window_s: float,
) -> tuple[float, float, float]:
    """
    Return (trigger_time, algorithm, sigma) for the merged trigger closest in
    time to event_start_s among those with |t - event_start_s| <= window_s.
    If none, return (0.0, 0.0, 0.0).
    """
    if merged.size == 0 or math.isnan(event_start_s):
        return (0.0, 0.0, 0.0)
    best_row = None
    best_abs = float('inf')
    for row in merged:
        t_tr, alg, sig = float(row[0]), float(row[1]), float(row[2])
        d = abs(t_tr - event_start_s)
        if d <= window_s and d < best_abs:
            best_abs = d
            best_row = (t_tr, alg, sig)
    if best_row is None:
        return (0.0, 0.0, 0.0)
    t_tr, alg, sig = best_row
    if alg == int(alg):
        alg = int(alg)
    return (t_tr, alg, sig)


def load_finite_event_start_times(events_path: str, start_col: str) -> np.ndarray:
    """All finite event start times (s) from events.csv (one row per event line)."""
    rows = trigger_outputs.read_table_rows(events_path)
    if not rows:
        return np.array([], dtype=np.float64)
    header = rows[0]
    i_start = header.index(start_col)
    vals = []
    for row in rows[1:]:
        if not row or all(c.strip() == '' for c in row):
            continue
        if i_start >= len(row):
            continue
        try:
            vals.append(float(row[i_start].strip()))
        except ValueError:
            continue
    return np.asarray(vals, dtype=np.float64)


def catalog_time_offset(
    acs_root: str,
    dataset_name: str,
    event_starts_s: np.ndarray,
) -> tuple[float, float]:
    """
    Offset (s) to add to trigger times (relative to the first ACS bin) to put
    them on the ``events.csv`` clock, and the fraction of events it covers.

    Catalogs are either relative to the first ACS bin (offset 0) or on the
    ACS file's absolute timebase (offset ``time_reference_s``). The candidate
    placing the most event starts inside the ACS time span wins. Raises if no
    candidate covers any event, so a clock mismatch cannot silently produce
    zero detections.
    """
    ev = np.asarray(event_starts_s, dtype=np.float64)
    ev = ev[np.isfinite(ev)]
    cache = trigger_outputs.find_binned_cache_dir(acs_root, dataset_name)
    if cache is None or ev.size == 0:
        return 0.0, float('nan')
    times = np.load(os.path.join(cache, 'bin_times.npy'))
    t_start = times[:, 0] if times.ndim == 2 else times
    t_end = times[:, 1] if times.ndim == 2 else times
    lo, hi = float(np.min(t_start)), float(np.max(t_end))
    ref_path = os.path.join(cache, 'time_reference_s.npy')
    ref = float(np.load(ref_path)) if os.path.isfile(ref_path) else 0.0

    best_off, best_frac = 0.0, -1.0
    for off in dict.fromkeys((0.0, ref)):
        frac = float(np.mean((ev >= lo + off) & (ev <= hi + off)))
        if frac > best_frac:
            best_off, best_frac = off, frac
    if best_frac <= 0.0:
        raise ValueError(
            f'[{dataset_name}] No event start time ({ev.min():.6g}–{ev.max():.6g} s) lies '
            f'inside the ACS data span, either relative ({lo:.6g}–{hi:.6g} s) or shifted '
            f'by time_reference_s = {ref:.10g} s. events.csv and combined_acs_data.hdf5 '
            'are on different clocks; matching would report zero detections.'
        )
    return best_off, best_frac


def spurious_merged_triggers(
    merged: np.ndarray,
    event_starts_s: np.ndarray,
    window_s: float,
) -> np.ndarray:
    """
    Merged triggers that have **no** event start with |t_trigger - t_event| <= window_s
    (same symmetric window as cross-matching). Same schema as merged: (n, 3).
    """
    merged = np.asarray(merged)
    if merged.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    event_starts_s = np.asarray(event_starts_s, dtype=np.float64)
    event_starts_s = event_starts_s[np.isfinite(event_starts_s)]
    t_tr = merged[:, 0].astype(np.float64, copy=False)
    if event_starts_s.size == 0:
        return merged.copy()
    dmin = np.min(np.abs(t_tr[:, np.newaxis] - event_starts_s[np.newaxis, :]), axis=1)
    mask = dmin > window_s
    out = merged[mask]
    out = np.array(out, dtype=merged.dtype)
    if len(out) > 1:
        out = out[np.argsort(out[:, 0])]
    return out


def write_spurious_csv(path: str, spurious: np.ndarray) -> None:
    header = ['Trigger time (s)', 'Algorithm', 'Significance']
    rows = [header] + [[str(row[0]), str(row[1]), str(row[2])] for row in spurious]
    trigger_outputs.write_csv_rows(path, rows)


def crossmatch_events_csv(
    events_path: str,
    merged: np.ndarray,
    out_path: str,
    window_s: float,
    start_col: str = START_TIME_COL,
) -> None:
    """Read the events catalog, append the match columns, write out_path as CSV."""
    rows = trigger_outputs.read_table_rows(events_path)

    if not rows:
        raise ValueError(f'Empty file: {events_path}')

    header = rows[0]
    try:
        i_start = header.index(start_col)
    except ValueError as e:
        raise ValueError(
            f'Column {start_col!r} not found in {events_path}; header={header!r}'
        ) from e

    # Every catalog event is written. Undetected events carry ``nan`` in the
    # three matched columns (algorithm 0 is a real test, so 0 cannot mean
    # "no match") and ``Detected`` = 0.
    out_header = header + [
        'Matched trigger time (s)',
        'Matched trigger algorithm',
        'Matched trigger significance',
        'Detected',
    ]
    out_rows = [out_header]

    for row in rows[1:]:
        if not row or all(c.strip() == '' for c in row):
            continue
        while len(row) < len(header):
            row.append('')
        try:
            t_event = float(row[i_start].strip())
        except ValueError:
            t_event = float('nan')

        mt, ma, ms = find_best_trigger_match(t_event, merged, window_s)
        if mt == 0.0 and ma == 0.0 and ms == 0.0:
            suffix = ['nan', 'nan', 'nan', '0']
        else:
            suffix = [str(mt), str(ma), str(ms), '1']

        out_rows.append(row[: len(header)] + suffix)

    trigger_outputs.write_csv_rows(out_path, out_rows)


# --- Run in IDE: execute this file (entire file or from here downward) ---


def main():
    global ACS_ROOT
    run_dir = trigger_outputs.read_current_run(_SCRIPT_DIR)
    print(f'Current run directory: {run_dir}\n')

    scope = trigger_outputs.parse_run_scope_argv(script_dir=_SCRIPT_DIR)
    complete = scope.complete
    ACS_ROOT = scope.acs_root
    print(f'ACS root: {ACS_ROOT}')
    expected = trigger_outputs.expected_acs_dataset_names(ACS_ROOT)
    suite = trigger_outputs.acs_datasets_for_mode(ACS_ROOT, complete=complete)
    if expected:
        if not suite:
            raise FileNotFoundError(
                f'Fast mode: no flux-1-10 / flux-10-100 folders with events.csv under {ACS_ROOT!r}. '
                'Re-run with ``--complete`` for all flux bands, or add those catalogs.'
            )
        print(
            'ACS scope: '
            + ('all flux datasets (--complete)' if complete else 'fast (flux-1-10, flux-10-100 only)')
            + '\n'
        )

    if expected:
        missing_trig = trigger_outputs.missing_flux_outputs_for_match(
            run_dir, ACS_ROOT, TRIGGERS_IN, complete=complete
        )
        if missing_trig:
            print(
                'Incomplete run for current flux scope (missing triggers.npy for '
                f'datasets): {", ".join(missing_trig)}'
            )
            print(
                'Processing only datasets that have triggers.npy; '
                'trigger_qa will skip population/global figures until every dataset '
                f'in scope has {EVENTS_OUT}.\n'
            )
        else:
            print(
                'All ACS flux datasets in scope have triggers.npy in this run '
                f'({len(suite)} folders).\n'
            )
        names_to_process = suite
    else:
        print(
            f'Warning: no ACS datasets with events.csv under {ACS_ROOT!r}; '
            'processing every subfolder of the run that has triggers.npy.\n'
        )
        names_to_process = sorted(
            n
            for n in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, n))
        )

    spur_canon: str | None = None
    if suite:
        spur_canon = trigger_outputs.canonical_spurious_source_dataset_name(
            ACS_ROOT, dataset_names=suite
        )
        if spur_canon:
            canon_dir = os.path.join(run_dir, spur_canon)
            if spur_canon not in names_to_process or not os.path.isfile(
                os.path.join(canon_dir, TRIGGERS_IN)
            ):
                print(
                    f'Warning: canonical spurious folder {spur_canon!r} is missing from '
                    f'this run or has no {TRIGGERS_IN}; computing {SPURIOUS_CSV} per dataset.'
                )
                spur_canon = None
            elif not trigger_outputs.find_events_csv(ACS_ROOT, spur_canon):
                print(
                    f'Warning: no events.csv for canonical {spur_canon!r}; '
                    f'computing {SPURIOUS_CSV} per dataset.'
                )
                spur_canon = None
            else:
                print(
                    'Spurious merged-trigger list is shared across flux datasets; '
                    f'computing ``{SPURIOUS_CSV}`` from {spur_canon!r} (lowest-flux ``a``) '
                    'and copying to other expected folders.\n'
                )

    n_done = 0
    for name in names_to_process:
        dataset_dir = os.path.join(run_dir, name)
        if not os.path.isdir(dataset_dir):
            print(f'Skip {name!r}: no dataset folder under run')
            continue

        path_in = os.path.join(dataset_dir, TRIGGERS_IN)
        path_out = os.path.join(dataset_dir, MERGED_OUT)
        path_events_matched = os.path.join(dataset_dir, EVENTS_OUT)
        path_spurious_csv = os.path.join(dataset_dir, SPURIOUS_CSV)

        if not os.path.isfile(path_in):
            print(f'Skip {name!r}: missing {path_in}')
            continue

        triggers_loaded = np.load(path_in)
        merged = merge_triggers(triggers_loaded, MERGE_WINDOW_S)
        np.save(path_out, merged)
        n_done += 1

        print(f'[{name}] Loaded {path_in} ({len(triggers_loaded)} raw rows)')
        print(f'[{name}] Merged window = {MERGE_WINDOW_S} s → {len(merged)} merged → {path_out}')

        events_path = trigger_outputs.find_events_csv(ACS_ROOT, name)
        if not events_path:
            print(f'[{name}] No events.csv under {ACS_ROOT}/{name}/; skipping cross-match.')
            continue

        event_starts = load_finite_event_start_times(events_path, START_TIME_COL)
        offset, covered = catalog_time_offset(ACS_ROOT, name, event_starts)
        if np.isfinite(covered):
            print(
                f'[{name}] events.csv clock = ACS relative time + {offset:.10g} s '
                f'({covered:.0%} of event starts inside the ACS data span)'
            )
            if covered < 1.0:
                print(f'[{name}] Warning: {1.0 - covered:.0%} of events fall outside the ACS data.')
        # trigger-merged.npy and trigger-spurious.csv stay on the ACS relative clock
        # (QA lightcurves index bin_times.npy); matched times use the catalog clock.
        merged_catalog = merged.copy()
        if merged_catalog.size:
            merged_catalog[:, 0] += offset
        if spur_canon is not None and name != spur_canon:
            print(f'[{name}] Spurious list will be copied from {spur_canon!r} (after pass).')
        else:
            spurious = spurious_merged_triggers(
                merged, event_starts - offset, CROSSMATCH_WINDOW_S
            )
            write_spurious_csv(path_spurious_csv, spurious)
            print(
                f'[{name}] Spurious merged triggers: {len(spurious)} / {len(merged)} '
                f'→ {path_spurious_csv}'
            )

        crossmatch_events_csv(
            events_path, merged_catalog, path_events_matched, CROSSMATCH_WINDOW_S
        )
        print(
            f'[{name}] Cross-matched (|Δt| ≤ {CROSSMATCH_WINDOW_S} s) '
            f'{events_path} → {path_events_matched}'
        )

    if spur_canon and os.path.isfile(os.path.join(run_dir, spur_canon, SPURIOUS_CSV)):
        src_sp = os.path.join(run_dir, spur_canon, SPURIOUS_CSV)
        copy_names = suite if suite else (expected if expected else names_to_process)
        for name in copy_names:
            if name == spur_canon:
                continue
            dst_dir = os.path.join(run_dir, name)
            dst_sp = os.path.join(dst_dir, SPURIOUS_CSV)
            ev_other = trigger_outputs.find_events_csv(ACS_ROOT, name)
            if not ev_other or not os.path.isdir(dst_dir):
                continue
            if not os.path.isfile(os.path.join(dst_dir, EVENTS_OUT)):
                continue
            shutil.copy2(src_sp, dst_sp)
            print(f'[{name}] Copied {SPURIOUS_CSV} from {spur_canon!r} → {dst_sp}')

    if expected and n_done == 0:
        print('No expected flux datasets were merged/matched (check triggers.npy paths).')
    elif expected and trigger_outputs.is_run_complete_for_match(
        run_dir, ACS_ROOT, TRIGGERS_IN, complete=complete
    ):
        print('Run is complete for trigger_match (all in-scope datasets have triggers.npy).')

    if expected:
        qa_agg = trigger_outputs.is_run_complete_for_qa(
            run_dir, ACS_ROOT, EVENTS_OUT, complete=complete
        )
        print(
            f'Ready for trigger_qa population/global figures (all in-scope have {EVENTS_OUT}): '
            f'{"yes" if qa_agg else "no — finish trigger_match for missing datasets above"}'
        )


if __name__ == '__main__':
    main()
