"""
Shared layout for per-run outputs under outputs/.

trigger_acs.py creates a new run directory, copies ini snapshots, and writes
_current_run.txt. trigger_match.py / trigger_qa.py read that file to find the
latest run to process.
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence
from datetime import datetime
from typing import NamedTuple


def _script_dir_here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def outputs_root(script_dir: str | None = None) -> str:
    base = script_dir or _script_dir_here()
    return os.path.join(base, 'outputs')


def current_run_marker_path(script_dir: str | None = None) -> str:
    return os.path.join(outputs_root(script_dir), '_current_run.txt')


def new_run_directory(script_dir: str | None = None) -> str:
    """Create outputs/run_YYYYMMDD_HHMMSS and return its absolute path."""
    root = outputs_root(script_dir)
    os.makedirs(root, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(root, f'run_{stamp}')
    if os.path.exists(run_dir):
        run_dir = os.path.join(root, f'run_{stamp}_{os.getpid()}')
    os.makedirs(run_dir, exist_ok=True)
    return os.path.abspath(run_dir)


def write_current_run(script_dir: str, run_dir: str) -> None:
    os.makedirs(outputs_root(script_dir), exist_ok=True)
    with open(current_run_marker_path(script_dir), 'w', encoding='utf-8') as f:
        f.write(os.path.abspath(run_dir) + '\n')


def read_current_run(script_dir: str | None = None) -> str:
    path = current_run_marker_path(script_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'Missing {path}; run trigger_acs.py first to create a run and register it.'
        )
    with open(path, encoding='utf-8') as f:
        line = f.readline().strip()
    if not line or not os.path.isdir(line):
        raise FileNotFoundError(
            f'Current run path in {path!r} is missing or not a directory: {line!r}'
        )
    return line


_FLUX_OUTER_RE = re.compile(r'^flux-([\d.]+)-([\d.]+)([a-z])$')
# Default (fast) mode: only these flux band replicas (keV edges), not e.g. flux-10-1000a.
_FAST_FLUX_BAND_RE = re.compile(r'^flux-(?:1-10|10-100)[a-z]$')


def acs_outer_name_from_combined(combined_h5_path: str) -> str:
    """
    Dataset folder name for a combined HDF5 path.

    Flat (current): ``.../flux-10-100a/combined_acs_data.hdf5`` -> ``flux-10-100a``
    Legacy nested: ``.../flux-10-100a/flux-10-100/combined_acs_data.hdf5`` -> ``flux-10-100a``
    """
    parent = os.path.dirname(os.path.abspath(combined_h5_path))
    name = os.path.basename(parent)
    if _FLUX_OUTER_RE.match(name):
        return name
    return os.path.basename(os.path.dirname(parent))


def find_events_csv(acs_root: str, outer_name: str) -> str | None:
    """Return path to events.csv under acs_root/<outer>/events.csv or None."""
    import glob

    direct = os.path.join(acs_root, outer_name, 'events.csv')
    if os.path.isfile(direct):
        return direct
    # Legacy nested layout: acs_root/<outer>/*/events.csv
    pattern = os.path.join(acs_root, outer_name, '*', 'events.csv')
    found = sorted(glob.glob(pattern))
    if not found:
        return None
    return found[0]


def is_fast_flux_dataset_name(name: str) -> bool:
    """True for ``flux-1-10[a-z]`` and ``flux-10-100[a-z]`` only (not ``flux-10-1000*``)."""
    return bool(_FAST_FLUX_BAND_RE.match(name))


def acs_datasets_for_mode(acs_root: str, *, complete: bool) -> list[str]:
    """
    ACS dataset folder names (with ``events.csv``) for the current run scope.

    * ``complete=False`` (default / *fast* mode): only ``flux-1-10`` and ``flux-10-100``
      ``a``/``b`` replicas.
    * ``complete=True``: every dataset returned by ``expected_acs_dataset_names``.
    """
    all_n = expected_acs_dataset_names(acs_root)
    if complete or not all_n:
        return all_n
    return [n for n in all_n if is_fast_flux_dataset_name(n)]


DEFAULT_ACS_ROOT_NAME = 'sgrbs-scaled-background-binned-acs'


class RunScope(NamedTuple):
    """CLI run scope: flux-band completeness and ACS data root."""

    complete: bool
    acs_root: str


def default_acs_root(script_dir: str | None = None) -> str:
    return os.path.join(script_dir or _script_dir_here(), DEFAULT_ACS_ROOT_NAME)


def resolve_acs_root(path: str | None, script_dir: str | None = None) -> str:
    """Resolve ``--acs-root`` against the scripts directory (not cwd)."""
    base = script_dir or _script_dir_here()
    if not path:
        return os.path.abspath(default_acs_root(base))
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base, path))


def parse_run_scope_argv(
    argv: Sequence[str] | None = None,
    *,
    script_dir: str | None = None,
) -> RunScope:
    """
    Parse ``sys.argv`` (or ``argv``) for ``--complete`` and ``--acs-root``.

    ``complete`` is True to process the **full** ACS flux suite; False for
    **fast** mode (only ``flux-1-10`` and ``flux-10-100`` replicas).
    ``acs_root`` is the resolved ACS binned-data directory (default:
    ``sgrbs-scaled-background-binned-acs`` next to the scripts).

    Unknown arguments are ignored (``parse_known_args``) so IDE runners can
    coexist with optional flags.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        '--complete',
        action='store_true',
        help=(
            'Process all ACS flux datasets under the ACS root '
            '(default: fast mode — only flux-1-10 and flux-10-100 a/b folders).'
        ),
    )
    parser.add_argument(
        '--acs-root',
        default=None,
        help=(
            'ACS binned-data directory containing flux-* folders. Relative paths '
            f'are resolved next to the scripts (default: {DEFAULT_ACS_ROOT_NAME}).'
        ),
    )
    ns, _unknown = parser.parse_known_args(args=None if argv is None else list(argv))
    return RunScope(
        complete=ns.complete,
        acs_root=resolve_acs_root(ns.acs_root, script_dir),
    )


def canonical_spurious_source_dataset_name(
    acs_root: str,
    dataset_names: Sequence[str] | None = None,
) -> str | None:
    """
    Lowest-flux ``flux-*`` dataset whose name ends in ``a`` (same catalog layout
    as ``expected_acs_dataset_names``).

    If ``dataset_names`` is set (e.g. fast-mode subset), only those names are
    considered; otherwise all expected ACS datasets are used.

    Spurious merged triggers are identical across flux bands and ``a``/``b``;
    ``trigger_match`` writes ``trigger-spurious.csv`` once from this folder and
    copies it to the others. Tie-break: lower high-edge after lower low-edge.
    Returns None if no matching ``*a`` name exists.
    """
    names = (
        list(dataset_names)
        if dataset_names is not None
        else expected_acs_dataset_names(acs_root)
    )
    scored: list[tuple[tuple[float, float], str]] = []
    for n in names:
        m = _FLUX_OUTER_RE.match(n)
        if not m or m.group(3) != 'a':
            continue
        lo, hi = float(m.group(1)), float(m.group(2))
        scored.append(((lo, hi), n))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def expected_acs_dataset_names(acs_root: str) -> list[str]:
    """
    Sorted top-level folder names under ``acs_root`` that contain a catalog
    ``events.csv`` (same layout as ``sgrbs-scaled-background-binned-acs/flux-*``).
    """
    if not os.path.isdir(acs_root):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(acs_root)):
        p = os.path.join(acs_root, name)
        if not os.path.isdir(p):
            continue
        ev = find_events_csv(acs_root, name)
        if ev and os.path.isfile(ev):
            out.append(name)
    return out


def missing_flux_outputs_for_match(
    run_dir: str,
    acs_root: str,
    triggers_filename: str = 'triggers.npy',
    *,
    complete: bool = False,
) -> list[str]:
    """ACS dataset names in scope that do not yet have ``triggers_filename`` in ``run_dir``."""
    req = acs_datasets_for_mode(acs_root, complete=complete)
    return [
        n
        for n in req
        if not os.path.isfile(os.path.join(run_dir, n, triggers_filename))
    ]


def missing_flux_outputs_for_qa(
    run_dir: str,
    acs_root: str,
    events_matched_filename: str = 'events-matched.csv',
    *,
    complete: bool = False,
) -> list[str]:
    """ACS dataset names in scope with no ``events_matched_filename`` under ``run_dir``."""
    req = acs_datasets_for_mode(acs_root, complete=complete)
    return [
        n
        for n in req
        if not os.path.isfile(os.path.join(run_dir, n, events_matched_filename))
    ]


def is_run_complete_for_match(
    run_dir: str,
    acs_root: str,
    triggers_filename: str = 'triggers.npy',
    *,
    complete: bool = False,
) -> bool:
    return len(
        missing_flux_outputs_for_match(
            run_dir, acs_root, triggers_filename, complete=complete
        )
    ) == 0


def is_run_complete_for_qa(
    run_dir: str,
    acs_root: str,
    events_matched_filename: str = 'events-matched.csv',
    *,
    complete: bool = False,
) -> bool:
    return len(
        missing_flux_outputs_for_qa(
            run_dir, acs_root, events_matched_filename, complete=complete
        )
    ) == 0
