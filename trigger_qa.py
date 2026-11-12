"""
Trigger QA plots per ACS dataset under the current outputs/run_* directory.

Reads events-matched.csv, triggers.npy, trigger-merged.npy, and
trigger-spurious.csv (from trigger_match.py; same spurious list for every flux
folder, copied from the lowest-flux ``a`` band). Writes ``trigger-spurious-summary.txt``
and ``trigger-checks-spurious-summary-canonical.png`` under the run (offsets vs
injected catalog starts, SNR and per-algorithm stats). Writes several PNGs next
to those files in each dataset subfolder (overview, spurious-trigger insight on
the canonical folder only, spurious nearest-trigger offset sanity plot on the
canonical folder only, photon-flux summaries). At the run level it also writes a
``spurious-lightcurves/`` subfolder (alongside the per-dataset folders) with one
PNG per spurious trigger: six panels (one per BGO detector) of the gamma- and
proton-channel count rate around the trigger time, annotated with the per-channel
SNRs (recomputed as in trigger_acs.py) and an algorithm-tuning footer (marginal
SNR = the n_chan_over_thresh-th highest enabled σ, plus the lowest triggering
gamma/proton σ). All per-channel SNRs are also saved to
``trigger-spurious-channel-snr.csv`` at the run level.

Dataset names like ``flux-10-100a`` and ``flux-10-100b`` are treated as one
*flux group* (base ``flux-10-100``): per-folder QA still runs separately, but
additional *population* flux-summary figures merge all ``a``/``b`` members
into one pair of PNGs under the run directory.

After all datasets, writes global figures under that run: recovery efficiency
(fraction of catalog events with a matched trigger) vs peak photon flux for
**10–10000**, **10–1000**, **50–300**, and **80–2000 keV** (fixed log bin edges
per band). The 10–1000 and 50–300 keV series use companion flux CSVs aligned
by ``Event``; the others use columns in ``events-matched.csv``.

Also writes matching ``*-across-runs.png`` files under ``outputs/``, each
overlaying one band for every **complete** ``run_*`` (all expected flux datasets
must have ``events-matched.csv``; incomplete runs are skipped).

For **50–300 keV** global recovery and merged population summaries, the reference
curve from ``gbm_64ms.txt`` (flux vs recovery) is drawn and labeled
``Fermi-GBM 64 ms``.

Supplementary tab-separated CSVs next to each ``events.csv`` under
``sgrbs-scaled-background-binned-acs/<dataset>/`` (named like ``flux-10-100a.csv``)
provide peak photon flux for **10–1000 keV** and **50–300 keV**; those columns
are joined by ``Event`` id to each row of ``events-matched.csv`` for the
corresponding brightness-summary figures.

If ACS defines a flux suite (``trigger_outputs.expected_acs_dataset_names``),
**no** QA runs until **every dataset in the current scope** has a folder under the
run with ``events-matched.csv``. **Fast mode** (default) scopes QA to ``flux-1-10``
and ``flux-10-100`` ``a``/``b`` only; pass ``--complete`` on the command line to
require the full ACS flux suite. Pass ``--acs-root`` to use a different ACS tree.
With no ACS catalog, the same completeness rule
applies to every subfolder of the run directory.

Run this file from the IDE or with optional ``--complete`` / ``--acs-root``.

Time offsets in this script use trigger time minus (Start time + Peak start time)
per row — not the cross-match reference used in trigger_match.py.
"""

import csv
import os
import re
from collections import defaultdict
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator

import trigger_match
import trigger_outputs

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACS_ROOT = trigger_outputs.default_acs_root(_SCRIPT_DIR)
# Reference curve: columns = peak photon flux (ph/cm²/s), 50–300 keV | recovery efficiency
GBM_64MS_FILE = 'gbm_64ms.txt'
GBM_OVERLAY_LABEL = 'Fermi-GBM 64 ms'

EVENTS_MATCHED = 'events-matched.csv'
TRIGGERS_RAW = 'triggers.npy'
TRIGGERS_MERGED = 'trigger-merged.npy'
TRIGGERS_SPURIOUS = 'trigger-spurious.csv'

FIG_OVERVIEW = 'trigger-checks-overview.png'
FIG_SPURIOUS_INSIGHT = 'trigger-checks-spurious-insight.png'
# Sanity (may remove later): spurious Δt to nearest other merged / nearest matched trigger.
FIG_SPURIOUS_NEAREST_TRIGGER = 'trigger-checks-spurious-nearest-trigger-offset.png'
# Per-detector lightcurves around each spurious trigger (one PNG per trigger, in a subfolder).
FIG_SPURIOUS_LC_SUBDIR = 'spurious-lightcurves'
# Per-spurious-trigger per-channel SNR table (preserves all channel σ for algorithm tuning).
SPURIOUS_CHANNEL_SNR_CSV = 'trigger-spurious-channel-snr.csv'
# Legacy BGO detector order written by trigger_acs.py (z→b, x, y panels).
BGO_DETECTORS = ('b1', 'b2', 'x1', 'x2', 'y1', 'y2')
# 12 iop_trig channels (CTM ch0..ch11) in the exact order trigger_acs.py builds
# ``counts12``: per detector (gamma, proton) interleaved.
N_IOP_CHANNELS = 12
IOP_CHANNEL_DET = (
    'b1', 'b1', 'b2', 'b2', 'x1', 'x1', 'x2', 'x2',
    'y1', 'y1', 'y2', 'y2',
)
IOP_CHANNEL_KIND = (
    'gamma', 'proton', 'gamma', 'proton', 'gamma', 'proton', 'gamma', 'proton',
    'gamma', 'proton', 'gamma', 'proton',
)
# Short per-channel names (e.g. ``b1_g``, ``b1_p``).
IOP_CHANNEL_NAMES = tuple(
    f'{d}_{k[0]}' for d, k in zip(IOP_CHANNEL_DET, IOP_CHANNEL_KIND)
)
# Minimum window on each side of the spurious trigger time shown in the lightcurves.
SPURIOUS_LC_HALFWIN_S = 15.0
# Each lightcurve is binned at its firing algorithm's integration timescale
# (bin_size * dt from algorithms.ini). This fallback display bin is used only when
# that timescale cannot be resolved for the firing algorithm index.
SPURIOUS_LC_DISPLAY_BIN_S = 0.2
# Ensure at least this many timescale-width display bins per side; widen the window
# (beyond SPURIOUS_LC_HALFWIN_S) for long-timescale algorithms so context is visible.
SPURIOUS_LC_MIN_BINS_PER_SIDE = 12
# Safety cap on how many per-trigger lightcurve figures to write per dataset.
SPURIOUS_LC_MAX_FIGURES = 200
# Canonical (lowest-flux ``a``) spurious summary for the whole run:
SPURIOUS_SUMMARY_TXT = 'trigger-spurious-summary.txt'
FIG_SPURIOUS_SUMMARY_CANONICAL = 'trigger-checks-spurious-summary-canonical.png'
FIG_FLUX_10_1000 = 'trigger-checks-flux-10-1000keV.png'
FIG_FLUX_50_300 = 'trigger-checks-flux-50-300keV.png'
FIG_FLUX_80_2000 = 'trigger-checks-flux-80-2000keV.png'
FIG_GLOBAL_RECOVERY_10_10000 = 'trigger-checks-global-recovery-10-10000keV.png'
FIG_GLOBAL_RECOVERY_10_1000 = 'trigger-checks-global-recovery-10-1000keV.png'
FIG_GLOBAL_RECOVERY_50_300 = 'trigger-checks-global-recovery-50-300keV.png'
FIG_GLOBAL_RECOVERY_80_2000 = 'trigger-checks-global-recovery-80-2000keV.png'
FIG_GLOBAL_RECOVERY_ACROSS_RUNS_10_10000 = (
    'trigger-checks-global-recovery-across-runs-10-10000keV.png'
)
FIG_GLOBAL_RECOVERY_ACROSS_RUNS_10_1000 = (
    'trigger-checks-global-recovery-across-runs-10-1000keV.png'
)
FIG_GLOBAL_RECOVERY_ACROSS_RUNS_50_300 = (
    'trigger-checks-global-recovery-across-runs-50-300keV.png'
)
FIG_GLOBAL_RECOVERY_ACROSS_RUNS_80_2000 = (
    'trigger-checks-global-recovery-across-runs-80-2000keV.png'
)

# Global efficiency plots: fixed log flux range per band (photon flux, ph/cm^2/s)
GLOBAL_FLUX_EDGES_LO = 0.01
GLOBAL_FLUX_EDGES_HI = 1.0e4
GLOBAL_10_1000_EDGES_LO = 0.01
GLOBAL_10_1000_EDGES_HI = 1.0e3
GLOBAL_50_300_EDGES_LO = 0.001
GLOBAL_50_300_EDGES_HI = 2.0e2
GLOBAL_80_2000_EDGES_LO = 0.01
GLOBAL_80_2000_EDGES_HI = 1.0e4
GLOBAL_FLUX_N_BINS = 24

# Per-run filename, across-runs filename, short band label, x-axis label, log edges, flux index in aggregate tuple
_GLOBAL_RECOVERY_SPECS: list[tuple[str, str, str, str, float, float, int]] = [
    (
        FIG_GLOBAL_RECOVERY_10_10000,
        FIG_GLOBAL_RECOVERY_ACROSS_RUNS_10_10000,
        '10–10000 keV',
        'Peak photon flux (ph/cm²/s), 10–10000 keV',
        GLOBAL_FLUX_EDGES_LO,
        GLOBAL_FLUX_EDGES_HI,
        1,
    ),
    (
        FIG_GLOBAL_RECOVERY_10_1000,
        FIG_GLOBAL_RECOVERY_ACROSS_RUNS_10_1000,
        '10–1000 keV',
        'Peak photon flux (ph/cm²/s), 10–1000 keV',
        GLOBAL_10_1000_EDGES_LO,
        GLOBAL_10_1000_EDGES_HI,
        2,
    ),
    (
        FIG_GLOBAL_RECOVERY_50_300,
        FIG_GLOBAL_RECOVERY_ACROSS_RUNS_50_300,
        '50–300 keV',
        'Peak photon flux (ph/cm²/s), 50–300 keV',
        GLOBAL_50_300_EDGES_LO,
        GLOBAL_50_300_EDGES_HI,
        3,
    ),
    (
        FIG_GLOBAL_RECOVERY_80_2000,
        FIG_GLOBAL_RECOVERY_ACROSS_RUNS_80_2000,
        '80–2000 keV',
        'Peak photon flux (ph/cm²/s), 80–2000 keV',
        GLOBAL_80_2000_EDGES_LO,
        GLOBAL_80_2000_EDGES_HI,
        4,
    ),
]

COL_EVENT = 'Event'
COL_START = 'Start time (s)'
COL_PEAK_START = 'Peak start time (s)'
COL_MT = 'Matched trigger time (s)'
COL_MA = 'Matched trigger algorithm'
COL_MS = 'Matched trigger significance'
# Closest available to “10–1000 keV” brightness in the catalog:
COL_FLUX_10_10000 = 'Peak photon flux (ph/cm^2/s) for 10-10000 keV'
COL_FLUX_80_2000 = 'Peak photon flux (ph/cm^2/s) for 80-2000 keV'
COL_FLUX_ENERG = 'Peak energy flux (erg/cm^2/s) for 10-10000 keV'
# Same directory as events.csv: companion CSV from catalog (joined by Event id)
COL_FLUX_SUPP_10_1000 = 'Peak photon flux (ph/cm^2/s) for 10-1000 keV'
COL_FLUX_SUPP_50_300 = 'Peak photon flux (ph/cm^2/s) for 50-300 keV'


def load_events_matched_tsv(path: str) -> tuple[list[str], dict[str, list[str]]]:
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        rows = list(reader)
    if not rows:
        raise ValueError(f'Empty file: {path}')
    header = rows[0]
    data = {h: [] for h in header}
    for row in rows[1:]:
        if not row or all(c.strip() == '' for c in row):
            continue
        for i, h in enumerate(header):
            data[h].append(row[i].strip() if i < len(row) else '')
    return header, data


def col_float(data: dict[str, list[str]], key: str) -> np.ndarray:
    out = []
    for s in data[key]:
        try:
            out.append(float(s))
        except ValueError:
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def is_matched_row(mt: float, ma: float, ms: float) -> bool:
    return not (mt == 0.0 and ma == 0.0 and ms == 0.0)


def flux_pair_group_key(dataset_name: str) -> str:
    """
    Map ``flux-1-10a`` … ``flux-1-10d`` → ``flux-1-10``; otherwise return name unchanged.

    Only strips a single trailing lowercase letter when preceded by a digit (band
    replicate convention).
    """
    if (
        len(dataset_name) >= 2
        and 'a' <= dataset_name[-1] <= 'z'
        and dataset_name[-2].isdigit()
    ):
        return dataset_name[:-1]
    return dataset_name


def _safe_group_filename_token(group_key: str) -> str:
    return re.sub(r'[^0-9A-Za-z._-]+', '_', group_key)


def _read_data_temporal_resolution_s(run_dir: str) -> float:
    """First line of ``data.ini`` snapshot under the run directory (seconds)."""
    p = os.path.join(run_dir, 'data.ini')
    with open(p, encoding='utf-8') as f:
        return float(f.readline().strip())


def _algorithm_timescale_threshold_map(
    run_dir: str,
) -> dict[int, tuple[float, float, int]]:
    """
    Parse ``algorithms.ini`` like ``trigger_acs.parse_algorithms``: optional top
    line ``iop_trig_ra_cfg power_n delay_bins`` (ignored here); then
    ``index  bin_size  n_chan_over_thresh  [phase_steps]`` plus optional
    ``iop_trig_thresh_cfg[0..11]`` and ``iop_trig_chan_cfg[0..11]``.
    Maps algorithm index → ``(timescale_s, σ threshold used in sim, n_chan_over_thresh)``.
    ``timescale_s`` is **integration duration** ``bin_size * dt`` (ACS bin width from
    ``data.ini``). Legacy 27-token rows omit ``phase_steps`` (ignored for this map).
    For the extended layout, σ is the minimum configured threshold among **enabled**
    channels (raw / ``IOP_TRIG_THRESH_SCALE``). Legacy short rows use
    ``SIGMA_TRIGGER_THRESHOLD``.
    """
    import trigger_acs

    dt = _read_data_temporal_resolution_s(run_dir)
    n_iop = int(trigger_acs.N_IOP_TRIG_CHANNELS)
    scale = float(trigger_acs.IOP_TRIG_THRESH_SCALE)
    need_legacy = 3 + 2 * n_iop
    need_new = 4 + 2 * n_iop
    path = os.path.join(run_dir, 'algorithms.ini')
    out: dict[int, tuple[float, float, int]] = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'iop_trig_ra_cfg':
                continue
            if len(parts) < 3:
                continue
            try:
                index = int(float(parts[0]))
                bin_size = int(float(parts[1]))
                n_chan = int(float(parts[2]))
            except (ValueError, IndexError):
                continue
            if index in out or bin_size < 1:
                continue
            if len(parts) not in (need_legacy, need_new):
                continue
            thresh_off = 4 if len(parts) == need_new else 3
            if len(parts) >= need_legacy:
                if not (1 <= n_chan <= n_iop):
                    continue
                try:
                    thresh_sigma = np.array(
                        [float(parts[thresh_off + i]) / scale for i in range(n_iop)],
                        dtype=np.float64,
                    )
                    chan_en = np.array(
                        [
                            int(float(parts[thresh_off + n_iop + i]))
                            for i in range(n_iop)
                        ],
                        dtype=np.int64,
                    )
                except (ValueError, IndexError):
                    continue
                enabled = chan_en == 1
                if not np.any(enabled):
                    continue
                sigma_fix = float(np.min(thresh_sigma[enabled]))
            else:
                if not (1 <= n_chan <= 6):
                    continue
                sigma_fix = float(trigger_acs.SIGMA_TRIGGER_THRESHOLD)
            timescale = float(bin_size) * dt
            out[index] = (timescale, sigma_fix, n_chan)
    return out


def plot_spurious_insight_figure(
    dataset_dir: str,
    label: str,
    run_dir: str,
    spurious: np.ndarray,
) -> None:
    """
    Summarize spurious merged triggers: firing algorithm index, configured
    timescale (from ``algorithms.ini``), observed SNR (σ), and timescale–SNR
    scatter (color = algorithm). Footer: σ relative to configured threshold.
    """
    out_path = os.path.join(dataset_dir, FIG_SPURIOUS_INSIGHT)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        f'Spurious triggers — algorithm, timescale, SNR ({label})',
        fontsize=12,
    )
    ax00, ax01, ax10, ax11 = axes.ravel()

    n = len(spurious)
    if n == 0:
        for ax in axes.ravel():
            ax.text(
                0.5,
                0.5,
                'No spurious merged triggers\n(none within match window of any catalog start)',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Saved figure: {out_path}')
        return

    t_trig = spurious[:, 0].astype(np.float64)
    alg_idx = np.rint(spurious[:, 1]).astype(np.int64)
    snr = spurious[:, 2].astype(np.float64)

    try:
        algo_map = _algorithm_timescale_threshold_map(run_dir)
    except (OSError, ValueError, FileNotFoundError):
        algo_map = {}

    ts_list: list[float] = []
    thr_list: list[float] = []
    for a in alg_idx:
        row = algo_map.get(int(a))
        if row is None:
            ts_list.append(float('nan'))
            thr_list.append(float('nan'))
        else:
            ts_list.append(row[0])
            thr_list.append(row[1])
    ts_arr = np.asarray(ts_list, dtype=np.float64)
    thr_arr = np.asarray(thr_list, dtype=np.float64)

    # --- Algorithm index counts ---
    uniq, counts = np.unique(alg_idx, return_counts=True)
    ax00.bar(
        [str(int(u)) for u in uniq],
        counts.astype(float),
        color='indianred',
        edgecolor='black',
        linewidth=0.5,
    )
    ax00.set_xlabel('Algorithm index')
    ax00.set_ylabel('Count')
    ax00.set_title('Spurious triggers by algorithm')
    ax00.tick_params(axis='x', rotation=45)

    # --- Configured timescale (s) ---
    ts_ok = ts_arr[np.isfinite(ts_arr) & (ts_arr > 0)]
    if len(ts_ok) > 0:
        u_ts = np.unique(np.sort(ts_ok))
        if len(u_ts) <= 20:
            tc = [int(np.sum(ts_arr == v)) for v in u_ts]
            ax01.bar(
                [str(v) for v in u_ts],
                tc,
                color='sandybrown',
                edgecolor='black',
                linewidth=0.5,
            )
            ax01.set_xlabel('Timescale (s) from algorithms.ini')
            ax01.tick_params(axis='x', rotation=45)
        else:
            ax01.hist(
                ts_ok,
                bins=min(24, max(8, len(ts_ok) // 3)),
                color='sandybrown',
                edgecolor='black',
                alpha=0.85,
            )
            ax01.set_xlabel('Timescale (s)')
        ax01.set_ylabel('Count')
        ax01.set_title('Spurious by trigger timescale')
    else:
        ax01.text(
            0.5,
            0.5,
            'No timescale map\n(check algorithms.ini)',
            ha='center',
            va='center',
            transform=ax01.transAxes,
        )
        ax01.set_axis_off()

    # --- Observed SNR (σ) ---
    snr_ok = snr[np.isfinite(snr) & (snr > 0)]
    if len(snr_ok) > 0:
        lo, hi = float(np.min(snr_ok)), float(np.max(snr_ok))
        if lo >= hi:
            lo, hi = lo * 0.99, hi * 1.01
        bins = np.logspace(np.log10(lo), np.log10(hi), min(28, max(10, n * 2)))
        ax10.hist(snr_ok, bins=bins, color='darkred', edgecolor='black', alpha=0.85)
        ax10.set_xscale('log')
        ax10.set_xlabel('Observed SNR (σ)')
        ax10.set_ylabel('Count')
        ax10.set_title('Spurious trigger significance')
    else:
        ax10.text(0.5, 0.5, 'No positive σ', ha='center', va='center', transform=ax10.transAxes)
        ax10.set_axis_off()

    # --- Timescale vs observed SNR ---
    m_sc = np.isfinite(ts_arr) & (ts_arr > 0) & np.isfinite(snr) & (snr > 0)
    if np.any(m_sc):
        sc = ax11.scatter(
            ts_arr[m_sc],
            snr[m_sc],
            c=alg_idx[m_sc],
            cmap='tab20',
            alpha=0.75,
            s=42,
            edgecolors='k',
            linewidths=0.35,
        )
        cbar = fig.colorbar(sc, ax=ax11, fraction=0.046, pad=0.02)
        cbar.set_label('Algorithm index')
        _style_log_axis(ax11, 'x')
        _style_log_axis(ax11, 'y')
        ax11.set_xlabel('Timescale (s) from algorithms.ini')
        ax11.set_ylabel('Observed SNR (σ)')
        ax11.set_title('Spurious: timescale vs SNR (color = algorithm)')
    else:
        ax11.text(
            0.5,
            0.5,
            'No spurious with mapped\ntimescale and σ > 0',
            ha='center',
            va='center',
            transform=ax11.transAxes,
        )
        ax11.set_axis_off()

    # Optional footer: excess over threshold
    ratio = snr / thr_arr
    ratio = ratio[np.isfinite(ratio) & (thr_arr > 0) & (snr > 0)]
    if len(ratio) > 0:
        fig.text(
            0.5,
            0.01,
            f'Spurious σ / ini threshold — median: {np.median(ratio):.2f}, '
            f'max: {np.max(ratio):.2f} (over {len(ratio)} triggers with mapped threshold)',
            ha='center',
            fontsize=9,
            style='italic',
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved figure: {out_path}')


def _spurious_dt_nearest_other_merged(spurious: np.ndarray, merged: np.ndarray) -> np.ndarray:
    """
    For each spurious row, min |t_spur − t_m| over merged rows that are not the same
    (time, algorithm, sigma) row.
    """
    spurious = np.asarray(spurious, dtype=np.float64)
    merged = np.asarray(merged, dtype=np.float64)
    if spurious.size == 0:
        return np.array([], dtype=np.float64)
    if merged.size == 0:
        return np.full(len(spurious), np.nan, dtype=np.float64)
    out = np.empty(len(spurious), dtype=np.float64)
    for k, r in enumerate(spurious):
        d = np.abs(merged[:, 0] - r[0])
        same = (
            np.isclose(merged[:, 0], r[0], rtol=0.0, atol=1e-5)
            & np.isclose(merged[:, 1], r[1], rtol=0.0, atol=1e-4)
            & np.isclose(merged[:, 2], r[2], rtol=0.0, atol=1e-3)
        )
        d2 = d[~same]
        out[k] = float(np.min(d2)) if d2.size > 0 else np.nan
    return out


def _spurious_dt_nearest_in_sorted_times(t_spur: np.ndarray, t_sorted: np.ndarray) -> np.ndarray:
    """For each time in t_spur, min |t − u| over u in t_sorted (1D, sorted, finite)."""
    t_spur = np.asarray(t_spur, dtype=np.float64)
    t_sorted = np.asarray(t_sorted, dtype=np.float64)
    t_sorted = t_sorted[np.isfinite(t_sorted)]
    if t_spur.size == 0:
        return np.array([], dtype=np.float64)
    if t_sorted.size == 0:
        return np.full(len(t_spur), np.nan, dtype=np.float64)
    t_sorted = np.sort(np.unique(t_sorted))
    out = np.empty(len(t_spur), dtype=np.float64)
    for k, t_s in enumerate(t_spur):
        i = int(np.searchsorted(t_sorted, t_s))
        best = np.inf
        if i > 0:
            best = min(best, abs(t_s - t_sorted[i - 1]))
        if i < len(t_sorted):
            best = min(best, abs(t_s - t_sorted[i]))
        out[k] = best if np.isfinite(best) else np.nan
    return out


def plot_spurious_nearest_trigger_offset_figure(
    dataset_dir: str,
    label: str,
    spurious: np.ndarray,
    merged: np.ndarray,
    matched: np.ndarray,
    t_matched_trig: np.ndarray,
) -> None:
    """
    Sanity figure: absolute time offset (s) from each spurious merged trigger to
    (1) the nearest *other* merged trigger, and (2) the nearest matched-trigger time
    from the catalog. Vertical line: ``trigger_match.CROSSMATCH_WINDOW_S`` used
    when labeling spurious (no event start within that window).
    """
    out_path = os.path.join(dataset_dir, FIG_SPURIOUS_NEAREST_TRIGGER)
    win = float(trigger_match.CROSSMATCH_WINDOW_S)
    n_sp = len(spurious)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.suptitle(
        f'Spurious triggers — distance to nearest trigger in time ({label})\n'
        f'(sanity check for cross-match window = {win:g} s)',
        fontsize=11,
    )

    if n_sp == 0:
        for ax in (ax0, ax1):
            ax.text(
                0.5,
                0.5,
                'No spurious merged triggers',
                ha='center',
                va='center',
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Saved figure: {out_path}')
        return

    dt_other = _spurious_dt_nearest_other_merged(spurious, merged)
    t_pool = t_matched_trig[np.asarray(matched, dtype=bool)]
    t_pool = t_pool[np.isfinite(t_pool) & (np.abs(t_pool) > 1e-12)]
    dt_match = (
        _spurious_dt_nearest_in_sorted_times(spurious[:, 0], t_pool)
        if t_pool.size > 0
        else np.full(len(spurious), np.nan, dtype=np.float64)
    )

    def _hist(ax, data: np.ndarray, title: str, xlabel: str) -> None:
        data = np.asarray(data, dtype=np.float64)
        data = data[np.isfinite(data) & (data >= 0)]
        if data.size == 0:
            ax.text(0.5, 0.5, 'No finite offsets', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return
        hi = float(np.max(data))
        hi = max(hi, win * 2.0, 1.0)
        bins = np.linspace(0.0, hi, min(48, max(12, int(np.sqrt(data.size)) * 4)))
        ax.hist(data, bins=bins, color='steelblue', edgecolor='black', alpha=0.85)
        ax.axvline(win, color='C3', linestyle='--', linewidth=1.5, label=f'Cross-match window ({win:g} s)')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.legend(loc='upper right', fontsize=8)
        med = float(np.median(data))
        ax.text(
            0.02,
            0.98,
            f'n = {data.size}\nmedian |Δt| = {med:.3g} s',
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

    _hist(
        ax0,
        dt_other,
        'Nearest other merged trigger',
        '|t_spurious − t_merged_other| (s)',
    )
    if t_pool.size == 0:
        ax1.text(
            0.5,
            0.5,
            'No matched trigger times in catalog\n(cannot compute panel)',
            ha='center',
            va='center',
            transform=ax1.transAxes,
        )
        ax1.set_axis_off()
    else:
        _hist(
            ax1,
            dt_match,
            'Nearest matched-trigger time',
            '|t_spurious − t_matched| (s)',
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved figure: {out_path}')


def _find_binned_cache_dir(acs_root: str, dataset_name: str) -> str | None:
    """
    Directory holding the per-detector binned-count caches written by
    ``trigger_acs.py`` (``bgo_<det>_binned_counts_gamma_channel.npy`` and
    ``bin_times.npy``) — i.e. the folder next to ``combined_acs_data.hdf5`` under
    ``acs_root/<dataset>/``. Returns None if the cache is not present.
    """
    import glob

    direct = os.path.join(acs_root, dataset_name, 'bin_times.npy')
    if os.path.isfile(direct):
        return os.path.dirname(direct)
    # Legacy nested layout: acs_root/<dataset>/*/bin_times.npy
    pattern = os.path.join(acs_root, dataset_name, '*', 'bin_times.npy')
    found = sorted(glob.glob(pattern))
    return os.path.dirname(found[0]) if found else None


def _load_detector_gp_counts(
    cache_dir: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load bin-start times (s) and the gamma- and proton-channel binned counts for
    each BGO detector from a binned cache directory. ``bin_times.npy`` is relative
    to the first ACS bin; ``time_reference_s.npy`` is added back so the times share
    the absolute (events.csv) timebase used for trigger / spurious times.
    """
    times = np.load(os.path.join(cache_dir, 'bin_times.npy'))
    times = np.asarray(times, dtype=np.float64)
    t_start = times[:, 0] if times.ndim == 2 else times
    ref_path = os.path.join(cache_dir, 'time_reference_s.npy')
    if os.path.isfile(ref_path):
        t_start = t_start + float(np.load(ref_path))
    gamma: dict[str, np.ndarray] = {}
    proton: dict[str, np.ndarray] = {}
    for det in BGO_DETECTORS:
        pg = os.path.join(cache_dir, f'bgo_{det}_binned_counts_gamma_channel.npy')
        pp = os.path.join(cache_dir, f'bgo_{det}_binned_counts_proton_channel.npy')
        gamma[det] = np.asarray(np.load(pg), dtype=np.float64)
        proton[det] = np.asarray(np.load(pp), dtype=np.float64)
    return np.asarray(t_start, dtype=np.float64), gamma, proton


def _build_counts12(
    gamma: dict[str, np.ndarray], proton: dict[str, np.ndarray]
) -> tuple[np.ndarray, ...]:
    """Interleave per-detector (gamma, proton) into the 12-channel tuple trigger_acs uses."""
    chans: list[np.ndarray] = []
    for det in BGO_DETECTORS:
        chans.append(gamma[det])
        chans.append(proton[det])
    return tuple(chans)


def _fmt_sigma(v: float) -> str:
    """Human-readable σ: blank/dash for the trigger's ``-999`` sentinel or non-finite."""
    return f'{v:.2f}' if np.isfinite(v) and v > -900.0 else '—'


def _spurious_channel_snr(
    counts12: tuple[np.ndarray, ...],
    time_dx: int,
    bg_mean_start_bins: int,
    bg_mean_end_bins: int,
    algo_row: tuple[int, int, int, int, np.ndarray, np.ndarray] | None,
) -> dict[str, object]:
    """
    Recompute the 12 per-channel SNRs at a spurious trigger bin (exactly as
    ``trigger_acs`` did) and derive algorithm-tuning quantities for the firing
    algorithm row ``(index, bin_size, n_chan_over_thresh, phase_steps, thresh_sigma,
    chan_enable)``.

    Returns a dict with the full ``sigmas`` array plus, for the enabled channels:
    ``marginal_snr`` (the ``n_chan_over_thresh``-th highest enabled σ — the largest
    uniform threshold that would still fire), the list of triggering channels
    (enabled and over their threshold), and the lowest triggering gamma / proton σ.
    """
    import trigger_acs

    if algo_row is None:
        return {'sigmas': np.full(N_IOP_CHANNELS, np.nan), 'algo': None}

    _idx, bin_size, n_chan_over_thresh, _phase, thresh_sigma, chan_enable = algo_row
    sigmas = trigger_acs.compute_iop_trig_sigmas(
        counts12, int(time_dx), bg_mean_start_bins, bg_mean_end_bins, int(bin_size)
    )
    enabled = np.asarray(chan_enable, dtype=np.int64) == 1
    thr = np.asarray(thresh_sigma, dtype=np.float64)
    over = enabled & (sigmas > thr)

    enabled_sigs = np.sort(sigmas[enabled])[::-1] if np.any(enabled) else np.array([])
    if enabled_sigs.size >= n_chan_over_thresh >= 1:
        marginal_snr = float(enabled_sigs[n_chan_over_thresh - 1])
    else:
        marginal_snr = float('nan')

    trig_idx = [c for c in range(N_IOP_CHANNELS) if over[c]]
    trig_gamma = [c for c in trig_idx if IOP_CHANNEL_KIND[c] == 'gamma']
    trig_proton = [c for c in trig_idx if IOP_CHANNEL_KIND[c] == 'proton']
    lowest_trig_gamma = (
        float(np.min(sigmas[trig_gamma])) if trig_gamma else float('nan')
    )
    lowest_trig_proton = (
        float(np.min(sigmas[trig_proton])) if trig_proton else float('nan')
    )

    return {
        'sigmas': sigmas,
        'thresh': thr,
        'enabled': enabled,
        'over': over,
        'n_chan_over_thresh': int(n_chan_over_thresh),
        'n_over': int(np.sum(over)),
        'marginal_snr': marginal_snr,
        'trig_idx': trig_idx,
        'lowest_trig_gamma': lowest_trig_gamma,
        'lowest_trig_proton': lowest_trig_proton,
        'algo': algo_row,
    }


def _rebin_sum(t_start: np.ndarray, counts: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum ``counts`` over consecutive groups of ``factor`` bins; keep each group's first start time."""
    factor = int(factor)
    if factor <= 1:
        return t_start, counts
    n = (len(t_start) // factor) * factor
    if n < factor:
        return t_start, counts
    t = t_start[:n].reshape(-1, factor)[:, 0]
    c = counts[:n].reshape(-1, factor).sum(axis=1)
    return t, c


def _spurious_snr_footer(alg: int, snr: dict[str, object]) -> str:
    """One-line algorithm-tuning summary for a spurious trigger figure footer."""
    if snr.get('algo') is None:
        return (
            f'Algorithm {alg}: not found in algorithms.ini snapshot — '
            'per-channel SNRs unavailable.'
        )
    sigmas = np.asarray(snr['sigmas'], dtype=np.float64)
    over = snr['over']
    n_req = snr['n_chan_over_thresh']
    n_over = snr['n_over']
    marginal = snr['marginal_snr']
    trig = [
        f'{IOP_CHANNEL_NAMES[c]}={_fmt_sigma(float(sigmas[c]))}'
        for c in snr['trig_idx']
    ]
    trig_str = ', '.join(trig) if trig else 'none'
    lg = snr['lowest_trig_gamma']
    lp = snr['lowest_trig_proton']
    return (
        f'Need {n_req} ch over threshold; {n_over} over.  '
        f'Marginal SNR (#{n_req} highest enabled) = {_fmt_sigma(float(marginal))} σ '
        f'(max uniform threshold that still fires).  '
        f'Lowest triggering: γ={_fmt_sigma(float(lg))} σ, p={_fmt_sigma(float(lp))} σ.\n'
        f'Triggering channels: {trig_str}'
    )


def _write_spurious_channel_snr_csv(
    path: str, label: str, records: list[dict[str, object]]
) -> None:
    """
    Tab-separated table of every spurious trigger's per-channel SNRs (all 12
    channels) plus the algorithm-tuning derived quantities. ``-999``/non-finite
    σ are written blank.
    """
    real_chans = list(range(N_IOP_CHANNELS))
    header = [
        'Source dataset', 'Trigger time (s)', 'Algorithm', 'Merged significance',
        'Timescale (s)', 'n_chan_over_thresh', 'n_over',
        'Marginal SNR', 'Lowest triggering gamma SNR', 'Lowest triggering proton SNR',
    ] + [f'{IOP_CHANNEL_NAMES[c]} SNR' for c in real_chans]

    def _num(v: object) -> str:
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return ''
        return f'{f:.6g}' if np.isfinite(f) and f > -900.0 else ''

    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, delimiter='\t', lineterminator='\n')
        w.writerow(header)
        for rec in records:
            snr = rec['snr']  # type: ignore[index]
            sigmas = np.asarray(snr['sigmas'], dtype=np.float64)  # type: ignore[index]
            if snr.get('algo') is None:  # type: ignore[union-attr]
                row = [
                    label, _num(rec['t_trig']), str(rec['alg']), _num(rec['sig']),
                    _num(rec['timescale_s']), '', '', '', '', '',
                ] + ['' for _ in real_chans]
            else:
                row = [
                    label, _num(rec['t_trig']), str(rec['alg']), _num(rec['sig']),
                    _num(rec['timescale_s']),
                    str(snr['n_chan_over_thresh']), str(snr['n_over']),  # type: ignore[index]
                    _num(snr['marginal_snr']),  # type: ignore[index]
                    _num(snr['lowest_trig_gamma']),  # type: ignore[index]
                    _num(snr['lowest_trig_proton']),  # type: ignore[index]
                ] + [_num(sigmas[c]) for c in real_chans]
            w.writerow(row)


def plot_spurious_per_detector_lightcurves(
    dataset_dir: str,
    label: str,
    spurious: np.ndarray,
    acs_root: str,
) -> None:
    """
    For each spurious merged trigger, write one PNG with six panels (one per BGO
    detector) showing the gamma-channel (solid) and proton-channel (dashed, right
    axis) count rate around the trigger time, plus the per-channel SNRs that drove
    the trigger so the algorithm can be retuned.

    Each figure is binned at the **firing algorithm's integration timescale**
    (``bin_size * dt`` from ``algorithms.ini``); ``SPURIOUS_LC_DISPLAY_BIN_S`` is used
    only as a fallback when that timescale cannot be resolved. The window is at
    least ±``SPURIOUS_LC_HALFWIN_S`` and is widened so each side holds at least
    ``SPURIOUS_LC_MIN_BINS_PER_SIDE`` timescale-width bins.

    SNRs are recomputed at the trigger bin exactly as ``trigger_acs`` did
    (``compute_iop_trig_sigmas`` with the same background window). Each panel is
    annotated with that detector's gamma/proton σ and whether the channel is
    enabled (E) and over threshold (✓). A footer reports, for the firing algorithm,
    the **marginal SNR** (the ``n_chan_over_thresh``-th highest σ among enabled
    channels — the largest uniform threshold that would still fire), the triggering
    channels, and the lowest triggering gamma / proton σ. All per-channel SNRs are
    also written to ``<run_dir>/`` ``trigger-spurious-channel-snr.csv``.

    Figures are written to ``<run_dir>/spurious-lightcurves/`` (the run level,
    alongside the per-dataset folders) since the spurious list is shared across
    flux bands. Requires the per-detector binned caches from ``trigger_acs.py`` next
    to ``combined_acs_data.hdf5`` under ``acs_root/<label>/``.
    """
    n = len(spurious)
    if n == 0:
        print(f'No spurious triggers — no per-detector lightcurves ({label}).')
        return

    cache_dir = _find_binned_cache_dir(acs_root, label)
    if cache_dir is None:
        print(
            f'Skip spurious lightcurves ({label}): no binned cache under '
            f'{acs_root}/{label}/ (run trigger_acs.py to generate it).'
        )
        return

    try:
        t_start, gamma, proton = _load_detector_gp_counts(cache_dir)
    except (OSError, ValueError) as e:
        print(f'Skip spurious lightcurves ({label}): {e}')
        return

    if len(t_start) < 2:
        print(f'Skip spurious lightcurves ({label}): too few time bins in cache.')
        return

    counts12 = _build_counts12(gamma, proton)
    dt = float(np.median(np.diff(t_start)))
    fallback_factor = max(1, int(round(SPURIOUS_LC_DISPLAY_BIN_S / dt))) if dt > 0 else 1

    run_dir = os.path.dirname(dataset_dir)
    # Full algorithm config + background window, parsed exactly as trigger_acs did,
    # so per-channel SNRs reproduce the values that fired each trigger.
    algo_full: dict[int, tuple] = {}
    bg_mean_start_bins = bg_mean_end_bins = 0
    try:
        import trigger_acs

        dt_data = _read_data_temporal_resolution_s(run_dir)
        algorithms, bg_mean_start_bins, bg_mean_end_bins = trigger_acs.parse_algorithms(
            os.path.join(run_dir, 'algorithms.ini'), dt_data
        )
        algo_full = {int(row[0]): row for row in algorithms}
    except (OSError, ValueError, FileNotFoundError) as e:
        print(f'Spurious lightcurves ({label}): could not parse algorithms.ini ({e}); '
              'SNR annotations will be unavailable.')

    out_dir = os.path.join(run_dir, FIG_SPURIOUS_LC_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    n_make = min(n, SPURIOUS_LC_MAX_FIGURES)
    if n_make < n:
        print(
            f'Spurious lightcurves ({label}): writing first {n_make} of {n} '
            f'(cap = {SPURIOUS_LC_MAX_FIGURES}).'
        )

    det_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    snr_records: list[dict[str, object]] = []
    n_written = 0
    for i in range(n_make):
        t_trig = float(spurious[i, 0])
        alg = int(round(spurious[i, 1]))
        sig = float(spurious[i, 2])

        algo_row = algo_full.get(alg)

        # Bin width = firing algorithm's integration timescale (fallback otherwise).
        timescale_s = float(algo_row[1]) * dt if (algo_row is not None and dt > 0) else float('nan')
        if dt > 0 and np.isfinite(timescale_s) and timescale_s > 0:
            factor = max(1, int(round(timescale_s / dt)))
            ts_note = f'algorithm timescale {timescale_s:.3g} s'
        else:
            factor = fallback_factor
            ts_note = f'fallback {SPURIOUS_LC_DISPLAY_BIN_S:.3g} s (no timescale map)'
        bin_width_s = factor * dt if dt > 0 else 1.0

        # Bin index of the trigger (spurious time == ACS bin start written by trigger_acs).
        time_dx = int(np.argmin(np.abs(t_start - t_trig)))

        snr = _spurious_channel_snr(
            counts12, time_dx, bg_mean_start_bins, bg_mean_end_bins, algo_row
        )
        sigmas = np.asarray(snr['sigmas'], dtype=np.float64)
        snr_records.append(
            {'i': i, 't_trig': t_trig, 'alg': alg, 'sig': sig,
             'timescale_s': timescale_s, 'snr': snr}
        )

        # Window: at least ±SPURIOUS_LC_HALFWIN_S, widened so each side holds enough bins.
        halfwin = max(SPURIOUS_LC_HALFWIN_S, bin_width_s * SPURIOUS_LC_MIN_BINS_PER_SIDE)
        lo = t_trig - halfwin
        hi = t_trig + halfwin
        idx = np.nonzero((t_start >= lo) & (t_start <= hi))[0]
        if idx.size == 0:
            print(
                f'  Spurious #{i} (t={t_trig:.3f} s): no cache bins in window; skipped.'
            )
            continue
        s, e = int(idx[0]), int(idx[-1]) + 1
        t_win = t_start[s:e]

        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f'Spurious trigger per-detector lightcurves + channel SNR ({label})\n'
            f't = {t_trig:.3f} s  |  algorithm {alg}  |  merged σ = {sig:.2f}  '
            f'|  {bin_width_s:.3g} s bins ({ts_note})',
            fontsize=11,
        )
        enabled = snr.get('enabled')
        over = snr.get('over')
        thr = snr.get('thresh')
        for pos, (ax, det, color) in enumerate(zip(axes.ravel(), BGO_DETECTORS, det_colors)):
            gi, pi = 2 * pos, 2 * pos + 1
            tb, cg = _rebin_sum(t_win, gamma[det][s:e], factor)
            _tb, cp = _rebin_sum(t_win, proton[det][s:e], factor)
            ax.step(tb - t_trig, cg / bin_width_s, where='post', color=color, linewidth=1.0,
                    label='gamma')
            ax.axvline(0.0, color='k', linestyle='--', linewidth=1.2, alpha=0.7)
            ax.set_title(f'BGO {det}', fontsize=10)
            ax.set_ylabel('γ rate (cts/s)', fontsize=9)
            ax.grid(True, alpha=0.25)

            axp = ax.twinx()
            axp.step(_tb - t_trig, cp / bin_width_s, where='post', color='dimgray',
                     linestyle=':', linewidth=0.9, alpha=0.75)
            axp.tick_params(axis='y', labelsize=7, colors='dimgray')
            if pos % 2 == 1:
                axp.set_ylabel('p rate (cts/s)', fontsize=8, color='dimgray')

            sg = float(sigmas[gi]) if sigmas.size > gi else float('nan')
            sp_ = float(sigmas[pi]) if sigmas.size > pi else float('nan')

            def _mark(ch: int) -> str:
                if enabled is None or over is None:
                    return ''
                en = 'E' if bool(enabled[ch]) else '·'
                ov = '✓' if bool(over[ch]) else ''
                t = f' (thr {float(thr[ch]):.2f})' if thr is not None and bool(enabled[ch]) else ''
                return f' [{en}{ov}]{t}'

            ann = (
                f'γ σ={_fmt_sigma(sg)}{_mark(gi)}\n'
                f'p σ={_fmt_sigma(sp_)}{_mark(pi)}'
            )
            ax.text(
                0.015, 0.97, ann, transform=ax.transAxes, va='top', ha='left',
                fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='0.7'),
            )
        for ax in axes[-1, :]:
            ax.set_xlabel('Time since spurious trigger (s)')

        fig.text(
            0.5, 0.012, _spurious_snr_footer(alg, snr),
            ha='center', va='bottom', fontsize=8.5,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.93])
        out_path = os.path.join(
            out_dir, f'alg{alg:02d}-spurious-lc-{i:03d}-t{t_trig:.3f}.png'
        )
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        n_written += 1

    csv_path = os.path.join(run_dir, SPURIOUS_CHANNEL_SNR_CSV)
    _write_spurious_channel_snr_csv(csv_path, label, snr_records)
    print(f'Saved {n_written} spurious per-detector lightcurve figure(s) under: {out_dir}')
    print(f'Wrote per-channel spurious SNR table: {csv_path}')


def load_spurious_array(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        return np.empty((0, 3), dtype=np.float64)
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    if len(rows) < 2:
        return np.empty((0, 3), dtype=np.float64)
    out = []
    for row in rows[1:]:
        if not row or all(c.strip() == '' for c in row):
            continue
        try:
            out.append([float(row[0]), float(row[1]), float(row[2])])
        except (ValueError, IndexError):
            continue
    if not out:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _plain_sigma_x_tick(x: float, pos: int | None) -> str:
    """Log-axis σ tick labels without scientific notation (canonical spurious summary)."""
    if not np.isfinite(x) or x <= 0:
        return ''
    if x >= 100:
        return f'{int(round(x))}'
    if x >= 10:
        if abs(x - round(x)) < 0.05 * max(x, 1.0):
            return f'{int(round(x))}'
        return f'{x:.1f}'.rstrip('0').rstrip('.')
    if x >= 1:
        if abs(x - round(x)) < 1e-5:
            return f'{int(round(x))}'
        return f'{x:.2f}'.rstrip('0').rstrip('.')
    s = f'{x:.4f}'.rstrip('0').rstrip('.')
    return s if s else ''


def summarize_spurious_canonical_run(
    run_dir: str,
    acs_root: str,
    *,
    suite_names: Sequence[str] | None = None,
) -> None:
    """
    One-run summary using **only** the lowest-flux ``a`` catalog (same spurious list
    as all flux / ``a``/``b`` copies): nearest injected-event **start** time offset
    per spurious row, global minimum offset, and SNR (σ) distribution including
    breakdown by firing algorithm index.

    ``suite_names`` restricts which ACS folders exist for this run (e.g. fast-mode
    subset); when ``None``, all ``expected_acs_dataset_names`` are considered.
    """
    suite = (
        list(suite_names)
        if suite_names is not None
        else trigger_outputs.expected_acs_dataset_names(acs_root)
    )
    canon = (
        trigger_outputs.canonical_spurious_source_dataset_name(
            acs_root, dataset_names=suite
        )
        if suite
        else None
    )
    txt_path = os.path.join(run_dir, SPURIOUS_SUMMARY_TXT)
    fig_path = os.path.join(run_dir, FIG_SPURIOUS_SUMMARY_CANONICAL)

    if not canon:
        msg = (
            'Spurious canonical summary skipped: no expected ACS suite or no '
            'lowest-flux ``flux-*`` folder ending in ``a``.\n'
        )
        print('=== Spurious summary (canonical, lowest-flux a) ===\n' + msg)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(msg)
        return

    path_sp = os.path.join(run_dir, canon, TRIGGERS_SPURIOUS)
    if not os.path.isfile(path_sp):
        msg = f'Spurious canonical summary skipped: missing {path_sp}\n'
        print('=== Spurious summary (canonical, lowest-flux a) ===\n' + msg)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(msg)
        return

    sp = load_spurious_array(path_sp)
    ev_path = trigger_outputs.find_events_csv(acs_root, canon)
    if not ev_path:
        msg = f'Spurious canonical summary skipped: no events.csv for {canon!r}\n'
        print('=== Spurious summary (canonical, lowest-flux a) ===\n' + msg)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(msg)
        return

    evt = trigger_match.load_finite_event_start_times(ev_path, trigger_match.START_TIME_COL)
    evt = evt[np.isfinite(evt)]
    win = float(trigger_match.CROSSMATCH_WINDOW_S)

    lines: list[str] = []
    lines.append(f'# Spurious trigger summary (canonical dataset: {canon})')
    lines.append('# Spurious list is identical for all flux bands and a/b variants.')
    lines.append(f'# Catalog: {ev_path}')
    lines.append(f'# Injected event starts (column {trigger_match.START_TIME_COL!r}): {len(evt)}')
    lines.append(f'# Spurious merged triggers (no start within {win:g} s): {len(sp)}')
    lines.append('')

    if len(sp) == 0:
        lines.append('No spurious triggers.')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print('=== Spurious summary (canonical, lowest-flux a) ===')
        print(f'Wrote {txt_path} (0 spurious).')
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'No spurious triggers', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f'Saved figure: {fig_path}\n')
        return

    t_sp = sp[:, 0].astype(np.float64)
    alg = np.rint(sp[:, 1]).astype(np.int64)
    sig = sp[:, 2].astype(np.float64)

    if evt.size == 0:
        dt_near = np.full(len(sp), np.nan, dtype=np.float64)
        lines.append('No catalog event start times; offsets undefined.')
    else:
        dt_near = np.min(np.abs(t_sp[:, np.newaxis] - evt[np.newaxis, :]), axis=1)

    ok_dt = np.isfinite(dt_near) & (dt_near >= 0)
    global_min_dt = float(np.min(dt_near[ok_dt])) if np.any(ok_dt) else float('nan')
    lines.append(
        'Absolute time offset to nearest injected catalog start (s), per spurious trigger:'
    )
    lines.append(f'  global minimum: {global_min_dt:.6g}')
    if np.any(ok_dt):
        d = dt_near[ok_dt]
        for p in (50, 90, 95, 99):
            lines.append(f'  p{p}: {float(np.percentile(d, p)):.6g}')
        lines.append(f'  median: {float(np.median(d)):.6g}')
        lines.append(f'  max: {float(np.max(d)):.6g}')
    lines.append('')
    lines.append('SNR (merged significance σ, spurious rows):')
    s_ok = sig[np.isfinite(sig) & (sig > 0)]
    if len(s_ok):
        lines.append(
            f'  min {float(np.min(s_ok)):.4g}  median {float(np.median(s_ok)):.4g}  '
            f'mean {float(np.mean(s_ok)):.4g}  std {float(np.std(s_ok)):.4g}  '
            f'max {float(np.max(s_ok)):.4g}'
        )
    else:
        lines.append('  (no finite positive σ)')
    lines.append('')
    lines.append('Per firing algorithm index (iop_trig_test_cfg id):')
    lines.append('  idx  n  median_σ  min_σ  max_σ  min_offset_s  median_offset_s')
    uniq_alg = np.unique(alg)
    for a in uniq_alg:
        m = alg == a
        dta = dt_near[m & ok_dt] if np.any(ok_dt) else np.array([])
        sa = sig[m]
        sa = sa[np.isfinite(sa)]
        med_s = float(np.median(sa)) if sa.size else float('nan')
        min_s = float(np.min(sa)) if sa.size else float('nan')
        max_s = float(np.max(sa)) if sa.size else float('nan')
        min_off = float(np.min(dta)) if dta.size else float('nan')
        med_off = float(np.median(dta)) if dta.size else float('nan')
        lines.append(
            f'  {int(a):3d}  {int(np.sum(m)):5d}  {med_s:10.4g}  {min_s:8.4g}  {max_s:8.4g}  '
            f'{min_off:13.6g}  {med_off:15.6g}'
        )

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('=== Spurious summary (canonical, lowest-flux a) ===')
    print(f'Wrote {txt_path}')
    if np.any(ok_dt):
        print(
            f'  Nearest catalog start: min |Δt| = {global_min_dt:.4g} s '
            f'(median {float(np.median(dt_near[ok_dt])):.4g} s; '
            f'{len(sp)} spurious; cross-match window {win:g} s)'
        )
    if len(s_ok):
        print(f'  SNR σ: median {float(np.median(s_ok)):.4g}, max {float(np.max(s_ok)):.4g}')

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    fig.suptitle(
        f'Spurious triggers (canonical {canon}) — offsets vs catalog starts, SNR, by algorithm',
        fontsize=11,
    )
    ax00, ax01, ax10, ax11 = axes.ravel()

    if np.any(ok_dt):
        d = dt_near[ok_dt]
        hi = max(float(np.max(d)), win * 2.0, 1.0)
        d_min = float(np.min(d))
        lo_edge = d_min if d_min > 0 else (
            float(np.min(d[d > 0])) if np.any(d > 0) else 1e-9
        )
        lo_edge = max(lo_edge, 1e-12)
        if lo_edge >= hi:
            hi = lo_edge * 1.01
        n_bins = min(50, max(15, int(np.sqrt(len(d))) * 3))
        bins = np.logspace(np.log10(lo_edge), np.log10(hi), n_bins)
        ax00.hist(d, bins=bins, color='steelblue', edgecolor='black', alpha=0.85)
        ax00.set_xscale('log')
        ax00.axvline(win, color='C3', linestyle='--', linewidth=1.5, label=f'Cross-match |Δt| = {win:g} s')
        ax00.set_xlabel('min |t_spurious − t_event_start| (s)')
        ax00.set_ylabel('Count')
        ax00.set_title('Nearest injected catalog start (absolute)')
        ax00.legend(loc='upper right', fontsize=8)
    else:
        ax00.set_axis_off()

    s_plot = sig[np.isfinite(sig) & (sig > 0)]
    if len(s_plot) >= 1:
        lo, hi = float(np.min(s_plot)), float(np.max(s_plot))
        if lo >= hi:
            lo, hi = lo * 0.99, hi * 1.01
        bins_s = np.logspace(np.log10(max(lo, 1e-6)), np.log10(hi), min(40, max(12, len(s_plot) * 2)))
        ax01.hist(s_plot, bins=bins_s, color='darkseagreen', edgecolor='black', alpha=0.85)
        ax01.set_xscale('log')
        ax01.xaxis.set_major_locator(LogLocator(base=10))
        ax01.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax01.xaxis.set_major_formatter(FuncFormatter(_plain_sigma_x_tick))
        ax01.tick_params(axis='x', which='major', bottom=True, top=False, length=5)
        ax01.tick_params(axis='x', which='minor', bottom=True, top=False, length=3)
        ax01.set_xlabel('σ (merged significance)')
        ax01.set_ylabel('Count')
        ax01.set_title('SNR distribution (spurious)')
    else:
        ax01.text(0.5, 0.5, 'No σ>0', ha='center', va='center', transform=ax01.transAxes)
        ax01.set_axis_off()

    counts = [int(np.sum(alg == a)) for a in uniq_alg]
    ax10.bar([str(int(a)) for a in uniq_alg], counts, color='coral', edgecolor='black')
    ax10.set_xlabel('Algorithm index')
    ax10.set_ylabel('Count')
    ax10.set_title('Spurious count by firing algorithm')

    med_by_alg = [float(np.median(sig[alg == a])) for a in uniq_alg]
    ax11.bar([str(int(a)) for a in uniq_alg], med_by_alg, color='mediumpurple', edgecolor='black')
    ax11.set_xlabel('Algorithm index')
    ax11.set_ylabel('Median σ')
    ax11.set_title('Median SNR by algorithm (spurious)')
    for i, (a, c) in enumerate(zip(uniq_alg, counts)):
        if c > 0:
            ax10.text(i, c, str(c), ha='center', va='bottom', fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f'Saved figure: {fig_path}\n')


def _load_triggers_npy(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        return np.empty((0, 3), dtype=np.float64)
    a = np.load(path)
    a = np.asarray(a)
    if a.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] != 3:
        raise ValueError(f'Expected (n,3) in {path}, got {a.shape}')
    return a


def binned_match_fraction(
    x: np.ndarray,
    matched: np.ndarray,
    n_bins: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ok = np.isfinite(x) & (x > 0)
    xv = x[ok]
    mv = matched[ok].astype(np.float64)
    if len(xv) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    lo, hi = float(np.min(xv)), float(np.max(xv))
    if lo <= 0 or not np.isfinite(lo + hi):
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    frac, err, counts = [], [], []
    for i in range(n_bins):
        if i < n_bins - 1:
            msk = (xv >= edges[i]) & (xv < edges[i + 1])
        else:
            msk = (xv >= edges[i]) & (xv <= edges[i + 1])
        n = int(np.sum(msk))
        counts.append(n)
        if n == 0:
            frac.append(np.nan)
            err.append(np.nan)
            continue
        p = float(np.mean(mv[msk]))
        frac.append(p)
        err.append(np.sqrt(max(p * (1.0 - p) / n, 0.0)))
    return (
        np.asarray(centers),
        np.asarray(frac),
        np.asarray(err),
        np.asarray(counts, dtype=np.int64),
    )


def binned_recovery_on_fixed_edges(
    x: np.ndarray,
    matched: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Binomial recovery efficiency (mean matched flag) per flux bin.
    Uses positive finite x only; values are clipped into [edges[0], edges[-1]]
    for assignment so the full catalog contributes to the end bins.
    """
    ok = np.isfinite(x) & (x > 0)
    xv = np.clip(x[ok], float(edges[0]), float(edges[-1]))
    mv = matched[ok].astype(np.float64)
    n_bins = len(edges) - 1
    if len(xv) < 1:
        return np.array([]), np.array([]), np.array([]), np.array([])
    centers = np.sqrt(edges[:-1] * edges[1:])
    frac, err, counts = [], [], []
    for i in range(n_bins):
        if i < n_bins - 1:
            msk = (xv >= edges[i]) & (xv < edges[i + 1])
        else:
            msk = (xv >= edges[i]) & (xv <= edges[i + 1])
        n = int(np.sum(msk))
        counts.append(n)
        if n == 0:
            frac.append(np.nan)
            err.append(np.nan)
            continue
        p = float(np.mean(mv[msk]))
        frac.append(p)
        err.append(np.sqrt(max(p * (1.0 - p) / n, 0.0)))
    return (
        np.asarray(centers),
        np.asarray(frac),
        np.asarray(err),
        np.asarray(counts, dtype=np.int64),
    )


_QA_REQUIRED_COLS = [
    COL_START,
    COL_PEAK_START,
    COL_MT,
    COL_MA,
    COL_MS,
    COL_FLUX_80_2000,
    COL_FLUX_10_10000,
    COL_FLUX_ENERG,
]


def catalog_vectors_from_tsv_data(
    data: dict[str, list[str]], path_for_errors: str = ''
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build flux / match arrays from already-loaded events-matched row dict.
    Returns (flux_10_10000, flux_80, matched, sig, flux_energy, n_rows).
    """
    ctx = path_for_errors or 'events-matched'
    for c in _QA_REQUIRED_COLS:
        if c not in data:
            raise KeyError(f'Missing column {c!r} in {ctx}')

    t_trig = col_float(data, COL_MT)
    alg = col_float(data, COL_MA)
    sig = col_float(data, COL_MS)
    flux_80 = col_float(data, COL_FLUX_80_2000)
    flux_10_10000 = col_float(data, COL_FLUX_10_10000)
    flux_energy = col_float(data, COL_FLUX_ENERG)

    n = len(t_trig)
    if n == 0:
        raise ValueError('No data rows in events-matched.csv')
    matched = np.array(
        [is_matched_row(t_trig[i], alg[i], sig[i]) for i in range(n)], dtype=bool
    )
    return flux_10_10000, flux_80, matched, sig, flux_energy, n


def load_catalog_qa_vectors(path_matched: str) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    _, data = load_events_matched_tsv(path_matched)
    return catalog_vectors_from_tsv_data(data, path_matched)


def find_supplementary_flux_csv(acs_root: str, dataset_name: str) -> str | None:
    """``.../<dataset>/flux-<band><suffix>.csv`` next to ``events.csv``."""
    events_path = trigger_outputs.find_events_csv(acs_root, dataset_name)
    if not events_path:
        return None
    cand = os.path.join(os.path.dirname(events_path), f'{dataset_name}.csv')
    return cand if os.path.isfile(cand) else None


def load_supplementary_flux_table(path: str) -> dict[str, tuple[float, float]]:
    """
    Map Event -> (10–1000 keV photon flux, 50–300 keV photon flux).
    Missing numeric entries become NaN.
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        rows = list(reader)
    if len(rows) < 2:
        return {}
    header = [h.strip() for h in rows[0]]
    try:
        i_ev = header.index(COL_EVENT)
        i_1k = header.index(COL_FLUX_SUPP_10_1000)
        i_50 = header.index(COL_FLUX_SUPP_50_300)
    except ValueError as e:
        raise ValueError(
            f'Expected columns {COL_EVENT!r}, {COL_FLUX_SUPP_10_1000!r}, '
            f'{COL_FLUX_SUPP_50_300!r} in {path}; header={header!r}'
        ) from e

    out: dict[str, tuple[float, float]] = {}
    for row in rows[1:]:
        if not row or all(c.strip() == '' for c in row):
            continue
        def _cell(i: int) -> str:
            return row[i].strip() if i < len(row) else ''

        ev = _cell(i_ev)
        if not ev:
            continue

        def _f(i: int) -> float:
            s = _cell(i)
            if not s:
                return float('nan')
            try:
                return float(s)
            except ValueError:
                return float('nan')

        out[ev] = (_f(i_1k), _f(i_50))
    return out


def supplementary_flux_vectors(
    data: dict[str, list[str]],
    path_supp: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row arrays aligned to ``data`` (same order as ``events-matched``)."""
    if COL_EVENT not in data:
        raise KeyError(f'Missing {COL_EVENT!r} in events-matched data')
    n = len(data[COL_EVENT])
    nan2 = (np.nan, np.nan)
    if not path_supp or not os.path.isfile(path_supp):
        empty = np.full(n, np.nan, dtype=np.float64)
        return empty.copy(), empty.copy()

    table = load_supplementary_flux_table(path_supp)
    f1k = np.empty(n, dtype=np.float64)
    f50 = np.empty(n, dtype=np.float64)
    for i, ev in enumerate(data[COL_EVENT]):
        t = table.get(ev.strip(), nan2)
        f1k[i] = t[0]
        f50[i] = t[1]
    return f1k, f50


def load_supplementary_flux_for_matched(path_matched: str) -> tuple[np.ndarray, np.ndarray]:
    _, data = load_events_matched_tsv(path_matched)
    name = os.path.basename(os.path.dirname(os.path.abspath(path_matched)))
    p = find_supplementary_flux_csv(ACS_ROOT, name)
    return supplementary_flux_vectors(data, p)


def load_gbm_64ms_reference() -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load ``gbm_64ms.txt`` (whitespace-separated flux, efficiency per line).
    Returns ``(flux_ph_cm2_s, recovery)`` or None if missing / empty / unparsable.
    """
    path = os.path.join(_SCRIPT_DIR, GBM_64MS_FILE)
    if not os.path.isfile(path):
        return None
    xs: list[float] = []
    ys: list[float] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                continue
    if not xs:
        return None
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _plot_gbm_64ms_on_ax(ax: plt.Axes, **plot_kw) -> bool:
    """Draw GBM reference on an existing log-x recovery-efficiency axis; return True if drawn."""
    gbm = load_gbm_64ms_reference()
    if gbm is None:
        return False
    gx, gy = gbm
    m = np.isfinite(gx) & np.isfinite(gy) & (gx > 0)
    if not np.any(m):
        return False
    gx, gy = gx[m], gy[m]
    n = len(gx)
    markevery = max(1, n // 15)
    defaults = dict(
        color='crimson',
        linewidth=2.0,
        marker='s',
        markersize=4,
        markevery=markevery,
        label=GBM_OVERLAY_LABEL,
        zorder=6,
        clip_on=True,
        alpha=0.92,
    )
    defaults.update(plot_kw)
    ax.plot(gx, gy, '-', **defaults)
    return True


def _hist_photon_flux_log_axis(
    ax, values: np.ndarray, n_bins: int, **hist_kw
) -> bool:
    """Histogram of flux in physical units; caller sets log x-scale and tick format."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 1:
        return False
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo >= hi:
        lo, hi = lo * 0.999, hi * 1.001
    bins = np.logspace(np.log10(lo), np.log10(hi), max(3, int(n_bins)))
    ax.hist(v, bins=bins, density=True, **hist_kw)
    return True


def _style_log_axis(ax, axis: str) -> None:
    """axis is 'x' or 'y'; powers-of-ten tick labels (e.g. 10¹, 10²)."""
    if axis == 'x':
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
    else:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())


def _style_log_flux_xaxis(ax) -> None:
    _style_log_axis(ax, 'x')


def plot_flux_summary_figure(
    flux: np.ndarray,
    matched: np.ndarray,
    sig: np.ndarray,
    flux_energy: np.ndarray,
    flux_label: str,
    energy_label: str,
    suptitle: str,
    out_path: str,
    n_events: int,
    n_spurious: int,
    *,
    overlay_gbm_match_rate: bool = False,
) -> None:
    """Photon flux panels + σ vs photon flux + σ vs energy flux (spurious in footer)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(suptitle, fontsize=11)

    ok = np.isfinite(flux) & (flux > 0)
    m_ok = matched & ok
    u_ok = (~matched) & ok

    ok_e = np.isfinite(flux_energy) & (flux_energy > 0)
    sig_ok = np.isfinite(sig) & (sig > 0)
    ms_flux = matched & ok & sig_ok
    ms_energy = matched & ok_e & sig_ok

    a00, a01, a10, a11 = axes.ravel()

    log_bins = min(28, max(12, n_events // 4))

    if np.any(ok):
        if np.any(m_ok):
            _hist_photon_flux_log_axis(
                a00,
                flux[m_ok],
                log_bins,
                alpha=0.65,
                color='steelblue',
                label=f'Matched (n={int(np.sum(m_ok))})',
                edgecolor='black',
                linewidth=0.4,
            )
        if np.any(u_ok):
            _hist_photon_flux_log_axis(
                a00,
                flux[u_ok],
                log_bins,
                alpha=0.65,
                color='coral',
                label=f'Unmatched (n={int(np.sum(u_ok))})',
                edgecolor='black',
                linewidth=0.4,
            )
        if len(a00.patches) == 0 and len(a00.lines) == 0:
            a00.text(0.5, 0.5, 'No positive flux', ha='center', va='center', transform=a00.transAxes)
        else:
            _style_log_flux_xaxis(a00)
            _style_log_axis(a00, 'y')
            a00.set_ylabel('Density')
            a00.set_title('Flux distribution: matched vs unmatched')
            a00.set_xlabel(flux_label)
            a00.legend(loc='upper right', fontsize=9)

        centers, frac, err, cnts = binned_match_fraction(flux, matched, n_bins=14)
        if len(centers) > 0:
            valid = np.isfinite(frac) & (cnts > 0)
            sim_label = 'Simulated ACS' if overlay_gbm_match_rate else None
            a01.errorbar(
                centers[valid],
                frac[valid],
                yerr=err[valid],
                fmt='o-',
                color='darkgreen',
                capsize=3,
                markersize=5,
                alpha=0.85,
                label=sim_label,
            )
            if overlay_gbm_match_rate:
                _plot_gbm_64ms_on_ax(a01, markersize=3)
                a01.legend(loc='best', fontsize=8, framealpha=0.92)
            _style_log_flux_xaxis(a01)
            a01.set_ylim(-0.05, 1.05)
            a01.set_xlabel(flux_label)
            a01.set_ylabel('Fraction matched in bin')
            a01.set_title('Match rate vs brightness (binomial errors)')
            a01.grid(True, alpha=0.3)
        else:
            a01.text(0.5, 0.5, 'Insufficient flux spread', ha='center', va='center', transform=a01.transAxes)
            a01.set_axis_off()
    else:
        a00.text(0.5, 0.5, 'No valid photon flux', ha='center', va='center', transform=a00.transAxes)
        a00.set_axis_off()
        a01.text(0.5, 0.5, 'No valid photon flux', ha='center', va='center', transform=a01.transAxes)
        a01.set_axis_off()

    if np.any(ms_flux):
        a10.scatter(
            flux[ms_flux],
            sig[ms_flux],
            alpha=0.65,
            s=36,
            c='darkgreen',
            edgecolors='k',
            linewidths=0.3,
        )
        _style_log_flux_xaxis(a10)
        _style_log_axis(a10, 'y')
        a10.set_xlabel(flux_label)
        a10.set_ylabel('Trigger significance (σ)')
        a10.set_title('Significance vs photon flux (matched)')
    else:
        a10.text(
            0.5,
            0.5,
            'No matched rows with σ>0 and valid flux',
            ha='center',
            va='center',
            transform=a10.transAxes,
        )
        a10.set_axis_off()

    if np.any(ms_energy):
        a11.scatter(
            flux_energy[ms_energy],
            sig[ms_energy],
            alpha=0.65,
            s=36,
            c='purple',
            edgecolors='k',
            linewidths=0.3,
        )
        _style_log_flux_xaxis(a11)
        _style_log_axis(a11, 'y')
        a11.set_xlabel(energy_label)
        a11.set_ylabel('Trigger significance (σ)')
        a11.set_title('Significance vs energy brightness (matched)')
    else:
        a11.text(
            0.5,
            0.5,
            'No matched rows with σ>0 and valid energy flux',
            ha='center',
            va='center',
            transform=a11.transAxes,
        )
        a11.set_axis_off()

    fig.text(
        0.5,
        0.01,
        f'Spurious triggers (no catalog event within match window): {n_spurious} — not tied to event flux here.',
        ha='center',
        fontsize=8,
        style='italic',
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved figure: {out_path}')


def _aggregate_global_for_entries(
    entries: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """
    Stack QA vectors from each dataset folder.

    Returns ``(matched, flux_10_10000, flux_10_1000, flux_50_300, flux_80_2000, n_catalog)``
    or None if ``entries`` is empty.
    """
    if not entries:
        return None
    flux_10k: list[np.ndarray] = []
    flux_1k: list[np.ndarray] = []
    flux_50: list[np.ndarray] = []
    flux_8: list[np.ndarray] = []
    matched_parts: list[np.ndarray] = []
    n_catalog = 0
    for _name, ddir in entries:
        p = os.path.join(ddir, EVENTS_MATCHED)
        fw, f8, m, _sg, _fe, n = load_catalog_qa_vectors(p)
        fk, fs = load_supplementary_flux_for_matched(p)
        flux_10k.append(fw)
        flux_1k.append(fk)
        flux_50.append(fs)
        flux_8.append(f8)
        matched_parts.append(m)
        n_catalog += n
    return (
        np.concatenate(matched_parts),
        np.concatenate(flux_10k),
        np.concatenate(flux_1k),
        np.concatenate(flux_50),
        np.concatenate(flux_8),
        n_catalog,
    )


def _aggregate_global_for_run(
    run_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Same as ``_aggregate_global_for_entries`` for every dataset under ``run_dir``."""
    entries: list[tuple[str, str]] = []
    for name in sorted(os.listdir(run_dir)):
        ddir = os.path.join(run_dir, name)
        if not os.path.isdir(ddir):
            continue
        p = os.path.join(ddir, EVENTS_MATCHED)
        if os.path.isfile(p):
            entries.append((name, ddir))
    return _aggregate_global_for_entries(entries)


def plot_global_recovery_efficiency_for_run(
    run_dir: str,
    flux: np.ndarray,
    matched: np.ndarray,
    n_catalog_total: int,
    *,
    fig_filename: str,
    band_label: str,
    xlabel: str,
    edges_lo: float,
    edges_hi: float,
    overlay_gbm: bool = False,
) -> None:
    """Single-panel recovery efficiency vs photon flux with fixed log bins for one band."""
    edges = np.logspace(
        np.log10(edges_lo),
        np.log10(edges_hi),
        GLOBAL_FLUX_N_BINS + 1,
    )
    centers, frac, err, cnts = binned_recovery_on_fixed_edges(flux, matched, edges)
    valid = np.isfinite(frac) & (cnts > 0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle(
        f'Global recovery efficiency vs peak photon flux ({band_label})',
        fontsize=12,
    )
    if np.any(valid):
        sim_lbl = 'Simulated ACS' if overlay_gbm else None
        ax.errorbar(
            centers[valid],
            frac[valid],
            yerr=err[valid],
            fmt='o-',
            color='darkgreen',
            capsize=3,
            markersize=5,
            alpha=0.9,
            label=sim_lbl,
        )
    if overlay_gbm:
        _plot_gbm_64ms_on_ax(ax, markersize=4)
        h, _ = ax.get_legend_handles_labels()
        if h:
            ax.legend(loc='best', fontsize=9, framealpha=0.92)
    _style_log_flux_xaxis(ax)
    ax.set_xlim(edges_lo * 0.92, edges_hi * 1.08)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Recovery efficiency (fraction of catalog events matched)')
    ax.set_title(
        f'All datasets combined (N={n_catalog_total} catalog rows with valid flux > 0)'
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(run_dir, fig_filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved figure: {out}')


def _list_run_directories(outputs_dir: str) -> list[str]:
    """Sorted absolute paths to ``outputs/run_*`` directories."""
    if not os.path.isdir(outputs_dir):
        return []
    out: list[str] = []
    for name in sorted(os.listdir(outputs_dir)):
        if not name.startswith('run_'):
            continue
        p = os.path.join(outputs_dir, name)
        if os.path.isdir(p):
            out.append(os.path.abspath(p))
    return out


def plot_global_recovery_efficiency_across_runs(script_dir: str) -> None:
    """
    Under ``outputs/``, write one comparison PNG per energy band in
    ``_GLOBAL_RECOVERY_SPECS`` (same bin edges as the per-run global plots).
    """
    outputs_dir = trigger_outputs.outputs_root(script_dir)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4'])

    run_paths = _list_run_directories(outputs_dir)
    if not run_paths:
        print(f'No outputs/run_* under {outputs_dir}; skipping cross-run global figures.')
        return

    skipped_runs: list[tuple[str, list[str]]] = []
    complete_run_paths: list[str] = []
    for rp in run_paths:
        if trigger_outputs.is_run_complete_for_qa(
            rp, ACS_ROOT, EVENTS_MATCHED, complete=False
        ):
            complete_run_paths.append(rp)
        else:
            miss = trigger_outputs.missing_flux_outputs_for_qa(
                rp, ACS_ROOT, EVENTS_MATCHED, complete=False
            )
            skipped_runs.append((os.path.basename(rp), miss))

    if skipped_runs:
        print(
            'Cross-run global figures: skipping incomplete runs (missing '
            f'expected {EVENTS_MATCHED}):'
        )
        for label, miss in skipped_runs:
            print(f'  {label}: missing {", ".join(miss)}')
        print()

    if not complete_run_paths:
        print(
            f'No complete runs under {outputs_dir} for cross-run global figures '
            f'(need all in-scope ACS flux datasets with {EVENTS_MATCHED}; fast mode).'
        )
        return

    for fname_run, fname_across, band_label, xlabel, elo, ehi, flux_i in _GLOBAL_RECOVERY_SPECS:
        edges = np.logspace(
            np.log10(elo),
            np.log10(ehi),
            GLOBAL_FLUX_N_BINS + 1,
        )
        series: list[
            tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
        ] = []
        for rp in complete_run_paths:
            agg = _aggregate_global_for_run(rp)
            if agg is None:
                continue
            matched_all = agg[0]
            flux_all = agg[flux_i]
            centers, frac, err, cnts = binned_recovery_on_fixed_edges(
                flux_all, matched_all, edges
            )
            valid = np.isfinite(frac) & (cnts > 0)
            if not np.any(valid):
                continue
            label = os.path.basename(rp)
            n_cat = agg[5]
            series.append((label, centers, frac, err, valid, n_cat))

        if not series:
            print(
                f'No runs with usable {band_label} flux for cross-run plot; '
                f'skipping {fname_across!r}.'
            )
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(
            f'Global recovery efficiency vs peak photon flux ({band_label}) — '
            f'comparison across runs',
            fontsize=12,
        )
        for i, (label, centers, frac, err, valid, n_cat) in enumerate(series):
            c = colors[i % len(colors)]
            ax.errorbar(
                centers[valid],
                frac[valid],
                yerr=err[valid],
                fmt='o-',
                color=c,
                capsize=2,
                markersize=4,
                alpha=0.88,
                label=f'{label} (N={n_cat})',
            )
        if fname_across == FIG_GLOBAL_RECOVERY_ACROSS_RUNS_50_300:
            _plot_gbm_64ms_on_ax(ax, markersize=4)
        _style_log_flux_xaxis(ax)
        ax.set_xlim(elo * 0.92, ehi * 1.08)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Recovery efficiency (fraction of catalog events matched)')
        ax.set_title(
            f'{len(series)} run(s) with match data — same bin edges as per-run '
            f'{fname_run!r}'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=7, framealpha=0.92)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out = os.path.join(outputs_dir, fname_across)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f'Saved figure: {out}')


def plot_merged_group_population_summaries(
    run_dir: str,
    group_key: str,
    member_labels: list[str],
    paths_matched: list[str],
) -> None:
    """Merge ``a``/``b`` (or more) catalogs and write flux-summary PNGs under run_dir."""
    flux_w = []
    flux_8 = []
    flux_1k = []
    flux_50 = []
    matched_a = []
    sig_a = []
    flux_e = []
    # Spurious list is identical for every flux replica; count once (first member).
    n_spur_total = 0
    if paths_matched:
        n_spur_total = len(
            load_spurious_array(
                os.path.join(os.path.dirname(paths_matched[0]), TRIGGERS_SPURIOUS)
            )
        )
    for lbl, p in zip(member_labels, paths_matched):
        fw, f8, m, sg, fe, _n = load_catalog_qa_vectors(p)
        fk, fs = load_supplementary_flux_for_matched(p)
        flux_w.append(fw)
        flux_8.append(f8)
        flux_1k.append(fk)
        flux_50.append(fs)
        matched_a.append(m)
        sig_a.append(sg)
        flux_e.append(fe)

    flux_10_10000 = np.concatenate(flux_w)
    flux_80 = np.concatenate(flux_8)
    flux_10_1000 = np.concatenate(flux_1k)
    flux_50_300 = np.concatenate(flux_50)
    matched = np.concatenate(matched_a)
    sig = np.concatenate(sig_a)
    flux_energy = np.concatenate(flux_e)
    n = len(matched)
    tok = _safe_group_filename_token(group_key)
    energy_label_plot = 'Peak energy flux (erg/cm²/s), 10–10000 keV'
    members_str = ', '.join(member_labels)

    plot_flux_summary_figure(
        flux_10_10000,
        matched,
        sig,
        flux_energy,
        'Peak photon flux (ph/cm²/s), 10–10000 keV',
        energy_label_plot,
        f'Population (merged {members_str}) — 10–10000 keV photon flux\n'
        f'Group {group_key!r} — spurious triggers (shared count): {n_spur_total}',
        os.path.join(run_dir, f'trigger-checks-population-{tok}-10-10000keV.png'),
        n,
        n_spur_total,
    )
    plot_flux_summary_figure(
        flux_10_1000,
        matched,
        sig,
        flux_energy,
        'Peak photon flux (ph/cm²/s), 10–1000 keV',
        energy_label_plot,
        f'Population (merged {members_str}) — 10–1000 keV photon flux (companion CSV)\n'
        f'Group {group_key!r} — spurious triggers (shared count): {n_spur_total}',
        os.path.join(run_dir, f'trigger-checks-population-{tok}-10-1000keV.png'),
        n,
        n_spur_total,
    )
    plot_flux_summary_figure(
        flux_50_300,
        matched,
        sig,
        flux_energy,
        'Peak photon flux (ph/cm²/s), 50–300 keV',
        energy_label_plot,
        f'Population (merged {members_str}) — 50–300 keV photon flux (companion CSV)\n'
        f'Group {group_key!r} — spurious triggers (shared count): {n_spur_total}',
        os.path.join(run_dir, f'trigger-checks-population-{tok}-50-300keV.png'),
        n,
        n_spur_total,
        overlay_gbm_match_rate=True,
    )
    plot_flux_summary_figure(
        flux_80,
        matched,
        sig,
        flux_energy,
        'Peak photon flux (ph/cm²/s), 80–2000 keV',
        energy_label_plot,
        f'Population (merged {members_str}) — 80–2000 keV photon flux\n'
        f'Group {group_key!r} — spurious triggers (shared count): {n_spur_total}',
        os.path.join(run_dir, f'trigger-checks-population-{tok}-80-2000keV.png'),
        n,
        n_spur_total,
    )


def _hist_sig(ax, s, fine: bool = False, **kwargs):
    s = np.asarray(s, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    if len(s) < 2:
        return
    lo, hi = float(np.min(s)), float(np.max(s))
    if lo >= hi:
        lo, hi = lo * 0.99, hi * 1.01
    if fine:
        n_bins = min(90, max(40, len(s) * 4))
    else:
        n_bins = min(28, max(10, len(s) // 2))
    bins = np.logspace(np.log10(lo), np.log10(hi), n_bins)
    ax.hist(s, bins=bins, density=True, histtype='stepfilled', linewidth=1.2, **kwargs)


def qa_one_dataset(
    dataset_dir: str,
    label: str,
    *,
    spur_canonical: str | None = None,
) -> None:
    path_matched = os.path.join(dataset_dir, EVENTS_MATCHED)
    if not os.path.isfile(path_matched):
        raise FileNotFoundError(f'Missing {path_matched}; run trigger_match.py first.')

    if spur_canonical is None:
        exp_for_spur = trigger_outputs.expected_acs_dataset_names(ACS_ROOT)
        spur_canonical = (
            trigger_outputs.canonical_spurious_source_dataset_name(ACS_ROOT)
            if exp_for_spur
            else None
        )
    skip_spurious_only_figures = spur_canonical is not None and label != spur_canonical

    _, data = load_events_matched_tsv(path_matched)
    flux_10_10000, flux_80, matched, sig, flux_energy, n = catalog_vectors_from_tsv_data(
        data, path_matched
    )
    flux_10_1000, flux_50_300 = supplementary_flux_vectors(
        data, find_supplementary_flux_csv(ACS_ROOT, label)
    )
    n_matched = int(np.sum(matched))

    t_start = col_float(data, COL_START)
    t_peak = col_float(data, COL_PEAK_START)
    t_event_ref = t_start + np.nan_to_num(t_peak, nan=0.0)
    t_trig = col_float(data, COL_MT)
    dt_s = t_trig - t_event_ref

    path_raw = os.path.join(dataset_dir, TRIGGERS_RAW)
    path_merged = os.path.join(dataset_dir, TRIGGERS_MERGED)
    path_spurious = os.path.join(dataset_dir, TRIGGERS_SPURIOUS)

    raw = _load_triggers_npy(path_raw)
    merged = _load_triggers_npy(path_merged)
    spurious = load_spurious_array(path_spurious)

    n_raw = len(raw)
    n_merged = len(merged)
    n_spurious = len(spurious)

    sig_raw = raw[:, 2] if n_raw else np.array([])
    sig_merged = merged[:, 2] if n_merged else np.array([])
    sig_spurious = spurious[:, 2] if n_spurious else np.array([])
    sig_matched_events = sig[matched]

    print(f'=== Trigger checks ({label}) ===')
    print(f'Raw triggers: {n_raw}  |  Merged: {n_merged}  |  Spurious: {n_spurious}')
    print(f'Catalog events: {n}  |  Matched: {n_matched}  |  Unmatched: {n - n_matched}')
    if n_matched > 0:
        sm = sig_matched_events[np.isfinite(sig_matched_events)]
        print(
            f'Matched σ — min/med/max: {np.min(sm):.3f} / {np.median(sm):.3f} / {np.max(sm):.3f}'
        )
        dt_m = dt_s[matched]
        print(
            f'Δt (trigger − [start + peak start]) s — med |Δt|: '
            f'{np.median(np.abs(dt_m)):.3f}'
        )

    # ---- Figure 1: overview (counts, significances, match timing, event pie) ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(11, 9))
    fig1.suptitle(
        f'Trigger overview — raw / merged / matched / spurious ({label})',
        fontsize=12,
    )
    ax_a, ax_b, ax_c, ax_d = axes1.ravel()

    labels_c = [
        'Raw triggers',
        'Merged triggers',
        'Catalog events',
        'Matched events',
        'Unmatched events',
        'Spurious triggers',
    ]
    counts_c = [n_raw, n_merged, n, n_matched, n - n_matched, n_spurious]
    colors_c = ['#7f7f7f', '#1f77b4', '#bcbd22', '#2ca02c', '#ff7f0e', '#d62728']
    counts_arr = np.asarray(counts_c, dtype=float)
    # Log x-axis: bar spans [1, 1+count] (count 0 → tiny span so log scale is valid)
    lefts = np.ones(len(counts_c), dtype=float)
    rights = 1.0 + np.maximum(counts_arr, 0.0)
    rights = np.where(counts_arr <= 0, 1.06, rights)
    widths = rights - lefts
    ax_a.barh(labels_c, widths, left=lefts, color=colors_c, edgecolor='black', linewidth=0.5)
    for i, c in enumerate(counts_c):
        ax_a.text(rights[i] * 1.06, i, str(int(c)), va='center', fontsize=8, clip_on=False)
    ax_a.set_xscale('log')
    ax_a.set_xlabel('Count (log scale; bar extent 1 → 1+N)')
    ax_a.set_title('Counts')
    ax_a.set_xlim(0.85, None)
    ax_a.invert_yaxis()

    ax_b.set_title('Trigger significance σ (normalized density)')
    ax_b.set_xlabel('σ')
    ax_b.set_ylabel('Density')

    _hist_sig(ax_b, sig_raw, fine=False, alpha=0.35, color='gray', label=f'Raw (n={n_raw})')
    _hist_sig(ax_b, sig_spurious, fine=False, alpha=0.42, color='C3', label=f'Spurious (n={n_spurious})')
    _hist_sig(ax_b, sig_merged, fine=True, alpha=0.45, color='C0', label=f'Merged (n={n_merged})')
    _hist_sig(ax_b, sig_matched_events, fine=False, alpha=0.55, color='C2', label=f'Matched events (n={n_matched})')
    if len(ax_b.patches) == 0:
        ax_b.text(0.5, 0.5, 'No positive σ to plot', ha='center', va='center', transform=ax_b.transAxes)
    else:
        ax_b.set_xscale('log')
        ax_b.legend(loc='upper right', fontsize=8)
        ax_b.set_xlim(left=max(ax_b.get_xlim()[0], 1e-3))

    if n_matched > 0:
        dt_m = dt_s[matched]
        ax_c.hist(dt_m, bins=min(30, max(10, n_matched // 3)), color='seagreen', edgecolor='black', alpha=0.85)
        ax_c.axvline(0.0, color='gray', linestyle='--', linewidth=1)
        ax_c.set_xlabel('Trigger − (start + peak start) (s)')
        ax_c.set_ylabel('Count')
        ax_c.set_title('Time offset (matched only)')
    else:
        ax_c.text(0.5, 0.5, 'No matched events', ha='center', va='center', transform=ax_c.transAxes)
        ax_c.set_axis_off()

    pie_slices = [
        (n_matched, f'Matched events\n({n_matched})', '#2ca02c'),
        (n - n_matched, f'Unmatched events\n({n - n_matched})', '#ff7f0e'),
        (n_spurious, f'Spurious triggers\n({n_spurious})', '#d62728'),
    ]
    pie_slices = [(s, lab, col) for s, lab, col in pie_slices if s > 0]
    if pie_slices:
        sz, lab, col = zip(*pie_slices)
        ax_d.pie(
            sz,
            labels=lab,
            autopct='%1.1f%%',
            colors=list(col),
            startangle=90,
        )
        ax_d.set_title('Outcomes: catalog match + spurious triggers')
    else:
        ax_d.set_axis_off()

    fig1.tight_layout()
    out1 = os.path.join(dataset_dir, FIG_OVERVIEW)
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f'Saved figure: {out1}')

    run_dir_qa = os.path.dirname(dataset_dir)
    if skip_spurious_only_figures:
        print(
            f'Skip spurious insight / nearest-trigger figures ({label}): '
            f'identical to {spur_canonical!r} (canonical spurious source).'
        )
    else:
        try:
            plot_spurious_insight_figure(dataset_dir, label, run_dir_qa, spurious)
        except (OSError, ValueError, FileNotFoundError) as e:
            print(f'Skip spurious insight figure ({label}): {e}')

        try:
            plot_spurious_nearest_trigger_offset_figure(
                dataset_dir, label, spurious, merged, matched, t_trig
            )
        except (OSError, ValueError, FileNotFoundError, IndexError) as e:
            print(f'Skip spurious nearest-trigger figure ({label}): {e}')

        try:
            plot_spurious_per_detector_lightcurves(
                dataset_dir, label, spurious, ACS_ROOT
            )
        except (OSError, ValueError, FileNotFoundError, IndexError) as e:
            print(f'Skip spurious per-detector lightcurves ({label}): {e}')

    energy_label_plot = 'Peak energy flux (erg/cm²/s), 10–10000 keV'

    plot_flux_summary_figure(
        flux_10_1000,
        matched,
        sig,
        flux_energy,
        'Peak photon flux (ph/cm²/s), 10–1000 keV',
        energy_label_plot,
        'Brightness summary — 10–1000 keV photon flux (companion CSV)\n'
        + f'Spurious triggers: {n_spurious} (no event flux)',
        os.path.join(dataset_dir, FIG_FLUX_10_1000),
        n,
        n_spurious,
    )

    plot_flux_summary_figure(
        flux_50_300,
        matched,
        sig,
        flux_energy,
        'Peak photon flux (ph/cm²/s), 50–300 keV',
        energy_label_plot,
        'Brightness summary — 50–300 keV photon flux (companion CSV)\n'
        + f'Spurious triggers: {n_spurious} (no event flux)',
        os.path.join(dataset_dir, FIG_FLUX_50_300),
        n,
        n_spurious,
    )

    plot_flux_summary_figure(
        flux_80,
        matched,
        sig,
        flux_energy,
        'Peak photon flux (ph/cm²/s), 80–2000 keV',
        energy_label_plot,
        'Brightness summary — 80–2000 keV photon flux\n'
        + f'Spurious triggers: {n_spurious} (no event flux)',
        os.path.join(dataset_dir, FIG_FLUX_80_2000),
        n,
        n_spurious,
    )


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
    spur_canonical = (
        trigger_outputs.canonical_spurious_source_dataset_name(
            ACS_ROOT, dataset_names=suite
        )
        if suite
        else None
    )

    if expected:
        if not suite:
            raise FileNotFoundError(
                f'Fast mode: no flux-1-10 / flux-10-100 folders with events.csv under {ACS_ROOT!r}. '
                'Re-run with ``--complete`` for all flux bands, or add those catalogs.'
            )
        names_loop = suite
        qa_scope = 'all ACS flux datasets (--complete)' if complete else 'fast (flux-1-10, flux-10-100 only)'
        print(
            f'QA scope: {qa_scope} — {len(suite)} dataset(s) under {ACS_ROOT}.\n'
        )
        missing = trigger_outputs.missing_flux_outputs_for_qa(
            run_dir, ACS_ROOT, EVENTS_MATCHED, complete=complete
        )
        if missing:
            raise FileNotFoundError(
                f'Run is incomplete; trigger_qa will not run until every in-scope '
                f'dataset has a folder under this run with {EVENTS_MATCHED}.\n'
                f'  Missing or incomplete: {", ".join(missing)}\n'
                f'Run directory: {run_dir}'
            )
    else:
        print(
            f'Warning: no ACS datasets with events.csv under {ACS_ROOT!r}; '
            'requiring every subfolder of the run to have '
            f'{EVENTS_MATCHED} before QA.\n'
        )
        names_loop = sorted(
            n
            for n in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, n))
        )
        if not names_loop:
            raise FileNotFoundError(
                f'No subdirectories under {run_dir}; nothing to QA.'
            )
        missing = [
            n
            for n in names_loop
            if not os.path.isfile(os.path.join(run_dir, n, EVENTS_MATCHED))
        ]
        if missing:
            raise FileNotFoundError(
                'Run is incomplete (no ACS catalog — every subfolder of the run '
                f'must have '
                f'{EVENTS_MATCHED}).\n'
                f'  Missing: {", ".join(missing)}\n'
                f'Run directory: {run_dir}'
            )

    entries: list[tuple[str, str]] = []
    for name in names_loop:
        dataset_dir = os.path.join(run_dir, name)
        path_matched = os.path.join(dataset_dir, EVENTS_MATCHED)
        qa_one_dataset(dataset_dir, name, spur_canonical=spur_canonical)
        entries.append((name, dataset_dir))
        print()

    summarize_spurious_canonical_run(
        run_dir, ACS_ROOT, suite_names=suite if suite else None
    )

    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, ddir in entries:
        groups[flux_pair_group_key(name)].append((name, ddir))

    for gkey, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        labels = [m[0] for m in members]
        paths_matched = [os.path.join(d, EVENTS_MATCHED) for _, d in members]
        print(f'=== Population merge for group {gkey!r} ({", ".join(labels)}) ===')
        plot_merged_group_population_summaries(run_dir, gkey, labels, paths_matched)
        print()

    agg = _aggregate_global_for_entries(entries)
    assert agg is not None
    matched_all = agg[0]
    n_catalog = agg[5]
    print('=== Global recovery efficiency (all datasets, all bands) ===')
    for fname_run, _fa, band_label, xlabel, elo, ehi, flux_i in _GLOBAL_RECOVERY_SPECS:
        plot_global_recovery_efficiency_for_run(
            run_dir,
            agg[flux_i],
            matched_all,
            n_catalog,
            fig_filename=fname_run,
            band_label=band_label,
            xlabel=xlabel,
            edges_lo=elo,
            edges_hi=ehi,
            overlay_gbm=(fname_run == FIG_GLOBAL_RECOVERY_50_300),
        )
    print()

    print('=== Global recovery efficiency across all outputs/run_* ===')
    plot_global_recovery_efficiency_across_runs(_SCRIPT_DIR)
    print()

    plt.show()


if __name__ == '__main__':
    main()
