import glob
import os
import shutil
import sys

import h5py
import numpy as np

import trigger_outputs


# Base directory of this script (so data paths work regardless of cwd)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parent of all flux-* ACS binned datasets (each subfolder has combined_acs_data.hdf5)
ACS_ROOT = trigger_outputs.default_acs_root(_SCRIPT_DIR)


def iter_acs_dataset_dirs(acs_root):
    """
    Walk top-level folders under acs_root (e.g. flux-10-100a). Each contains
    combined_acs_data.hdf5 directly in that folder. Yield
    (combined_h5_path, output_dir, name) where output_dir is the directory
    containing the HDF5 (shared npy caches go there) and name is the top-level
    folder name (used under outputs/run_*/).
    """
    if not os.path.isdir(acs_root):
        raise FileNotFoundError(f'Missing ACS root: {acs_root}')
    for name in sorted(os.listdir(acs_root)):
        outer = os.path.join(acs_root, name)
        if not os.path.isdir(outer) or name.startswith('.'):
            continue
        combined = os.path.join(outer, 'combined_acs_data.hdf5')
        if os.path.isfile(combined):
            yield combined, outer, name
            continue
        # Legacy nested layout: flux-*/flux-*/combined_acs_data.hdf5
        pattern = os.path.join(outer, '*', 'combined_acs_data.hdf5')
        for nested in sorted(glob.glob(pattern)):
            yield nested, os.path.dirname(nested), name


def _gamma_proton_energy_bin_weights(energy_edges_kev, split_kev):
    """
    Per energy bin, fraction of counts assigned to gamma (E < split_kev) vs
    proton (E >= split_kev). Bins that straddle split_kev are split linearly in
    energy. If the upper edge is non-finite, an effective upper edge is used
    only so that a bin spanning split_kev gets a defined apportionment.
    """
    lo = np.asarray(energy_edges_kev[:-1], dtype=np.float64)
    hi = np.asarray(energy_edges_kev[1:], dtype=np.float64)
    n_e = len(lo)
    w_g = np.zeros(n_e, dtype=np.float64)
    w_p = np.zeros(n_e, dtype=np.float64)
    for j in range(n_e):
        el = float(lo[j])
        eh_raw = hi[j]
        if np.isfinite(eh_raw):
            eh = float(eh_raw)
            if eh <= el:
                continue
            if eh <= split_kev:
                w_g[j] = 1.0
            elif el >= split_kev:
                w_p[j] = 1.0
            else:
                frac = (split_kev - el) / (eh - el)
                w_g[j] = frac
                w_p[j] = 1.0 - frac
        else:
            if el >= split_kev:
                w_p[j] = 1.0
            else:
                eh_eff = split_kev + max(split_kev - el, 1e-6)
                if eh_eff <= el:
                    w_p[j] = 1.0
                else:
                    frac = (split_kev - el) / (eh_eff - el)
                    w_g[j] = frac
                    w_p[j] = 1.0 - frac
    return w_g, w_p


def load_combined_acs_bgo_counts_gamma_proton(combined_path, split_kev=2000.0):
    """
    Load pre-binned ACS counts from combined_acs_data.hdf5 and split each time
    bin into gamma-channel (< split_kev keV) and proton-channel (>= split_kev)
    sums over energy bins (linear apportionment when a bin straddles split_kev).

    Datasets are ACS-A-Z0/Z1, ACS-C-X0/X1, ACS-B-Y0/Y1 with shape
    (n_time_bins, n_energy_bins).
    Energy bin edges are in 'energy bins (keV)' (typically 80, 2000, inf).

    Maps ACS names to the legacy BGO order: b1,b2,x1,x2,y1,y2
    (ACS-A-Z0,Z1 → b1,b2; ACS-C-X0,X1 → x1,x2; ACS-B-Y0,Y1 → y1,y2).

    Trigger logic uses gamma-channel rates only; proton channels are written to
    disk for downstream analysis.
    """
    with h5py.File(combined_path, 'r') as hf:
        time_edges = np.asarray(hf['time bins (s)'][:], dtype=np.float64)
        energy_edges = np.asarray(hf['energy bins (keV)'][:], dtype=np.float64)
        n_e = len(energy_edges) - 1
        n_time = len(time_edges) - 1
        w_g, w_p = _gamma_proton_energy_bin_weights(energy_edges, split_kev)
        mapping = (
            ('ACS-A-Z0', 'b1'),
            ('ACS-A-Z1', 'b2'),
            ('ACS-C-X0', 'x1'),
            ('ACS-C-X1', 'x2'),
            ('ACS-B-Y0', 'y1'),
            ('ACS-B-Y1', 'y2'),
        )
        counts_gamma = {}
        counts_proton = {}
        for acs_name, legacy in mapping:
            block = np.asarray(hf[acs_name], dtype=np.float64)
            if block.shape != (n_time, n_e):
                raise ValueError(
                    f'{acs_name}: expected shape ({n_time}, {n_e}), got {block.shape}'
                )
            counts_gamma[legacy] = np.sum(block * w_g[np.newaxis, :], axis=1)
            counts_proton[legacy] = np.sum(block * w_p[np.newaxis, :], axis=1)
        dt = np.diff(time_edges)
        if not np.allclose(dt, dt[0], rtol=0, atol=1e-6):
            raise ValueError('non-uniform time bin width in combined_acs_data.hdf5')
    time_bins_start = time_edges[:-1]
    time_bins_end = time_edges[1:]
    times = np.column_stack((time_bins_start, time_bins_end))
    return {
        'counts_gamma': counts_gamma,
        'counts_proton': counts_proton,
        'time_edges': time_edges,
        'times': times,
        'energy_edges_kev': energy_edges,
        'bin_width_s': float(dt[0]),
        'split_kev': float(split_kev),
    }


def relative_times_from_absolute(times):
    """
    Return bin start/end times with the first bin start as 0 (seconds relative
    to the start of the series). Also returns the subtracted absolute reference.
    """
    t0 = float(times[0, 0])
    return times - t0, t0


# This file contains a representation of COSI trigger algorithm. 

# It is currently written to be easily understandable and not optimized for 
# speed. Possible optimizations are listed in the comments.

# The algorithm is based on the Fermi-GBM trigger implementation. A number of 
# trigger algorithms are run concurrently over the same data stream. These are
# set by an initialziation file. Such a file should be utilized in the full
# implementation for the flight software, as it allows for ease of updates to
# the trigger algorithms in order to optimize in-flight performance, with
# minimal overhead in actually implementing the changes.

# The COSI trigger algorithm will likely be a two-step proces. The first relies 
# on the BGO shield data. Once a given algorithm achieves a trigger condition,
# the second stage will be run on data gathered from the Germanium detectors
# This second stage will prioritize triggers which have significant Compton 
# events, as those which should be promptly downlinked.






# Filled in main() before running the trigger loop (used by get_binned_ged_data)
data_temporal_resolution = None
data_energy_bins_low = None
data_energy_bins_high = None


def read_data_ini(script_dir: str):
    with open(os.path.join(script_dir, 'data.ini'), 'r') as f:
        dt_res = float(f.readline().strip('\n'))
        data_energy_bins = np.array(f.readline().strip('\n').split())
    return dt_res, data_energy_bins[:-1], data_energy_bins[1:]


# COSI ``iop_trig_thresh_cfg`` (COSI_FSW CTM): per-channel threshold is unsigned with
# 28 bits integer + 4 bits fractional precision; LSB = 1/16 (e.g. 0x259 → 37.5625).
IOP_TRIG_THRESH_SCALE = 16.0
IOP_TRIG_THRESH_DEFAULT_RAW = 72
# Nominal σ matching default raw 72/16 (used when ini omits extended columns).
SIGMA_TRIGGER_THRESHOLD = IOP_TRIG_THRESH_DEFAULT_RAW / IOP_TRIG_THRESH_SCALE
# CTM: exactly 12 BGO channels (ch0..ch11 in iop_trig_thresh_cfg / iop_trig_chan_cfg)
# = 6 shield crystals × 2 energy channels. (The "16" in the ICD is the number of
# trigger *tests*, iop_trig_test_cfg id 0-15, not the channel count.)
N_IOP_TRIG_CHANNELS = 12
# CTM iop_trig_test_cfg field limits.
N_IOP_TRIG_TESTS = 16          # test id 0..15
IOP_TRIG_BIN_SIZE_MAX = 256    # bin_size: u8, max_val 256
IOP_TRIG_PHASE_STEPS_MAX = 256 # phase_steps: u8, max_val 256
# CTM iop_trig_ra_cfg field limits: n is 4 bits with max_val 9 (window = 2^n ≤ 512
# samples); delay is 9 bits with max_val 512 (50 ms ticks).
IOP_TRIG_RA_POWER_MAX = 9
IOP_TRIG_RA_DELAY_MAX = 512


def iop_trig_raw_to_sigma(raw: float) -> float:
    """Convert ``iop_trig_thresh_cfg`` raw code to σ (same as fixed-point value; ICD: ÷ 16)."""
    return float(raw) / IOP_TRIG_THRESH_SCALE


def compute_iop_trig_sigmas(
    counts12: tuple[np.ndarray, ...],
    time_dx: int,
    bg_mean_start_bins: int,
    bg_mean_end_bins: int,
    bin_size_bins: int,
) -> np.ndarray:
    """
    Poisson-style σ per ``iop_trig_chan_cfg`` channel (running-average background window).

    ``bin_size_bins`` is the SNR integration length in ACS bins (``iop_trig`` ``bin_size``):
    signal = sum of that many bins ending at ``time_dx``; background mean per bin is
    scaled by the same length to match the integrated signal.
    """
    sigmas = np.full(N_IOP_TRIG_CHANNELS, -999.0, dtype=np.float64)
    for c in range(N_IOP_TRIG_CHANNELS):
        arr = counts12[c]
        bg = float(
            np.mean(arr[time_dx - bg_mean_start_bins : time_dx - bg_mean_end_bins]) * bin_size_bins
        )
        sc = float(np.sum(arr[time_dx + 1 - bin_size_bins : time_dx + 1]))
        if bg > 0.0 and sc > bg:
            sigmas[c] = (sc - bg) / np.sqrt(bg)
    return sigmas


def parse_algorithms(
    algorithms_path: str, data_temporal_resolution: float
) -> tuple[list[tuple[int, int, int, int, np.ndarray, np.ndarray]], int, int]:
    """
    Load algorithms from a snapshot file (e.g. under outputs/run_*/).

    **Global line** (optional, before algorithm rows)::

        iop_trig_ra_cfg <power_n> <delay_bins>

    COSI ``iop_trig_ra_cfg`` (CTM): running average for GRB trigger uses ``2**power_n``
    samples from the ACS stream (SM 0x31 counter cadence: 50 ms per tick in flight).
    ``delay_bins`` is the delay in those same ticks (shift-register depth before the
    average). Simulation uses ACS bins from ``data.ini`` line 1; when that equals
    50 ms, ``power_n`` and ``delay_bins`` match the ICD 1:1. Background mean is over
    ``[time_dx - (delay_bins + 2**power_n) : time_dx - delay_bins)``.
    If this line is absent, falls back to legacy ``round(17/dt)`` and ``round(4/dt)``.

    Each algorithm non-comment line has **4 + 24** columns (preferred)::

        index  bin_size  n_chan_over_thresh  phase_steps
        iop_trig_thresh_cfg[0..11]   iop_trig_chan_cfg[0..11]

    * ``index`` — ``iop_trig_test_cfg`` ``id`` (test 0–15 in ICD; unique in file).
    * ``bin_size`` — SNR integration length in ACS bins (ICD ``bin_size``, max 256;
      one bin = ``data_temporal_resolution`` s).
    * ``n_chan_over_thresh`` — ICD ``n_chan_over_thresh`` (max 12).
    * ``phase_steps`` — stride in ACS bins between evaluations (ICD max 256): window
      end ``time_dx`` is used only when ``(time_dx - (bin_size - 1)) % phase_steps == 0``.

    **Legacy** 27-token rows without ``phase_steps`` imply ``phase_steps = 1`` (old
    behavior: test every bin).

    * Next 12 integers — ``iop_trig_thresh_cfg`` ch0..ch11 raw codes; σ gate = raw / 16.
    * Next 12 integers 0/1 — ``iop_trig_chan_cfg`` ch0..ch11 enables.

    Returns ``(algorithms, bg_mean_start_bins, bg_mean_end_bins)`` where each row is
    ``(index, bin_size, n_chan_over_thresh, phase_steps, thresh_sigma, chan_enable)``.
    """
    print('Initializing algorithms...')
    with open(algorithms_path, 'r') as f:
        raw_lines = [ln.strip('\n') for ln in f if ln.strip() and not ln.strip().startswith('#')]

    ra_power_n: int | None = None
    ra_delay_bins: int | None = None
    algorithm_lines: list[str] = []
    for ln in raw_lines:
        parts = ln.split()
        if not parts:
            continue
        if parts[0] == 'iop_trig_ra_cfg':
            if len(parts) < 3:
                raise ValueError(
                    f'{algorithms_path}: iop_trig_ra_cfg requires two integers '
                    '(power_n delay_bins)'
                )
            ra_power_n = int(float(parts[1]))
            ra_delay_bins = int(float(parts[2]))
            continue
        algorithm_lines.append(ln)

    if ra_power_n is not None:
        if not (0 <= ra_power_n <= IOP_TRIG_RA_POWER_MAX):
            raise ValueError(
                f'{algorithms_path}: iop_trig_ra_cfg power_n must be in '
                f'[0, {IOP_TRIG_RA_POWER_MAX}] (CTM: 4-bit field, max 9), got {ra_power_n}'
            )
        if ra_delay_bins is None or not (0 <= ra_delay_bins <= IOP_TRIG_RA_DELAY_MAX):
            raise ValueError(
                f'{algorithms_path}: iop_trig_ra_cfg delay_bins must be in '
                f'[0, {IOP_TRIG_RA_DELAY_MAX}] (CTM: 9-bit field), got {ra_delay_bins}'
            )
        window_bins = int(2**ra_power_n)
        bg_mean_end_bins = int(ra_delay_bins)
        bg_mean_start_bins = bg_mean_end_bins + window_bins
        if bg_mean_start_bins <= bg_mean_end_bins:
            raise ValueError(f'{algorithms_path}: iop_trig_ra_cfg produced invalid background slice')
        print(
            f'  iop_trig_ra_cfg: power_n={ra_power_n} (window={window_bins} bins), '
            f'delay_bins={bg_mean_end_bins} → mean slice length {window_bins} bins, '
            f'start offset {bg_mean_start_bins} bins before current bin.'
        )
    else:
        bg_mean_end_bins = int(round(4.0 / float(data_temporal_resolution)))
        bg_mean_start_bins = int(round(17.0 / float(data_temporal_resolution)))
        if bg_mean_start_bins <= bg_mean_end_bins:
            raise ValueError('legacy background slice: need start offset > end offset')
        print(
            f'  iop_trig_ra_cfg absent: legacy background slice '
            f'[{bg_mean_start_bins}, {bg_mean_end_bins}) bins (~17 s / ~4 s at data.ini resolution).'
        )

    indices: list[int] = []
    for algorithm in algorithm_lines:
        parts = algorithm.split()
        if not parts:
            continue
        index = int(float(parts[0]))
        if index in indices:
            print(
                'Ending initialization as there are duplicate algorithm indices ('
                + str(index)
                + ') in the algorithms file.'
            )
            sys.exit(1)
        indices.append(index)

    need_legacy = 3 + 2 * N_IOP_TRIG_CHANNELS
    need_new = 4 + 2 * N_IOP_TRIG_CHANNELS
    algorithms: list[tuple[int, int, int, int, np.ndarray, np.ndarray]] = []
    for algorithm in algorithm_lines:
        parts = algorithm.split()
        if not parts:
            continue
        if not round(float(parts[0]), 14).is_integer():
            print('Skipping algorithm ' + str(parts[0]) + ' because the index is not an integer.')
            continue
        index = int(float(parts[0]))
        if not (0 <= index < N_IOP_TRIG_TESTS):
            print(
                f'Skipping algorithm {index}: iop_trig_test_cfg id must be in '
                f'[0, {N_IOP_TRIG_TESTS - 1}]'
            )
            continue
        if len(parts) < 3:
            print(
                f'Skipping algorithm {index}: need index, bin_size, n_chan_over_thresh '
                f'[and phase_steps], then thresh and chan columns'
            )
            continue
        try:
            bin_size = int(float(parts[1]))
            n_chan_over_thresh = int(float(parts[2]))
        except ValueError:
            print(f'Skipping algorithm {index}: non-integer bin_size or n_chan_over_thresh')
            continue
        if not (1 <= bin_size <= IOP_TRIG_BIN_SIZE_MAX):
            print(
                f'Skipping algorithm {index}: bin_size must be in [1, {IOP_TRIG_BIN_SIZE_MAX}]'
            )
            continue

        if len(parts) == need_new:
            try:
                phase_steps = int(float(parts[3]))
            except ValueError:
                print(f'Skipping algorithm {index}: non-integer phase_steps')
                continue
            thresh_off = 4
        elif len(parts) == need_legacy:
            phase_steps = 1
            thresh_off = 3
        else:
            extra = (
                f' If len={len(parts)} with need={need_new}, you may be missing the phase_steps '
                f'column (expected {need_new} tokens) or have an extra thresh token.'
                if len(parts) == need_new + 1
                else ''
            )
            print(
                f'Skipping algorithm {index}: expected {need_legacy} (legacy) or {need_new} '
                f'tokens (index bin_size n_chan_over_thresh [phase_steps] + '
                f'{N_IOP_TRIG_CHANNELS} thresh + {N_IOP_TRIG_CHANNELS} chan_en), '
                f'got {len(parts)}.{extra}'
            )
            continue

        if not (1 <= phase_steps <= IOP_TRIG_PHASE_STEPS_MAX):
            print(
                f'Skipping algorithm {index}: phase_steps must be in '
                f'[1, {IOP_TRIG_PHASE_STEPS_MAX}]'
            )
            continue

        try:
            thresh_raw = np.array(
                [int(float(parts[thresh_off + i])) for i in range(N_IOP_TRIG_CHANNELS)],
                dtype=np.int64,
            )
            chan_enable = np.array(
                [
                    int(float(parts[thresh_off + N_IOP_TRIG_CHANNELS + i]))
                    for i in range(N_IOP_TRIG_CHANNELS)
                ],
                dtype=np.int8,
            )
        except ValueError:
            print(f'Skipping algorithm {index}: non-integer iop_trig_thresh_cfg or iop_trig_chan_cfg')
            continue

        if np.any(chan_enable < 0) or np.any(chan_enable > 1):
            print(f'Skipping algorithm {index}: iop_trig_chan_cfg entries must be 0 or 1')
            continue

        n_enabled = int(np.sum(chan_enable))
        if n_enabled < 1:
            print(f'Skipping algorithm {index}: at least one channel must be enabled')
            continue
        if not (1 <= n_chan_over_thresh <= N_IOP_TRIG_CHANNELS):
            print(
                f'Skipping algorithm {index}: n_chan_over_thresh must be in '
                f'[1, {N_IOP_TRIG_CHANNELS}], got {n_chan_over_thresh}'
            )
            continue
        if n_chan_over_thresh > n_enabled:
            print(
                f'Skipping algorithm {index}: n_chan_over_thresh ({n_chan_over_thresh}) '
                f'exceeds enabled channel count ({n_enabled})'
            )
            continue

        thresh_sigma = thresh_raw.astype(np.float64) / IOP_TRIG_THRESH_SCALE

        timescale_s = float(bin_size) * float(data_temporal_resolution)
        if timescale_s > 4.0:
            print(
                'Algorithm '
                + str(index)
                + ' has an integration window > 4 s. This overlaps with the existing background '
                'calculation. This should be studied in the future.'
            )

        algorithms.append(
            (index, bin_size, n_chan_over_thresh, phase_steps, thresh_sigma, chan_enable)
        )

    print(
        f'Algorithms initialized ({len(algorithms)} rows, '
        f'iop_trig_thresh_cfg scale = 1/{IOP_TRIG_THRESH_SCALE:g} σ per raw unit).\n'
    )
    return algorithms, bg_mean_start_bins, bg_mean_end_bins





















verbose_flag = True
# verbose_flag = False

print('There is currently no implementation of spectral / energy information')
print('This should probably go in the ini file')
# print('Triggering on a single detector seems to exceed 5.5 sigma far too often for a base Poisson rate. Why could this be?')
print('still have configurable number of detectors for sigma to exceed?')
print('twelve channels, also configurable? reserve some for TGF triggers?')
print('')

# Data is here read in from GRB simulations generated using MEGAlib. The point
# of contact for the GRB-specific simulations is Eliza Neights. In reality,
# this data must be preprocessed on-board and passed to the trigger algorithm.
# Additionally, the data will be streamed, which is not handled here. Instead
# we send the full data set to the trigger algorithm.


def read_tte_file(fn):
    fn = open(fn, 'r')
    column_names = fn.readline().strip('\n').split()
    hit_times = []
    hit_energies = []
    for line in fn:
        line = line.strip('\n').split()
        hit_times.append(float(line[0]))
        hit_energies.append(float(line[1]))
    fn.close()
    return np.column_stack((np.array(hit_times), np.array(hit_energies)))


def get_binned_ged_data(fn, bins):
    ged_hits = read_tte_file(fn)

    unique_times = np.unique(ged_hits[:,0])

    for pha_dx in range(len(data_energy_bins_low)):
        channel_hit_times = ged_hits[:,0][(ged_hits[:,1] >= float(50)) & (ged_hits[:,1] < float(300))]
        channel_hit_counts = np.histogram(channel_hit_times, bins=bins)[0]
        if pha_dx == 0:
            counts_array = channel_hit_counts
        else:
            counts_array = np.column_stack((counts_array, channel_hit_counts))
    return counts_array


# Gamma vs proton split for BGO binned caches (keV). Matches 2 MeV boundary in user spec.
BGO_GAMMA_PROTON_SPLIT_KEV = 2000.0
RECOMPUTE_BGO_CACHE = True


def main():
    global ACS_ROOT, data_temporal_resolution, data_energy_bins_low, data_energy_bins_high

    data_temporal_resolution, data_energy_bins_low, data_energy_bins_high = read_data_ini(_SCRIPT_DIR)

    run_dir = trigger_outputs.new_run_directory(_SCRIPT_DIR)
    scope = trigger_outputs.parse_run_scope_argv(script_dir=_SCRIPT_DIR)
    complete = scope.complete
    ACS_ROOT = scope.acs_root
    shutil.copy2(
        os.path.join(_SCRIPT_DIR, 'algorithms.ini'),
        os.path.join(run_dir, 'algorithms.ini'),
    )
    shutil.copy2(os.path.join(_SCRIPT_DIR, 'data.ini'), os.path.join(run_dir, 'data.ini'))
    mode_note = (
        'Default trigger_acs scope is **fast**: only flux-1-10 and flux-10-100 ``a``/``b`` '
        'folders (pass ``--complete`` on the command line for every ACS flux band).\n'
    )
    with open(os.path.join(run_dir, 'README.txt'), 'w', encoding='utf-8') as rf:
        rf.write(
            'Per-run trigger outputs. algorithms.ini and data.ini are snapshots.\n'
            + mode_note
            + f'ACS root: {ACS_ROOT}\n'
            + 'algorithms.ini: optional line ``iop_trig_ra_cfg power_n delay_bins`` (COSI RA background); '
            'then per row: index  bin_size  n_chan_over_thresh  phase_steps  '
            f'iop_trig_thresh_cfg[0..{N_IOP_TRIG_CHANNELS - 1}]  '
            f'iop_trig_chan_cfg[0..{N_IOP_TRIG_CHANNELS - 1}] (see parse_algorithms).\n'
            'Subfolders (flux-*) hold triggers.npy for each ACS dataset.\n'
            'Shared binned caches live next to combined_acs_data.hdf5 under the ACS root:\n'
            '  bgo_*_binned_counts_gamma_channel.npy, bgo_*_binned_counts_proton_channel.npy, '
            'bin_times.npy, time_reference_s.npy.\n'
        )
    trigger_outputs.write_current_run(_SCRIPT_DIR, run_dir)
    print(f'Run output directory: {run_dir}')
    print(f'ACS root: {ACS_ROOT}')
    print(
        'ACS scope: '
        + ('all flux datasets (--complete)' if complete else 'fast mode (flux-1-10, flux-10-100 only)')
        + '\n'
    )

    algorithms_path = os.path.join(run_dir, 'algorithms.ini')
    algorithms, bg_mean_start_bins, bg_mean_end_bins = parse_algorithms(
        algorithms_path, data_temporal_resolution
    )
    if not algorithms:
        raise ValueError(f'No valid algorithms parsed from {algorithms_path}')

    _acs_runs = list(iter_acs_dataset_dirs(ACS_ROOT))
    if not _acs_runs:
        raise FileNotFoundError(
            f'No combined_acs_data.hdf5 found under any subdirectory of {ACS_ROOT}'
        )

    acs_catalog = trigger_outputs.expected_acs_dataset_names(ACS_ROOT)
    if acs_catalog:
        suite = trigger_outputs.acs_datasets_for_mode(ACS_ROOT, complete=complete)
        if not suite:
            raise FileNotFoundError(
                f'Fast mode: no flux-1-10 / flux-10-100 folders with events.csv under {ACS_ROOT!r}. '
                'Re-run with ``--complete`` to process every ACS flux band, or add those catalogs.'
            )
        allowed = set(suite)
        _acs_runs = [t for t in _acs_runs if t[2] in allowed]
    if not _acs_runs:
        raise FileNotFoundError(
            f'No ACS datasets to process under {ACS_ROOT} after applying scope filter.'
        )

    for COMBINED_ACS_PATH, ACS_OUTPUT_DIR, outer_name in _acs_runs:
        input_sample_folder = ACS_OUTPUT_DIR + os.sep
        print(f'\n--- ACS dataset: {COMBINED_ACS_PATH} ---\n')

        if not os.path.isfile(COMBINED_ACS_PATH):
            raise FileNotFoundError(f'Missing combined ACS file: {COMBINED_ACS_PATH}')

        _gamma_b1 = input_sample_folder + 'bgo_b1_binned_counts_gamma_channel.npy'
        _proton_b1 = input_sample_folder + 'bgo_b1_binned_counts_proton_channel.npy'
        _bin_times = input_sample_folder + 'bin_times.npy'
        if (
            RECOMPUTE_BGO_CACHE
            or not os.path.isfile(_bin_times)
            or not os.path.isfile(_gamma_b1)
            or not os.path.isfile(_proton_b1)
        ):
            print('add deadtime')
            print('test background')
            acs = load_combined_acs_bgo_counts_gamma_proton(
                COMBINED_ACS_PATH, split_kev=BGO_GAMMA_PROTON_SPLIT_KEV
            )
            # HDF5 times are float64; successive diffs can differ from data.ini by ~1e-7 s
            if not np.isclose(
                acs['bin_width_s'], data_temporal_resolution, rtol=1e-6, atol=1e-7
            ):
                raise ValueError(
                    f"data.ini temporal resolution ({data_temporal_resolution}) does not match "
                    f"ACS file bin width ({acs['bin_width_s']})"
                )
            times, time_reference_s = relative_times_from_absolute(acs['times'])
            cg = acs['counts_gamma']
            cp = acs['counts_proton']
            b1_counts_gamma_channel = cg['b1']
            b2_counts_gamma_channel = cg['b2']
            x1_counts_gamma_channel = cg['x1']
            x2_counts_gamma_channel = cg['x2']
            y1_counts_gamma_channel = cg['y1']
            y2_counts_gamma_channel = cg['y2']
            b1_counts_proton_channel = cp['b1']
            b2_counts_proton_channel = cp['b2']
            x1_counts_proton_channel = cp['x1']
            x2_counts_proton_channel = cp['x2']
            y1_counts_proton_channel = cp['y1']
            y2_counts_proton_channel = cp['y2']

            np.save(input_sample_folder + 'bgo_b1_binned_counts_gamma_channel.npy', b1_counts_gamma_channel)
            np.save(input_sample_folder + 'bgo_b2_binned_counts_gamma_channel.npy', b2_counts_gamma_channel)
            np.save(input_sample_folder + 'bgo_x1_binned_counts_gamma_channel.npy', x1_counts_gamma_channel)
            np.save(input_sample_folder + 'bgo_x2_binned_counts_gamma_channel.npy', x2_counts_gamma_channel)
            np.save(input_sample_folder + 'bgo_y1_binned_counts_gamma_channel.npy', y1_counts_gamma_channel)
            np.save(input_sample_folder + 'bgo_y2_binned_counts_gamma_channel.npy', y2_counts_gamma_channel)
            np.save(input_sample_folder + 'bgo_b1_binned_counts_proton_channel.npy', b1_counts_proton_channel)
            np.save(input_sample_folder + 'bgo_b2_binned_counts_proton_channel.npy', b2_counts_proton_channel)
            np.save(input_sample_folder + 'bgo_x1_binned_counts_proton_channel.npy', x1_counts_proton_channel)
            np.save(input_sample_folder + 'bgo_x2_binned_counts_proton_channel.npy', x2_counts_proton_channel)
            np.save(input_sample_folder + 'bgo_y1_binned_counts_proton_channel.npy', y1_counts_proton_channel)
            np.save(input_sample_folder + 'bgo_y2_binned_counts_proton_channel.npy', y2_counts_proton_channel)
            np.save(input_sample_folder + 'bin_times.npy', times)
            np.save(input_sample_folder + 'time_reference_s.npy', np.float64(time_reference_s))
        else:
            b1_counts_gamma_channel = np.load(input_sample_folder + 'bgo_b1_binned_counts_gamma_channel.npy')
            b2_counts_gamma_channel = np.load(input_sample_folder + 'bgo_b2_binned_counts_gamma_channel.npy')
            x1_counts_gamma_channel = np.load(input_sample_folder + 'bgo_x1_binned_counts_gamma_channel.npy')
            x2_counts_gamma_channel = np.load(input_sample_folder + 'bgo_x2_binned_counts_gamma_channel.npy')
            y1_counts_gamma_channel = np.load(input_sample_folder + 'bgo_y1_binned_counts_gamma_channel.npy')
            y2_counts_gamma_channel = np.load(input_sample_folder + 'bgo_y2_binned_counts_gamma_channel.npy')
            b1_counts_proton_channel = np.load(input_sample_folder + 'bgo_b1_binned_counts_proton_channel.npy')
            b2_counts_proton_channel = np.load(input_sample_folder + 'bgo_b2_binned_counts_proton_channel.npy')
            x1_counts_proton_channel = np.load(input_sample_folder + 'bgo_x1_binned_counts_proton_channel.npy')
            x2_counts_proton_channel = np.load(input_sample_folder + 'bgo_x2_binned_counts_proton_channel.npy')
            y1_counts_proton_channel = np.load(input_sample_folder + 'bgo_y1_binned_counts_proton_channel.npy')
            y2_counts_proton_channel = np.load(input_sample_folder + 'bgo_y2_binned_counts_proton_channel.npy')
            times = np.load(input_sample_folder + 'bin_times.npy')
            ref_path = input_sample_folder + 'time_reference_s.npy'
            if os.path.isfile(ref_path):
                time_reference_s = float(np.load(ref_path))
            else:
                # Legacy cache: bin_times.npy may still use absolute MET; convert once
                times, time_reference_s = relative_times_from_absolute(times)
                np.save(ref_path, np.float64(time_reference_s))
                np.save(input_sample_folder + 'bin_times.npy', times)

        print(
            'Time axis: seconds relative to first bin start; '
            f'absolute reference (MET) = {time_reference_s} s. '
            'Trigger times are written on the absolute (events.csv) timebase.'
        )

        # IOP channels ch0..ch11: pairs (gamma, proton) per detector b1,b2,x1,x2,y1,y2
        # (CTM iop_trig_thresh_cfg / iop_trig_chan_cfg define exactly 12 channels).
        counts12 = (
            b1_counts_gamma_channel,
            b1_counts_proton_channel,
            b2_counts_gamma_channel,
            b2_counts_proton_channel,
            x1_counts_gamma_channel,
            x1_counts_proton_channel,
            x2_counts_gamma_channel,
            x2_counts_proton_channel,
            y1_counts_gamma_channel,
            y1_counts_proton_channel,
            y2_counts_gamma_channel,
            y2_counts_proton_channel,
        )

        # The real trigger will run on a stream of data. The end condition must be set in a different manner
        # time_dx will likely need to be replaced, as the trigger will access only some fraction of the most recent data
        # This could be rewritten the emulate this
        time_dx = 0
        bdx = 0
        search_flag = False
        search_warmup_bins = bg_mean_start_bins
        max_sigma = 0
        triggers = []
        while time_dx < len(times):

            # time_start = time_bins_start[time_dx]
            # time_end = time_bins_end[time_dx]

            if not search_flag:
                if bdx>0:
                    if bdx == search_warmup_bins:
                        search_flag = True
                        phase = 0
                    bdx = bdx+1
                if bdx == 0 and b1_counts_gamma_channel[time_dx] > 0:
                    bdx = 1

            if search_flag:
                # Legacy behavior: all-gamma background sum for end-of-search uses the last row's bin_size.
                index_range_bg = int(algorithms[-1][1])
                b1_background_counts = float(
                    np.mean(
                        b1_counts_gamma_channel[
                            time_dx - bg_mean_start_bins : time_dx - bg_mean_end_bins
                        ]
                    )
                    * index_range_bg
                )
                b2_background_counts = float(
                    np.mean(
                        b2_counts_gamma_channel[
                            time_dx - bg_mean_start_bins : time_dx - bg_mean_end_bins
                        ]
                    )
                    * index_range_bg
                )
                x1_background_counts = float(
                    np.mean(
                        x1_counts_gamma_channel[
                            time_dx - bg_mean_start_bins : time_dx - bg_mean_end_bins
                        ]
                    )
                    * index_range_bg
                )
                x2_background_counts = float(
                    np.mean(
                        x2_counts_gamma_channel[
                            time_dx - bg_mean_start_bins : time_dx - bg_mean_end_bins
                        ]
                    )
                    * index_range_bg
                )
                y1_background_counts = float(
                    np.mean(
                        y1_counts_gamma_channel[
                            time_dx - bg_mean_start_bins : time_dx - bg_mean_end_bins
                        ]
                    )
                    * index_range_bg
                )
                y2_background_counts = float(
                    np.mean(
                        y2_counts_gamma_channel[
                            time_dx - bg_mean_start_bins : time_dx - bg_mean_end_bins
                        ]
                    )
                    * index_range_bg
                )

                for algorithm in algorithms:
                    index, bin_size, n_chan_over_thresh, phase_steps, thresh_sigma, chan_enable = (
                        algorithm
                    )
                    bin_size_i = int(bin_size)
                    phase_steps_i = int(phase_steps)
                    if bin_size_i < 1 or phase_steps_i < 1:
                        continue
                    first_end = bin_size_i - 1
                    if time_dx < first_end:
                        continue
                    if (time_dx - first_end) % phase_steps_i != 0:
                        continue

                    sigmas_12 = compute_iop_trig_sigmas(
                        counts12,
                        time_dx,
                        bg_mean_start_bins,
                        bg_mean_end_bins,
                        bin_size_i,
                    )
                    over = (chan_enable == 1) & (sigmas_12 > thresh_sigma)
                    if int(np.sum(over)) < n_chan_over_thresh:
                        continue

                    pool = sigmas_12[chan_enable == 1]
                    if pool.size >= 2:
                        trigger_significance = float(np.partition(pool, pool.size - 2)[pool.size - 2])
                    else:
                        trigger_significance = float(np.max(pool))
                    if trigger_significance > max_sigma:
                        max_sigma = trigger_significance
                    trigger_time_s = times[:, 0][time_dx] + time_reference_s
                    if verbose_flag:
                        print(
                            'Trigger identified at '
                            + '%.2f' % trigger_time_s
                            + ' by algorithm '
                            + str(index)
                            + ' with sigma '
                            + '%.2f' % trigger_significance
                        )
                    triggers.append([trigger_time_s, index, trigger_significance])

                if (
                    b1_background_counts
                    + b2_background_counts
                    + x1_background_counts
                    + x2_background_counts
                    + y1_background_counts
                    + y2_background_counts
                ) == 0:
                    search_flag = False
                    bdx = 0
            time_dx += 1

        triggers = np.array(triggers)
        dataset_run_dir = os.path.join(run_dir, outer_name)
        os.makedirs(dataset_run_dir, exist_ok=True)
        np.save(os.path.join(dataset_run_dir, 'triggers.npy'), triggers)
        print(f'Wrote {len(triggers)} trigger rows to {dataset_run_dir}/triggers.npy')

if __name__ == '__main__':
    main()