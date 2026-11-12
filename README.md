# cosi-triggers

Python emulation and QA of the COSI GRB onboard trigger, which runs on the
binned Anti-Coincidence Shield (ACS / BGO) count rates. The configuration
mirrors the flight-software command set in `COSI_FSW_014_CTM`
(`iop_trig_ra_cfg`, `iop_trig_test_cfg`, `iop_trig_thresh_cfg`,
`iop_trig_chan_cfg`), so a table tuned here can be commanded to the instrument
field-for-field.

The pipeline replays simulated 24-hour ACS lightcurves with injected short GRBs,
records every trigger, cross-matches triggers to the injected catalog, and
produces recovery-efficiency and false-trigger diagnostics.

## Requirements

- Python 3.10 or newer (the code uses `X | None` type syntax; tested with 3.11)
- `numpy`, `h5py`, `matplotlib`

Install into any environment you like, for example:

```bash
# pip
python3 -m pip install numpy h5py matplotlib

# or conda
conda create -n cosi-triggers python=3.11 numpy h5py matplotlib
conda activate cosi-triggers
```

No installation of this package is needed; the scripts are run in place from a
clone of the repository.

## Data

The binned ACS datasets are not in the repository. Download them from the
shared Google Drive folder
<https://drive.google.com/drive/folders/1oOoow-dy0ER-ap2B8gVr6YJ0_Sqa5i03?usp=share_link>
and place (or extract) them at the top level of the clone so that the layout is:

```
cosi-triggers/
├── trigger_acs.py, trigger_match.py, trigger_qa.py, ...
└── sgrbs-scaled-background-binned-acs/
    ├── flux-0.01-0.1a/
    │   ├── combined_acs_data.hdf5      binned ACS counts (50 ms x energy bins) per shield panel
    │   ├── events.csv                  injected GRB catalog (start time, duration, position,
    │   │                               spectrum, peak flux for 10–10000 and 80–2000 keV)
    │   ├── flux-0.01-0.1a.csv          companion peak fluxes for 10–1000 and 50–300 keV
    │   ├── inputs/                     cosipyburst configuration used to build the dataset
    │   └── README.md                   provenance
    ├── flux-0.01-0.1b/
    ├── ...
    └── flux-1000-10000b/
```

There are 12 datasets: six peak-flux bands (0.01–0.1 … 1000–10000 ph cm⁻² s⁻¹
in 80–2000 keV), each with two independent realisations `a` and `b`, 170 bursts
per dataset on the same background day. The background has been scaled by ×1.2
relative to the raw simulation to represent activation build-up.

The data directory must be writable: `trigger_acs.py` writes per-channel
`.npy` caches next to each `combined_acs_data.hdf5` (about 200 MB per dataset).
They are regenerated on every run and can be deleted at any time. The data
folder and these caches are excluded from git via `.gitignore`.

To use data stored elsewhere, pass `--acs-root /path/to/dir` to every stage
(relative paths are resolved against the script directory, not the current
working directory).

## Quick start

From the repository directory:

```bash
python3 trigger_acs.py && python3 trigger_match.py && python3 trigger_qa.py
```

By default this runs in **fast mode**, which only processes the `flux-1-10` and
`flux-10-100` datasets (`a` and `b`), i.e. the bands where the trigger
efficiency turns over. Add `--complete` to every stage to process all 12
datasets:

```bash
python3 trigger_acs.py --complete && python3 trigger_match.py --complete && python3 trigger_qa.py --complete
```

Each dataset covers 24 h at 50 ms resolution (1.7 M bins) and the emulation is a
plain Python loop, so expect several minutes per dataset (roughly half an hour
for fast mode, well over an hour for `--complete`, depending on the machine).
`trigger_acs.py` prints one line per trigger while it runs; set
`verbose_flag = False` near the top of that file to silence it.

## Pipeline stages

1. **`trigger_acs.py`** — the trigger emulation. Reads `algorithms.ini` and
   `data.ini`, creates a new run directory `outputs/run_YYYYMMDD_HHMMSS/`,
   snapshots both ini files into it, records the run path in
   `outputs/_current_run.txt`, and for each dataset writes
   `<run>/<dataset>/triggers.npy`: one row `[time_s, algorithm, significance]`
   per firing of any test (times are seconds from the start of the dataset).
2. **`trigger_match.py`** — merges trigger rows within 5 s into single
   triggers keeping the most significant (`trigger-merged.npy`), attaches the
   nearest merged trigger within 5 s of each injected start to the catalog
   (`events-matched.csv`: standard comma-separated CSV with every catalog event
   and a final `Detected` column of 1/0; undetected events have `nan` in the
   three matched-trigger columns. Input `events.csv` files may be tab-separated;
   the readers accept either),
   and lists merged triggers with no injected burst within 5 s as
   **spurious** (`trigger-spurious.csv`). Because the background is identical
   across datasets, the spurious list is computed once from the lowest-flux `a`
   dataset in scope and copied to the others.
3. **`trigger_qa.py`** — plots. Per dataset: an overview, flux summaries.
   Per run: recovery efficiency (fraction of catalog events with a matched
   trigger) versus peak photon flux in four energy bands
   (`trigger-checks-global-recovery-*.png`), population plots per flux band,
   a spurious-trigger summary (`trigger-spurious-summary.txt`,
   `trigger-checks-spurious-summary-canonical.png`), per-detector lightcurves
   around every spurious trigger (`spurious-lightcurves/`) and their per-channel
   SNRs (`trigger-spurious-channel-snr.csv`). Under `outputs/` it also writes
   `trigger-checks-global-recovery-across-runs-*.png`, overlaying every complete
   run so configurations can be compared. The 50–300 keV plots include the
   Fermi-GBM 64 ms reference curve from `gbm_64ms.txt`.

Stages 2 and 3 operate on the run named in `outputs/_current_run.txt`, so run
them after stage 1 and with the same `--complete` / `--acs-root` options. The
`outputs/` directory is git-ignored.

## Configuration

### `algorithms.ini`

The complete flight trigger configuration; every value maps 1:1 onto a CTM
command field. Comment lines start with `#`. The file has extensive inline
documentation; in brief:

- One global line `iop_trig_ra_cfg <n> <delay>` sets the running-average
  background for all channels: the mean of the `2^n` most recent 50 ms
  samples, ending `delay` samples before the current one (the delay keeps the
  burst itself out of its own background estimate). Current values `8 100`
  give a 12.8 s window ending 5 s before the current bin.
- Sixteen test rows, one per `iop_trig_test_cfg` id (0–15), each with 28
  tokens:

  ```
  index  bin_size  n_chan_over_thresh  phase_steps   thresh[ch0..ch11]   enable[ch0..ch11]
  ```

  `bin_size` is the integration length in 50 ms samples (1–256),
  `n_chan_over_thresh` the number of enabled channels that must exceed
  threshold simultaneously, `phase_steps` the stride between evaluations in
  samples. Thresholds are the raw `iop_trig_thresh_cfg` codes (fixed point,
  σ = code / 16). Enables are `1`/`0` per channel; the 12 channels are the
  (gamma, proton) energy pair for each of the six shield panels, and the
  current table enables the gamma channels only.

Editing this file and re-running the three stages produces a new
`outputs/run_*` directory; the across-runs figures then show the old and new
configurations together.

`algorithms_deprecated_unscaled-background.ini` is the earlier table tuned
before the ×1.2 background scaling; it is kept for reference and is not read by
the code.

### `data.ini`

Two lines: the ACS time-bin width in seconds (`0.05`), which must match the
HDF5 files, and the energy bin edges in keV (`80 2000 10000000`), which are
informational. The gamma/proton split actually applied to the HDF5 energy bins
is `BGO_GAMMA_PROTON_SPLIT_KEV = 2000.0` in `trigger_acs.py`.

### `gbm_64ms.txt`

Fermi-GBM 64 ms trigger efficiency versus 50–300 keV peak photon flux, used only
as a reference curve in the QA plots.

## What the emulation does

For every 50 ms bin and every enabled test: sum the last `bin_size` samples in
each channel (`S`), scale the running-average background to the same length
(`B`), and compute `SNR = (S − B) / sqrt(B)` when `B > 0` and `S > B`. The test
fires when at least `n_chan_over_thresh` enabled channels have `SNR` strictly
above their threshold in the same bin. The SNR definition is an emulation
assumption: the CTM specifies the running average, binning, thresholds and
coincidence logic, but not the significance formula itself. The channel
ordering (which physical shield panel and energy band is `ch0` … `ch11`) is
likewise assumed and documented in `algorithms.ini`; verify it against the
shield-module message definition before commanding flight thresholds.
