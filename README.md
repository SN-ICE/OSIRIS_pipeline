# GTC_PypeIt

Automated reduction pipelines for GTC/OSIRIS spectroscopy and broadband imaging.

## Contents

| Script | Purpose |
|--------|---------|
| [`reduce_osiris.py`](#spectroscopic-reduction-reduce_osirispy) | Full spectral reduction pipeline using PypeIt |
| [`reduce_imaging.py`](#imaging-reduction-reduce_imagingpy) | Broadband imaging pipeline (bias, flat, align, combine) |
| [`notebooks/`](notebooks/) | Original step-by-step Jupyter notebooks |

---

## Spectroscopic reduction — `reduce_osiris.py`

Automated end-to-end spectral reduction for GTC/OSIRIS longslit data using
[PypeIt](https://pypeit.readthedocs.io). Handles both R1000B and R1000R
gratings. Acquisition images (GRISM=OPEN) are automatically excluded.

### Input directory structure

```
OB0007/
├── arc/       arc lamp exposures
├── bias/      bias frames
├── flat/      spectral flat frames
├── object/    science exposures
└── stds/      standard star exposures
```

### Reduction steps

1. **Filter raw files** — symlinks to `_raw_pypeit/` excluding GRISM=OPEN acquisition images
2. **`pypeit_setup`** — auto-detects grating/slit/binning configurations
3. **Bias propagation** — copies bias rows between configurations as needed
4. **`run_pypeit`** — bias subtraction, flat fielding, wavelength calibration, object extraction
5. **[interactive] Trace validation** — inspect 2D spectrum + trace; specify manual extraction column if needed
6. **`pypeit_sensfunc`** — sensitivity function from standard star, with Savitzky-Golay smoothing (window=291, poly=3) over telluric bands
7. **[interactive] Sensfunc validation** — inspect raw vs. smoothed sensitivity function; adjust smoothing window if needed
8. **`pypeit_flux_calib`** — flux calibration in units of erg s⁻¹ cm⁻² Å⁻¹
9. **ASCII export** — 3-column output (wavelength, flux, error) with observation metadata header
10. **Telluric correction** — continuum fitting around O₂ and H₂O absorption bands using the standard star spectrum
11. **[interactive] Telluric validation** — inspect correction; adjust band limits and continuum windows per band if needed
12. **Combination** — sigma-clipped mean over multiple exposures of the same target/grating

### Output files

All outputs are written to the OB directory:

```
OB0007/
├── TARGET_R1000B_YYYYMMDD.txt                  wavelength, flux, err (flux-calibrated)
├── TARGET_R1000B_YYYYMMDD_tellcorr.txt         telluric-corrected
├── TARGET_R1000B_YYYYMMDD_combined_tellcorr.txt  combined over exposures
├── TARGET_R1000B_YYYYMMDD_tellcorr.png         final spectrum plot
└── gtc_osiris_plus_A/                          PypeIt working directory
    ├── Science/                                extracted spec1d/spec2d FITS
    └── Masters/                                calibration masters
```

Each `.txt` file has a commented header with telescope, instrument, object name,
grating, date, exposure time, and airmass.

### Usage

```bash
conda activate pypeit

# Standard run (interactive)
python reduce_osiris.py OB0007

# Fully automated (no plots, no pauses)
python reduce_osiris.py OB0007 --no-interactive

# Reuse existing PypeIt output, redo post-processing only
python reduce_osiris.py OB0007 --skip-setup --skip-pypeit

# Options
python reduce_osiris.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--spectrograph` | `gtc_osiris_plus` | Use `gtc_osiris` for the old CCD |
| `--maxnumber-sci` | `1` | Max objects extracted per exposure |
| `--snr-thresh` | PypeIt default (10) | S/N threshold for object finding |
| `--find-fwhm` | PypeIt default (5) | FWHM in pixels for object finding |
| `--find-min-max MIN MAX` | — | Restrict object search to pixel range |
| `--overwrite` | off | Overwrite existing Masters/Science |
| `--no-interactive` | off | Skip all validation plots |
| `--skip-setup` | off | Reuse existing pypeit_setup output |
| `--skip-pypeit` | off | Reuse existing run_pypeit output |
| `--skip-telluric` | off | Skip telluric correction |

### Telluric bands

| Band | Range (Å) | Species |
|------|-----------|---------|
| B1 | 6855 – 6940 | O₂ |
| B2 | 7155 – 7332 | H₂O |
| B3 | 7580 – 7690 | O₂ |
| B4 | 8110 – 8357 | H₂O (R1000R only) |

---

## Imaging reduction — `reduce_imaging.py`

Calibration and combination pipeline for GTC/OSIRIS broadband images.

### Input directory structure

```
OB0010/
├── bias/      bias frames
├── flat/      sky flat frames
└── object/    science exposures
```

### Reduction steps

1. **Overscan subtraction** — median overscan level per row (from `BIASSEC`), then trim to `TRIMSEC`
2. **Master bias** — median combination of all overscan-corrected bias frames
3. **Master flat** — per-frame normalisation → sigma-clipped (3σ) median stack
4. **Science reduction** — overscan subtract → trim → bias subtract → flat divide
5. **WCS alignment** — pixel shifts computed from each frame's WCS; applied with cubic spline interpolation. Falls back to phase cross-correlation if WCS is absent
6. **Combination** — sigma-clipped (3σ) mean

### Output files

```
OB0010/Reduced/
├── TARGET_FILTER_YYYYMMDD.fits   combined, calibrated science image (with WCS)
├── master_bias.fits
└── master_flat.fits
```

### Usage

```bash
conda activate pypeit

# Standard run
python reduce_imaging.py OB0010

# Custom output directory
python reduce_imaging.py OB0010 --output-dir /path/to/output

# Skip WCS alignment (e.g. single exposure or fixed pointing)
python reduce_imaging.py OB0010 --no-align
```

---

## Dependencies

Both scripts run inside the `pypeit` conda environment. Additional packages:

```
astropy
scipy
specutils      # reduce_osiris.py
spectres       # reduce_osiris.py
scikit-image   # reduce_imaging.py (cross-correlation fallback)
```

PypeIt installation: https://pypeit.readthedocs.io/en/latest/installing.html
