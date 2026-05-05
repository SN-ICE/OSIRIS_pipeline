# GTC_PypeIt

Reduction tools for GTC/OSIRIS long-slit spectroscopy and broadband imaging.

## Contents

| Script / folder | Purpose |
| --- | --- |
| [`reduce_osiris.py`](reduce_osiris.py) | End-to-end OSIRIS spectroscopy reduction with PypeIt |
| [`reduce_imaging.py`](reduce_imaging.py) | Broadband imaging reduction, per-filter combination, shared-calibration fallback |
| [`reduce_imaging_manual_alignement.py`](reduce_imaging_manual_alignement.py) | Manual 3-point imaging alignment and combination after calibration |
| [`img_cal/`](img_cal/) | Shared ready-to-use imaging master bias/flat calibrations |
| [`notebooks/`](notebooks/) | Original exploratory notebooks |

## Spectroscopy — `reduce_osiris.py`

`reduce_osiris.py` runs a full OSIRIS long-slit reduction using
[PypeIt](https://pypeit.readthedocs.io), including flux calibration, telluric
correction, optional exposure combination, and optional `R1000B+R1000R`
stitching.

It now supports both detector families:

- `gtc_osiris` for the old CCD
- `gtc_osiris_plus` for the new CCD

The default `--spectrograph auto` mode inspects the raw headers and chooses the
right one automatically.

### Expected OB structure

```text
OB0007/
├── arc/
├── bias/
├── flat/
├── object/
└── stds/
```

### Main workflow

1. Stage raw files into `_raw_pypeit/`, excluding `GRISM=OPEN` acquisition
   images from `object/` and `stds/`.
2. Patch staged FITS headers when needed so PypeIt can run on archive data
   cleanly. This includes inferred `CCDSUM`, `BINNING`, and `RDNOISE`.
3. Run `pypeit_setup`.
4. Patch the generated `.pypeit` files:
   - inject reduction parameters
   - propagate missing bias rows between configs when needed
   - force files from `arc/` to be typed as `arc,tilt`
5. Run `run_pypeit`.
6. Optionally inspect the extracted traces interactively.
7. Build and smooth a sensfunc from the standard star.
8. Flux calibrate the science spectra.
9. Export 3-column ASCII spectra with metadata headers.
10. Build and apply a telluric correction from the standard star.
11. Combine multiple exposures of the same target/grating if more than one
    corrected spectrum exists.
12. Stitch `R1000B` and `R1000R` into `R1000BR` when both are present.

### Interactive trace choices

During trace validation, science frames can use:

- `[g]` accept the current green trace
- `[o]` accept the orange preview trace without rerunning PypeIt
- `[m]` type a manual spatial position and rerun `run_pypeit -o`
- `[h]` shift the green trace by a constant spatial offset and rerun
  `run_pypeit -o`
- `[s]` draw a fully custom “supermanual” trace by clicking points on the 2D
  spectrum

Notes:

- `[h]` should usually be used once. After the rerun, if the new green trace is
  correct, accept it with `[g]` or `Enter`.
- `[s]` is only for science frames, not standards.
- `[s]` uses a custom extraction from the reduced `spec2d` product and applies
  a local residual background subtraction around the clicked trace.
- `[h]` also applies a local residual background subtraction at export time, so
  it works better when galaxy light contaminates the trace.

### Output naming

Per-exposure spectra use unique filenames based on object, grating, real
`DATE-OBS`, and raw ID:

```text
TARGET_GRATING_YYYYMMDDThhmmss_RAWID.txt
TARGET_GRATING_YYYYMMDDThhmmss_RAWID_tellcorr.txt
```

If multiple corrected spectra exist for the same `(target, grating)` in the OB,
the code writes:

```text
TARGET_GRATING_YYYYMMDDThhmmss_combined_tellcorr.txt
```

where the timestamp is the average observation time of the combined exposures.

If both `R1000B` and `R1000R` exist for the same target, the code writes:

```text
TARGET_R1000BR_YYYYMMDDThhmmss_merged_tellcorr.txt
```

using the midpoint timestamp of the two arms. The merge uses:

- a default split at `6200 Å`
- a default matching window of `6180–6220 Å`
- a fitted multiplicative scale applied to `R1000R`

### Combination and stitching behavior

- Same-grating combinations are simple means on a common wavelength grid.
- If wavelength grids differ slightly, spectra are resampled to the first
  spectrum’s grid before combining.
- The combined header records `N_EXPOSURES` and an averaged `DATE-OBS`.
- The stitched `R1000BR` header records:
  - `MERGE_REF`
  - `MERGE_ADD`
  - `MERGE_WIN`
  - `MERGE_SPLIT`
  - `MERGE_SCALE_R`

### Example outputs

```text
OB0007/
├── TARGET_R1000B_20240315T055117_0004342098.txt
├── TARGET_R1000B_20240315T055117_0004342098_tellcorr.txt
├── TARGET_R1000B_20240315T061200_combined_tellcorr.txt
├── TARGET_R1000BR_20240315T055551_merged_tellcorr.txt
├── TARGET_R1000B_20240315T055117_0004342098_tellcorr.png
└── gtc_osiris_plus_A/
    ├── Science/
    └── Calibrations/
```

ASCII headers include telescope, instrument, object, grism, `DATE-OBS`,
exposure time, airmass, and any special extraction metadata.

For custom extraction paths you may also see header lines such as:

- `EXTRACT   : SUPERMANUAL`
- `EXTRACT   : HSHIFT_LOCALBG`
- `SKY_BANDS : ...`
- `TRACE_PTS : ...`

### Usage

```bash
conda activate pypeit

# Standard interactive reduction
python reduce_osiris.py OB0007

# Prefer a science trace near a given spatial column
python reduce_osiris.py OB0007 --science-spat 665

# Fully automatic run
python reduce_osiris.py OB0007 --no-interactive

# Reuse setup and PypeIt products, redo post-processing only
python reduce_osiris.py OB0007 --skip-setup --skip-pypeit

# Change the B/R stitch location
python reduce_osiris.py OB0007 --merge-center 6200 --merge-width 40
```

### Important options

| Option | Meaning |
| --- | --- |
| `--spectrograph {auto,gtc_osiris_plus,gtc_osiris}` | Auto-detect by default; can be forced explicitly |
| `--science-spat FLOAT` | Prefer the science object closest to this spatial column |
| `--maxnumber-sci` | Maximum number of extracted objects per exposure |
| `--snr-thresh` | PypeIt object-finding S/N threshold |
| `--find-fwhm` | PypeIt object-finding FWHM |
| `--find-min-max MIN MAX` | Restrict the object-finding pixel range |
| `--trace-npoly` | Override PypeIt trace polynomial order |
| `--overwrite` | Pass `-o` to `run_pypeit` |
| `--no-interactive` | Skip validation plots |
| `--skip-setup` | Reuse existing `pypeit_setup` output |
| `--skip-pypeit` | Reuse existing `run_pypeit` output |
| `--skip-telluric` | Stop after flux-calibrated ASCII export |
| `--merge-center` | B/R stitch wavelength |
| `--merge-width` | Width of the B/R scaling window |

### Telluric bands

| Band | Range (Å) | Species |
| --- | --- | --- |
| B1 | 6855–6940 | O₂ |
| B2 | 7155–7332 | H₂O |
| B3 | 7580–7690 | O₂ |
| B4 | 8110–8357 | H₂O (`R1000R` only) |

## Imaging — `reduce_imaging.py`

`reduce_imaging.py` reduces OSIRIS broadband imaging OBs, grouping science
frames by filter and combining only frames that share the same filter.

### Expected OB structure

```text
OB0010/
├── bias/
├── flat/
└── object/
```

### Current imaging workflow

1. Read all science frames and require a single science `CCDSUM`.
2. Prefer local bias/flat calibrations from the OB directory.
3. If local calibrations are missing, fall back to shared masters in
   [`img_cal/`](img_cal/), but only when
   `CCDSUM` matches the science frames.
4. Overscan subtract and trim every frame.
5. Repair pathological overscan rows before subtraction when needed.
6. Build a master bias if a ready master is not already being used.
7. Build one master flat per filter if a ready master is not already being used.
8. Reduce science frames per filter.
9. Align to the first science frame using source-based translation from detected
   stars, with phase cross-correlation fallback.
10. Sigma-clip and combine each filter separately.

### Shared calibration fallback

The pipeline now supports shared ready-to-use masters in:

- [img_cal/bias](img_cal/bias)
- [img_cal/flats](img_cal/flats)

Current naming convention:

```text
img_cal/bias/master_bias_2x2.fits
img_cal/bias/master_bias_1x1.fits
img_cal/flats/master_flat_Sloan_r_2x2.fits
img_cal/flats/master_flat_Sloan_r_1x1.fits
...
```

Only matching `CCDSUM` calibrations are accepted. This prevents accidental use
of `2x2` calibrations for `1x1` science, or vice versa.

### Imaging outputs

```text
OB0010/Reduced/
├── TARGET_Sloan_g_YYYYMMDD.fits
├── TARGET_Sloan_r_YYYYMMDD.fits
├── master_bias.fits
├── master_flat_Sloan_g.fits
└── master_flat_Sloan_r.fits
```

If only one filter is present, the script also writes a convenience
`master_flat.fits`.

### Usage

```bash
conda activate pypeit

# Standard reduction
python reduce_imaging.py OB0010

# Custom output directory
python reduce_imaging.py OB0010 --output-dir /path/to/output

# Skip alignment
python reduce_imaging.py OB0010 --no-align
```

## Manual imaging alignment — `reduce_imaging_manual_alignement.py`

This script is for cases where the automatic imaging registration is not good
enough. It assumes calibration already exists and performs only the manual
alignment/combination step.

It:

1. Reuses `Reduced/master_bias.fits` and the relevant `Reduced/master_flat_*`
   products.
2. Re-reduces the raw science frames with those masters.
3. Lets you click the same 3 sources in every image.
4. Refines each click to the nearest source centroid.
5. Solves a rotation+translation transform relative to the first frame.
6. Combines the aligned frames.

### Usage

```bash
# Single-filter OB
python reduce_imaging_manual_alignement.py OB0008a --overwrite

# OB with multiple filters
python reduce_imaging_manual_alignement.py OB0034 --filter Sloan_u --overwrite
```

Useful options:

- `--crop-size`
- `--centroid-box`
- `--output-dir`

## Dependencies

Typical environment:

```text
PypeIt
astropy
numpy
scipy
specutils
photutils
scikit-image
matplotlib
```

PypeIt installation instructions:
[https://pypeit.readthedocs.io/en/latest/installing.html](https://pypeit.readthedocs.io/en/latest/installing.html)
