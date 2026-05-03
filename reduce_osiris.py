#!/usr/bin/env python
"""
Automated OSIRIS/GTC spectral reduction pipeline using PypeIt.

Usage:
    conda activate pypeit
    python reduce_osiris.py OB_DIR [options]

OB_DIR must be a GTC observation block directory containing:
    arc/, bias/, flat/, object/, stds/

The script runs the full reduction chain:
  1. pypeit_setup
  2. .pypeit file editing (bias fix, reduction params)
  3. run_pypeit
  3b. [interactive] trace validation — re-run with manual extraction if needed
  4. pypeit_sensfunc + Savitzky-Golay smoothing
  4b. [interactive] sensfunc smoothing validation — re-smooth if needed
  5. pypeit_flux_calib
  6. ASCII export
  7. Telluric correction via standard-star continuum fitting
  7b. [interactive] telluric correction validation

Final outputs (written to OB_DIR):
    TARGET_GRATING_DATE.txt                    – flux-calibrated (wave, flux, err)
    TARGET_GRATING_DATE_tellcorr.txt           – telluric-corrected
    TARGET_GRATING_DATE_combined_tellcorr.txt  – averaged over exposures (if >1)

All ASCII files include a commented header with observation metadata.
Pass --no-interactive to run the full pipeline without pausing for inspection.
"""

import argparse
import copy
import os
import re
import subprocess
import sys
import numpy as np
from pathlib import Path

from astropy import units as u
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from specutils.fitting.continuum import fit_continuum
from specutils.spectra import Spectrum1D


# ── Constants ──────────────────────────────────────────────────────────────

RAW_SUBDIRS = ['arc', 'bias', 'flat', 'object', 'stds']

# Telluric absorption bands [Å].  4th band is R1000R only.
# Each entry: (band_lo, band_hi, left_continuum_window, right_continuum_window)
TELLURIC_BANDS = [
    (6855, 6940,  (6700, 6825),  (6950, 7100)),   # O2
    (7155, 7332,  (6950, 7150),  (7300, 7500)),   # H2O
    (7580, 7690,  (7560, 7570),  (7700, 7720)),   # O2
    (8110, 8357,  (7800, 8100),  (8360, 8550)),   # H2O – R1000R only
]

SAVGOL_WINDOW = 291
SAVGOL_POLY   = 3


# ── Shell helper ───────────────────────────────────────────────────────────

def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


# ── pypeit_setup helpers ───────────────────────────────────────────────────

def find_config_dirs(ob_dir: Path):
    return sorted(p for p in ob_dir.iterdir()
                  if p.is_dir() and re.match(r'gtc_osiris.*_[A-Z]$', p.name))


def get_dispname(pypeit_file: Path) -> str:
    for line in pypeit_file.read_text().splitlines():
        if 'dispname:' in line:
            return line.split(':')[1].strip()
    return 'UNKNOWN'


def get_calib_group(pypeit_file: Path) -> str:
    for line in pypeit_file.read_text().splitlines():
        if '| arc,tilt |' in line or '| pixelflat' in line:
            parts = line.split('|')
            if len(parts) > 11:
                return parts[11].strip()
    return '0'


def has_bias(pypeit_file: Path) -> bool:
    # Use regex to handle variable column padding across PypeIt versions
    return bool(re.search(r'\|\s+bias\s+\|', pypeit_file.read_text()))


def copy_bias_from(src: Path, dst: Path):
    src_text = src.read_text()
    bias_lines = [l for l in src_text.splitlines()
                  if re.search(r'\|\s+bias\s+\|', l)]
    if not bias_lines:
        return False
    dst_calib = get_calib_group(dst)
    fixed = []
    for line in bias_lines:
        parts = line.split('|')
        if len(parts) > 11:
            parts[11] = f'     {dst_calib} '
        fixed.append('|'.join(parts))
    dst_text = dst.read_text()
    dst_text = dst_text.replace('data end', '\n'.join(fixed) + '\ndata end')
    dst.write_text(dst_text)
    return True


def patch_pypeit_params(pypeit_file: Path, maxnumber_sci: int,
                        snr_thresh, find_fwhm, find_min_max, spectrograph: str,
                        trace_npoly: int = None):
    rdx_block = f'[rdx]\n    spectrograph = {spectrograph}\n'
    if spectrograph == 'gtc_osiris':
        rdx_block += '    detnum = 2\n'
    reduce_block  = '[reduce]\n    [[findobj]]\n'
    reduce_block += f'        maxnumber_sci = {maxnumber_sci}\n'
    if snr_thresh   is not None: reduce_block += f'        snr_thresh = {snr_thresh}\n'
    if find_fwhm    is not None: reduce_block += f'        find_fwhm = {find_fwhm}\n'
    if find_min_max is not None:
        reduce_block += f'        find_min_max = {find_min_max[0]}, {find_min_max[1]}\n'
    if trace_npoly  is not None: reduce_block += f'        trace_npoly = {trace_npoly}\n'

    text = pypeit_file.read_text()
    if '# Setup' not in text:
        print(f"  WARNING: '# Setup' not found in {pypeit_file.name}, skipping param patch")
        return
    pre, post = text.split('# Setup', 1)

    clean, skip = [], False
    for line in pre.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('[rdx]') or stripped.startswith('[reduce]'):
            skip = True; continue
        if skip:
            if stripped == '' or stripped.startswith('#') or (
                    stripped.startswith('[') and not stripped.startswith('[[')):
                skip = False
            else:
                continue
        clean.append(line)
    pypeit_file.write_text(''.join(clean) + rdx_block + '\n' + reduce_block + '\n# Setup' + post)


def set_manual_extraction(pypeit_file: Path, spec1d_name: str,
                          spat: float, spec_px: float, fwhm: float, det: int = 1):
    """Add / update the manual extraction column in the .pypeit data block.

    Works regardless of whether the pypeit file uses leading/trailing '|' per
    row (older PypeIt) or bare '|'-separated values (newer PypeIt), and whether
    'manual' already exists as a column from a previous run.

    Strategy:
      - Parse the column-header line to count fields and detect 'manual'.
      - Place values by column index (split/rejoin on '|'), never by string
        appending, so the last existing column is never clobbered.
    """
    manual_str = f'{det}:{spat:.3f}:{spec_px:.3f}:{fwhm:.1f}'

    m = re.match(r'spec1d_(\d+)-', spec1d_name)
    if not m:
        print(f"  WARNING: cannot parse raw ID from {spec1d_name}")
        return
    raw_id = m.group(1)

    text = pypeit_file.read_text()
    if 'data read' not in text or 'data end' not in text:
        print(f"  WARNING: data block not found in {pypeit_file.name}")
        return

    pre, rest = text.split('data read', 1)
    block, post = rest.split('data end', 1)
    block_lines = block.splitlines(keepends=True)

    # ── Pass 1: locate the header line ────────────────────────────────────
    hdr_idx = None
    hdr_parts = None    # raw split fields (preserving whitespace)
    hdr_names = None    # stripped column names
    for i, line in enumerate(block_lines):
        if 'filename' in line and '|' in line:
            hdr_idx  = i
            hdr_parts = line.rstrip('\n').split('|')
            hdr_names = [c.strip() for c in hdr_parts]
            break

    if hdr_idx is None:
        print(f"  WARNING: header line not found in {pypeit_file.name}")
        return

    has_manual = 'manual' in hdr_names
    n_cols = len(hdr_parts)   # expected number of '|'-split fields per data row

    # ── Pass 2: rewrite the block ─────────────────────────────────────────
    new_block = list(block_lines)

    if not has_manual:
        # Extend header with a 'manual' column
        new_block[hdr_idx] = '|'.join(hdr_parts).rstrip() + ' | manual\n'
        n_cols += 1

    for i, line in enumerate(new_block):
        if i == hdr_idx:
            continue
        s = line.rstrip('\n').rstrip()
        if not s or '|' not in s:
            continue                   # blank line, path line, etc.

        parts = s.split('|')

        if raw_id in s:
            # Science target row: set value at the manual column position
            # Pad to n_cols-1 fields first if the row is short (shouldn't happen,
            # but guards against pre-existing corruption)
            while len(parts) < n_cols - 1:
                parts.append('')
            if len(parts) < n_cols:
                parts.append(f' {manual_str} ')
            else:
                parts[-1] = f' {manual_str} '
            new_block[i] = '|'.join(parts) + '\n'

        elif not has_manual and len(parts) < n_cols:
            # Other data rows: pad with empty field when we just added the manual col
            parts.append('')
            new_block[i] = '|'.join(parts) + '\n'

    pypeit_file.write_text(pre + 'data read' + ''.join(new_block) + 'data end' + post)
    print(f"  Manual extraction set: det={det}, spat={spat:.1f}, "
          f"spec={spec_px:.1f}, fwhm={fwhm:.1f}")


# ── Observation metadata ───────────────────────────────────────────────────

def _hdr_get(hdr, *keys, default='UNKNOWN'):
    for k in keys:
        v = hdr.get(k)
        if v is not None:
            return v
    return default


def read_obs_meta(spec1d_fits: Path, ob_dir: Path) -> dict:
    """Read observation metadata, preferring the raw science FITS header."""
    m = re.match(r'spec1d_(\d+-\d{8}-OSIRIS-OsirisLongSlitSpectroscopy)-', spec1d_fits.name)
    raw_hdr = None
    if m:
        candidates = list((ob_dir / 'object').glob(f'{m.group(1)}*.fits'))
        if candidates:
            with fits.open(candidates[0]) as h:
                raw_hdr = h[0].header
    if raw_hdr is None:
        with fits.open(spec1d_fits) as h:
            raw_hdr = h[0].header

    return {
        'telescope':  _hdr_get(raw_hdr, 'TELESCOP', 'TELESCOPE',  default='GTC'),
        'instrument': _hdr_get(raw_hdr, 'INSTRUME',  'INSTRUMENT', default='OSIRIS'),
        'object':     _hdr_get(raw_hdr, 'OBJECT', 'TARGET', 'OBJNAME', default='UNKNOWN'),
        'grism':      _hdr_get(raw_hdr, 'GRISM', 'DISPNAME', default='UNKNOWN'),
        'date_obs':   _hdr_get(raw_hdr, 'DATE-OBS', 'MJD-OBS', default='UNKNOWN'),
        'exptime':    _hdr_get(raw_hdr, 'EXPTIME', default='UNKNOWN'),
        'airmass':    _hdr_get(raw_hdr, 'AIRMASS', default='UNKNOWN'),
    }


def build_header(meta: dict, n_exposures: int = 1, extra_lines: list = None) -> str:
    """Return a header string (without leading '#'; numpy adds them on savetxt)."""
    lines = [
        f"TELESCOPE : {meta['telescope']}",
        f"INSTRUMENT: {meta['instrument']}",
        f"OBJECT    : {meta['object']}",
        f"GRISM     : {meta['grism']}",
        f"DATE-OBS  : {meta['date_obs']}",
        f"EXPTIME   : {meta['exptime']} s",
        f"AIRMASS   : {meta['airmass']}",
    ]
    if n_exposures > 1:
        lines.append(f"N_EXPOSURES: {n_exposures}")
    if extra_lines:
        lines.extend(extra_lines)
    lines.append("COLUMNS   : wavelength[AA]  flux[erg/s/cm2/AA]  flux_err[erg/s/cm2/AA]")
    return '\n'.join(lines)


# ── Sensitivity function ───────────────────────────────────────────────────

def _do_smooth_sensfunc(sensfunc_raw: Path, sensfunc_smth: Path, window: int = SAVGOL_WINDOW):
    with fits.open(sensfunc_raw) as hdul:
        lam  = hdul[2].data.T[0].copy()
        sens = hdul[3].data.T[0].copy()
        hdu_copy = copy.deepcopy(hdul)

    for (w0, w1, (l0, l1), (r0, r1)) in TELLURIC_BANDS:
        if w1 < lam.min() or w0 > lam.max():
            continue
        left  = (lam >= l0) & (lam <= l1)
        right = (lam >= r0) & (lam <= r1)
        band  = (lam >= w0) & (lam <= w1)
        if not (np.any(left) and np.any(right)):
            continue
        x = np.concatenate([lam[left], lam[right]])
        y = np.concatenate([sens[left], sens[right]])
        sens[band] = interp1d(x, y, kind='linear', fill_value='extrapolate')(lam[band])

    smoothed = savgol_filter(sens, window, SAVGOL_POLY)
    hdu_copy[3].data = smoothed.reshape(-1, 1)
    hdu_copy.writeto(str(sensfunc_smth), overwrite=True)
    return lam, sens, smoothed   # original-after-interp, smoothed


def smooth_sensfunc(sensfunc_raw: Path, sensfunc_smth: Path):
    _do_smooth_sensfunc(sensfunc_raw, sensfunc_smth, SAVGOL_WINDOW)
    print(f"  Smoothed sensfunc → {sensfunc_smth.name}")


# ── Flux calibration ───────────────────────────────────────────────────────

def write_flux_file(flux_txt: Path, sci_basenames: list, sensfunc_rel: str):
    with open(flux_txt, 'w') as f:
        f.write('flux read\n')
        f.write('    filename | sensfile\n')
        for i, name in enumerate(sci_basenames):
            sf = sensfunc_rel if i == 0 else ''
            f.write(f'    {name} | {sf}\n')
        f.write('flux end\n')


def export_ascii(spec1d_fits: Path, out_txt: Path, ob_dir: Path):
    with fits.open(spec1d_fits) as h:
        data = h[1].data
        wave = data['OPT_WAVE'].astype(float)
        flux = data['OPT_FLAM'].astype(float) * 1e-17
        if 'OPT_FLAM_SIG' in data.names:
            err = data['OPT_FLAM_SIG'].astype(float) * 1e-17
        else:
            print(f"  WARNING: OPT_FLAM_SIG not in {spec1d_fits.name}; errors set to 0")
            err = np.zeros_like(flux)
    meta = read_obs_meta(spec1d_fits, ob_dir)
    hdr  = build_header(meta)
    np.savetxt(str(out_txt), np.column_stack([wave, flux, err]),
               fmt=['%.3f', '%e', '%e'], header=hdr, comments='# ')
    print(f"  ASCII spectrum → {out_txt.name}")


def parse_spec1d_meta(filename: str):
    m = re.search(r'-(\d{8})-OSIRIS-OsirisLongSlitSpectroscopy-(.+?)_OSIRIS_',
                  os.path.basename(filename))
    return (m.group(2), m.group(1)) if m else (None, None)


def _read_ascii3(path: Path):
    data = np.loadtxt(str(path), comments='#')
    return data[:, 0], data[:, 1], data[:, 2]


def _read_header_lines(path: Path) -> list:
    lines = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                lines.append(line.rstrip())
            else:
                break
    return lines


# ── Telluric correction ────────────────────────────────────────────────────

def _fit_or_interp(spectrum, lam, flux, w0, w1, l0, l1, r0, r1):
    band = (lam >= w0) & (lam <= w1)
    try:
        cont = fit_continuum(spectrum, window=[(l0*u.AA, l1*u.AA), (r0*u.AA, r1*u.AA)])
        return cont(lam[band] * u.AA).value
    except Exception as e:
        print(f"    WARNING: continuum fit {w0}–{w1} Å failed ({e}); using linear interp")
        y_pts = [flux[np.argmin(np.abs(lam - w0))], flux[np.argmin(np.abs(lam - w1))]]
        return np.interp(lam[band], [w0, w1], y_pts)


def build_tell_correction(std_spec1d: Path, grating: str, bands=None):
    """Build telluric correction from standard star OPT_COUNTS.

    bands: list of (band_lo, band_hi, (l0,l1), (r0,r1)) entries to use.
           Defaults to TELLURIC_BANDS[:3] for R1000B, TELLURIC_BANDS for R1000R.
    """
    if bands is None:
        bands = TELLURIC_BANDS if grating == 'R1000R' else TELLURIC_BANDS[:3]
    with fits.open(std_spec1d) as h:
        lam  = h[1].data['OPT_WAVE'].astype(float)
        flux = h[1].data['OPT_COUNTS'].astype(float)
    spectrum = Spectrum1D(spectral_axis=lam * u.AA, flux=flux * u.ct)
    f = flux.copy()
    for (w0, w1, (l0, l1), (r0, r1)) in bands:
        if w1 < lam.min() or w0 > lam.max():
            continue
        f[(lam >= w0) & (lam <= w1)] = _fit_or_interp(spectrum, lam, flux, w0, w1, l0, l1, r0, r1)
    bad = ~np.isfinite(f) | (f <= 0)
    f[bad] = flux[bad]
    tell_corr = flux / f
    tell_corr[~np.isfinite(tell_corr)] = 1.0
    return lam, tell_corr


def apply_tell_correction(science_txt: Path, lam_tell, tell_corr, out_txt: Path):
    wave, flux, err = _read_ascii3(science_txt)
    tc = np.interp(wave, lam_tell, tell_corr, left=1.0, right=1.0)
    orig_hdr = _read_header_lines(science_txt)
    hdr_str  = '\n'.join(l.lstrip('# ') for l in orig_hdr) + '\nTELLURIC  : corrected'
    np.savetxt(str(out_txt), np.column_stack([wave, flux / tc, err / tc]),
               fmt=['%.3f', '%e', '%e'], header=hdr_str, comments='# ')
    print(f"  Telluric-corrected → {out_txt.name}")


def combine_spectra(txt_files: list, out_txt: Path):
    arrays  = [_read_ascii3(f) for f in txt_files]
    wave    = arrays[0][0]
    n       = len(arrays)
    flux_avg = np.mean([a[1] for a in arrays], axis=0)
    err_avg  = np.sqrt(np.sum([a[2]**2 for a in arrays], axis=0)) / n
    orig_hdr = _read_header_lines(txt_files[0])
    hdr_str  = '\n'.join(l.lstrip('# ') for l in orig_hdr)
    hdr_str  = hdr_str.replace('TELLURIC  : corrected',
                                f'N_EXPOSURES: {n}\nTELLURIC  : corrected')
    np.savetxt(str(out_txt), np.column_stack([wave, flux_avg, err_avg]),
               fmt=['%.3f', '%e', '%e'], header=hdr_str, comments='# ')
    print(f"  Combined ({n} exposures) → {out_txt.name}")


# ── Interactive validation ─────────────────────────────────────────────────

def _ask(prompt: str, default: str = 'y') -> str:
    """Print prompt and return stripped lowercase answer (never blocks in non-interactive)."""
    ans = input(prompt).strip().lower()
    return ans if ans else default


def _compute_trace(img: np.ndarray, init_trace: np.ndarray,
                   step: int = 50, half_window: int = 20):
    """Estimate the object trace by finding the peak-flux spatial pixel
    in blocks of `step` spectral rows, searching within `half_window`
    pixels of the initial (PypeIt) trace.

    Returns (sample_rows, sample_peaks, full_trace) where full_trace is
    a polynomial fit evaluated at every spectral row.  Returnsif
    fewer than 3 valid samples are found.
    """
    nspec, nspat = img.shape
    rows, peaks = [], []

    for r0 in range(0, nspec, step):
        r1 = min(r0 + step, nspec)
        mid = (r0 + r1) // 2

        # Anchor: median of PypeIt trace in this row range (fall back to centre)
        if init_trace is not None:
            anchor = int(np.nanmedian(init_trace[r0:r1]))
        else:
            anchor = nspat // 2

        lo = max(0, anchor - half_window)
        hi = min(nspat, anchor + half_window + 1)

        profile = np.nanmedian(img[r0:r1, lo:hi], axis=0)
        valid = np.isfinite(profile)
        if not np.any(valid):
            continue

        peak_spat = lo + int(np.argmax(np.where(valid, profile, -np.inf)))
        rows.append(float(mid))
        peaks.append(float(peak_spat))

    if len(rows) < 3:
        return None

    rows_arr  = np.array(rows)
    peaks_arr = np.array(peaks)
    deg       = min(4, len(rows_arr) - 1)
    coeffs    = np.polyfit(rows_arr, peaks_arr, deg)
    full_spec = np.arange(nspec, dtype=float)
    return rows_arr, peaks_arr, np.polyval(coeffs, full_spec)


def validate_trace(spec2d_path: Path, spec1d_path: Path, target: str) -> tuple:
    """Show 2D spectrum + spatial profile with extracted trace.

    Returns (ok: bool, manual_params: (spat, spec, fwhm) or None).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ── read 2D image ──────────────────────────────────────────────────────
    sciimg = skymodel = None
    with fits.open(spec2d_path) as h:
        for ext in h:
            if ext.data is None or np.ndim(ext.data) != 2:
                continue
            name = ext.name.upper()
            if 'SCIIMG' in name and sciimg is None:
                sciimg = ext.data.copy()
            elif 'SKYMODEL' in name and skymodel is None:
                skymodel = ext.data.copy()
    if sciimg is None:
        print(f"  WARNING: SCIIMG not found in {spec2d_path.name} – skipping trace plot")
        return True, None
    img = (sciimg - skymodel) if skymodel is not None else sciimg
    nspec, nspat = img.shape

    # ── read trace ─────────────────────────────────────────────────────────
    trace_spat = None
    with fits.open(spec1d_path) as h:
        cols = [c.name.upper() for c in h[1].columns]
        key  = next((c for c in cols if 'TRACE_SPAT' in c), None)
        if key:
            trace_spat = h[1].data[key].astype(float)

    # ── plot ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f'Trace validation – {target}  ({spec1d_path.name})', fontsize=11)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax2d = fig.add_subplot(gs[0])
    axsp = fig.add_subplot(gs[1], sharey=None)

    vmin, vmax = np.nanpercentile(img, [2, 98])
    ax2d.imshow(img, origin='lower', aspect='auto', cmap='RdYlBu_r',
                vmin=vmin, vmax=vmax,
                extent=[0, nspat - 1, 0, nspec - 1])

    # PypeIt polynomial trace (green)
    if trace_spat is not None:
        ax2d.plot(trace_spat, np.arange(nspec), color='lime',
                  lw=1.5, label=f'PypeIt trace (spat≈{np.nanmedian(trace_spat):.0f})')

    # Custom peak-finding trace (orange): max flux every ~50 rows ± 20 px window
    custom = _compute_trace(img, trace_spat)
    if custom is not None:
        sample_rows, sample_peaks, full_trace = custom
        ax2d.scatter(sample_peaks, sample_rows, s=12, color='orange',
                     zorder=5, label='peak samples')
        ax2d.plot(full_trace, np.arange(nspec), color='orange',
                  lw=1.5, ls='--', label=f'custom trace (spat≈{np.nanmedian(full_trace):.0f})')

    ax2d.legend(fontsize=9, loc='upper right')
    ax2d.set_xlabel('Spatial pixel')
    ax2d.set_ylabel('Spectral pixel')
    ax2d.set_title('Sky-subtracted 2D spectrum')

    profile = np.nanmedian(img, axis=0)
    axsp.plot(profile, np.arange(nspat), 'k', lw=1)
    if trace_spat is not None:
        med = int(np.nanmedian(trace_spat))
        axsp.axhline(med, color='lime', lw=2, label=f'PypeIt spat={med}')
    if custom is not None:
        med_custom = int(np.nanmedian(full_trace))
        axsp.axhline(med_custom, color='orange', lw=2, ls='--',
                     label=f'custom spat={med_custom}')
    axsp.legend(fontsize=9)
    axsp.set_xlabel('Median counts')
    axsp.set_title('Spatial profile')
    axsp.yaxis.tick_right()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    has_custom = custom is not None
    if has_custom:
        prompt = f"\n  [{target}] Which trace to use? [g]reen, [o]range, [m]anual or [s]kip? "
    else:
        prompt = f"\n  [{target}] Which trace to use? [g]reen, [m]anual or [s]kip? "

    ans = _ask(prompt, default='g')
    plt.close('all')

    # ── green: accept PypeIt trace as-is ──────────────────────────────────
    if ans in ('g', 's', ''):
        return True, None

    # ── orange: use custom peak-finding trace ──────────────────────────────
    if ans == 'o' and has_custom:
        spat    = float(np.nanmedian(full_trace))
        spec_px = float(nspec // 2)
        fwhm    = 4.0
        print(f"  Using custom (orange) trace: spat={spat:.1f}, spec={spec_px:.0f}, fwhm={fwhm}")
        return False, (spat, spec_px, fwhm)

    # ── manual: user types in the spatial position ─────────────────────────
    print("  Enter the position of the correct object (read from the plot):")
    try:
        spat    = float(input("    Spatial pixel : "))
        sp_def  = nspec // 2
        sp_in   = input(f"    Spectral pixel (Enter = {sp_def}): ").strip()
        spec_px = float(sp_in) if sp_in else float(sp_def)
        fw_in   = input("    FWHM in pixels (Enter = 4.0): ").strip()
        fwhm    = float(fw_in) if fw_in else 4.0
    except ValueError:
        print("  Invalid input – skipping manual extraction")
        return True, None

    return False, (spat, spec_px, fwhm)


def validate_sensfunc(sensfunc_raw: Path, sensfunc_smth: Path) -> None:
    """Show original vs. smoothed sensfunc; offer to re-smooth with different window."""
    import matplotlib.pyplot as plt

    while True:
        with fits.open(sensfunc_raw) as h:
            lam  = h[2].data.T[0]
            orig = h[3].data.T[0].copy()
        with fits.open(sensfunc_smth) as h:
            smth = h[3].data.T[0].copy()

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(lam, orig, color='0.3', lw=2,   alpha=0.7, label='Original sensfunc')
        ax.plot(lam, smth, color='red', lw=2,   alpha=0.9, label='Smoothed sensfunc')
        for i, (w0, w1, _, _) in enumerate(TELLURIC_BANDS):
            if w1 >= lam.min() and w0 <= lam.max():
                ax.axvspan(w0, w1, alpha=0.12, color='steelblue',
                           label='Telluric bands (masked)' if i == 0 else '')
        ax.set_xlabel('Wavelength [Å]', fontsize=12)
        ax.set_ylabel('Sensitivity', fontsize=12)
        ax.set_title(f'Sensitivity function – {sensfunc_raw.parent.name}', fontsize=12)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)

        ans = _ask("\n  Sensfunc smoothing OK? [y/n/skip] (default y): ")
        plt.close('all')

        if ans in ('y', 'skip', ''):
            return

        win_in = input(f"  New Savitzky-Golay window (current {SAVGOL_WINDOW}, must be odd): ").strip()
        try:
            new_win = int(win_in)
            if new_win % 2 == 0:
                new_win += 1
            print(f"  Re-smoothing with window={new_win}…")
            _do_smooth_sensfunc(sensfunc_raw, sensfunc_smth, window=new_win)
            print(f"  Re-smoothed. Showing updated plot…")
        except ValueError:
            print("  Invalid value – keeping current smoothing.")
            return


def _draw_telluric_plot(std_spec1d, lam_tell, tell_corr,
                        grating, science_txts, bands):
    """Draw the two-panel telluric correction figure; return the figure."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    with fits.open(std_spec1d) as h:
        lam_std  = h[1].data['OPT_WAVE'].astype(float)
        flux_std = h[1].data['OPT_COUNTS'].astype(float)

    band_colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']
    band_labels = ['O₂ 6855–6940 Å', 'H₂O 7155–7332 Å',
                   'O₂ 7580–7690 Å', 'H₂O 8110–8357 Å']

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.suptitle(f'Telluric correction check – {grating}', fontsize=12)

    # ── Panel 1: std star + continuum model ───────────────────────────────
    ax = axes[0]
    ax.plot(lam_std, flux_std, color='0.2', lw=1, alpha=0.8, label='Std star (counts)')
    tc_on_std = np.interp(lam_std, lam_tell, tell_corr, left=1.0, right=1.0)
    ax.plot(lam_std, flux_std / tc_on_std, color='tomato', lw=2, alpha=0.8,
            label='Continuum model')
    patch_handles = []
    for i, (w0, w1, (l0, l1), (r0, r1)) in enumerate(bands):
        if w1 < lam_std.min() or w0 > lam_std.max():
            continue
        col = band_colors[i % len(band_colors)]
        ax.axvspan(w0, w1, alpha=0.20, color=col)
        # Continuum windows as hatched regions
        for wl, wh in [(l0, l1), (r0, r1)]:
            if wh >= lam_std.min() and wl <= lam_std.max():
                ax.axvspan(wl, wh, alpha=0.08, color=col, hatch='//')
        mid = (w0 + w1) / 2
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else flux_std.max(),
                f'B{i+1}', ha='center', va='top', fontsize=9, color=col, fontweight='bold')
        patch_handles.append(mpatches.Patch(color=col, alpha=0.5,
                                            label=f'B{i+1}: {band_labels[i]}'))
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_title('Standard star  (shaded = band, hatched = continuum windows)', fontsize=10)
    if patch_handles:
        ax.legend(handles=[ax.get_lines()[0], ax.get_lines()[1]] + patch_handles,
                  fontsize=9, loc='upper left')

    # ── Panel 2: science before / after ───────────────────────────────────
    ax = axes[1]
    if science_txts:
        wave_sci, flux_sci, _ = _read_ascii3(science_txts[0])
        tc_sci = np.interp(wave_sci, lam_tell, tell_corr, left=1.0, right=1.0)
        ax.plot(wave_sci, flux_sci,          color='0.2', lw=1, alpha=0.6,
                label='Before correction')
        ax.plot(wave_sci, flux_sci / tc_sci, color='tomato', lw=1.5,
                label='After correction')
        for i, (w0, w1, _, _) in enumerate(bands):
            if w1 >= wave_sci.min() and w0 <= wave_sci.max():
                ax.axvspan(w0, w1, alpha=0.15, color=band_colors[i % len(band_colors)])
        ax.set_xlabel('Wavelength [Å]', fontsize=11)
        ax.set_ylabel('Flux [erg/s/cm²/Å]', fontsize=11)
        ax.set_title(f'Science spectrum – {science_txts[0].name}', fontsize=10)
        ax.legend(fontsize=10)

    plt.tight_layout()
    return fig


def validate_telluric(std_spec1d: Path, lam_tell, tell_corr,
                      grating: str, science_txts: list) -> tuple:
    """Interactive telluric correction check.

    Returns (lam_tell, tell_corr) — either the original or rebuilt after
    the user adjusts one or more band / continuum-window definitions.
    """
    import matplotlib.pyplot as plt

    active_bands = list(TELLURIC_BANDS if grating == 'R1000R' else TELLURIC_BANDS[:3])
    band_labels  = ['O₂ 6855–6940', 'H₂O 7155–7332',
                    'O₂ 7580–7690', 'H₂O 8110–8357']

    while True:
        fig = _draw_telluric_plot(std_spec1d, lam_tell, tell_corr,
                                  grating, science_txts, active_bands)
        plt.show(block=False)
        plt.pause(0.5)

        ans = _ask("\n  Telluric correction OK? [y/n/skip] (default y): ")
        plt.close('all')

        if ans in ('y', 'skip', ''):
            return lam_tell, tell_corr

        # Show current band table
        print("\n  Current band definitions (B = band region, L/R = continuum windows):")
        for i, (w0, w1, (l0, l1), (r0, r1)) in enumerate(active_bands):
            print(f"  B{i+1} ({band_labels[i]}): "
                  f"band [{w0}–{w1}]  left [{l0}–{l1}]  right [{r0}–{r1}]")

        band_in = input(f"\n  Which band to adjust? (1–{len(active_bands)}, Enter to accept): ").strip()
        if not band_in:
            return lam_tell, tell_corr

        try:
            idx = int(band_in) - 1
            if not 0 <= idx < len(active_bands):
                print("  Invalid band number.")
                continue
        except ValueError:
            print("  Invalid input.")
            continue

        w0, w1, (l0, l1), (r0, r1) = active_bands[idx]
        print(f"\n  Adjusting B{idx+1} — press Enter to keep current value.")

        def _ask_float(label, default):
            v = input(f"    {label} [{default}]: ").strip()
            return float(v) if v else default

        try:
            new_w0 = _ask_float(f"Band start  (current {w0})", w0)
            new_w1 = _ask_float(f"Band end    (current {w1})", w1)
            new_l0 = _ask_float(f"Left win lo (current {l0})", l0)
            new_l1 = _ask_float(f"Left win hi (current {l1})", l1)
            new_r0 = _ask_float(f"Right win lo (current {r0})", r0)
            new_r1 = _ask_float(f"Right win hi (current {r1})", r1)
        except ValueError:
            print("  Invalid input — keeping previous correction.")
            continue

        active_bands[idx] = (new_w0, new_w1, (new_l0, new_l1), (new_r0, new_r1))
        print(f"  Rebuilding telluric correction with updated B{idx+1}…")
        lam_tell, tell_corr = build_tell_correction(std_spec1d, grating,
                                                    bands=active_bands)
        print("  Done. Showing updated plot…")


# ── Final summary plot ────────────────────────────────────────────────────

def plot_final_spectra(ob_dir: Path) -> None:
    """Save one PNG per final spectrum in OB_DIR, named {spectrum_stem}.png.

    Prefers *_tellcorr.txt files (individual or combined).  If none exist,
    falls back to the plain flux-calibrated *_R1000*.txt files.
    """
    import matplotlib.pyplot as plt

    # Gather files: prefer combined > individual tellcorr > plain flux
    tellcorr = sorted(ob_dir.glob('*_tellcorr.txt'))
    plain    = [f for f in sorted(ob_dir.glob('*.txt'))
                if re.search(r'_R\d{4}[BR]_\d{8}\.txt$', f.name)]
    candidates = tellcorr if tellcorr else plain
    if not candidates:
        print("  No spectra found for the final plot.")
        return

    for txt in candidates:
        try:
            wave, flux, err = _read_ascii3(txt)
        except Exception as e:
            print(f"  WARNING: could not read {txt.name} for plot ({e})")
            continue

        fig, ax = plt.subplots(figsize=(14, 5))

        m = re.match(r'(.+?)_(R\d{4}[BR])_(\d{8})', txt.stem)
        label = txt.stem if m is None else f"{m.group(1)}  {m.group(2)}  {m.group(3)}"
        if 'combined' in txt.stem:
            label += '  (combined)'

        ax.plot(wave, flux, color='steelblue', lw=1.2, label=label)
        ax.fill_between(wave, flux - err, flux + err,
                        color='steelblue', alpha=0.2)

        # Mark telluric bands
        for j, (w0, w1, _, _) in enumerate(TELLURIC_BANDS):
            ax.axvspan(w0, w1, alpha=0.08, color='steelblue',
                       label='Telluric bands' if j == 0 else '')

        ax.set_xlabel('Wavelength [Å]', fontsize=12)
        ax.set_ylabel('Flux [erg s⁻¹ cm⁻² Å⁻¹]', fontsize=12)
        ax.set_title(label, fontsize=13)
        ax.legend(fontsize=9, loc='upper left')
        ax.set_ylim(bottom=0)
        plt.tight_layout()

        out_png = txt.with_suffix('.png')
        fig.savefig(str(out_png), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Plot saved → {out_png.name}")


# ── Raw-file filtering ────────────────────────────────────────────────────

def create_filtered_rawdir(ob_dir: Path) -> Path:
    """Build _raw_pypeit/ with symlinks to every non-OPEN-grism file.

    Files with GRISM=OPEN in object/ or stds/ are acquisition images
    (through-slit pointing checks) and must be excluded so pypeit_setup
    does not create spurious configurations for them.

    Calibration frames (arc/, bias/, flat/) are always kept regardless of
    their GRISM keyword — on the old OSIRIS CCD, bias frames carry GRISM=OPEN
    in their headers even though they are genuine calibrations.

    Handles both .fits and .fits.gz files.
    Returns the path to the staging directory.
    """
    # Only filter GRISM=OPEN from these subdirs; calibration dirs are never filtered
    FILTER_SUBDIRS = {'object', 'stds'}

    raw_dir = ob_dir / '_raw_pypeit'
    raw_dir.mkdir(exist_ok=True)
    # Remove stale links from a previous run (both .fits and .fits.gz)
    for old in list(raw_dir.glob('*.fits')) + list(raw_dir.glob('*.fits.gz')):
        old.unlink()

    n_kept = n_excl = 0
    for sd in RAW_SUBDIRS:
        all_fits = sorted((ob_dir / sd).glob('*.fits')) + \
                   sorted((ob_dir / sd).glob('*.fits.gz'))
        for f in all_fits:
            exclude = False
            if sd in FILTER_SUBDIRS:
                try:
                    with fits.open(f) as h:
                        grism = str(h[0].header.get('GRISM', '')).strip().upper()
                except Exception:
                    grism = ''
                if grism == 'OPEN':
                    exclude = True

            if exclude:
                n_excl += 1
            else:
                link = raw_dir / f.name
                if not link.exists():
                    link.symlink_to(f.resolve())
                n_kept += 1

    print(f"  {n_kept} files kept, {n_excl} GRISM=OPEN acquisition images excluded")
    return raw_dir


def config_has_science(pypeit_file: Path) -> bool:
    """Return True if this config contains at least one science frame."""
    return bool(re.search(r'\|\s+science\s+\|', pypeit_file.read_text()))


# ── Main pipeline ──────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ob_dir',
                    help='OB directory (contains arc/, bias/, flat/, object/, stds/)')
    ap.add_argument('--spectrograph', default='gtc_osiris_plus',
                    choices=['gtc_osiris_plus', 'gtc_osiris'],
                    help='PypeIt spectrograph name (default: gtc_osiris_plus)')
    ap.add_argument('--maxnumber-sci', type=int, default=1,
                    help='Max objects to extract per exposure (default: 1)')
    ap.add_argument('--snr-thresh', type=float, default=None,
                    help='SNR threshold for object finding (PypeIt default: 10.0)')
    ap.add_argument('--find-fwhm', type=float, default=None,
                    help='FWHM (px) for object finding (PypeIt default: 5.0)')
    ap.add_argument('--find-min-max', type=int, nargs=2, default=None,
                    metavar=('MIN', 'MAX'),
                    help='Pixel range for object finding, e.g. --find-min-max 1200 2000')
    ap.add_argument('--trace-npoly', type=int, default=None,
                    help='Polynomial order for object trace fitting in PypeIt '
                         '(default: PypeIt built-in = 5; increase to 7-8 for curved traces)')
    ap.add_argument('--overwrite', action='store_true',
                    help='Pass -o to run_pypeit to overwrite existing Masters/Science')
    ap.add_argument('--no-interactive', action='store_true',
                    help='Skip all validation plots and run fully automatically')
    ap.add_argument('--skip-setup', action='store_true',
                    help='Skip pypeit_setup (reuse existing config dirs)')
    ap.add_argument('--skip-pypeit', action='store_true',
                    help='Skip run_pypeit (reuse existing Science/ outputs)')
    ap.add_argument('--skip-telluric', action='store_true',
                    help='Skip telluric correction step')
    return ap.parse_args()


def main():
    args = parse_args()
    interactive = not args.no_interactive

    ob_dir = Path(args.ob_dir).resolve()
    if not ob_dir.is_dir():
        sys.exit(f"ERROR: {ob_dir} is not a directory")
    for sd in RAW_SUBDIRS:
        if not (ob_dir / sd).is_dir():
            sys.exit(f"ERROR: missing expected subdirectory: {ob_dir / sd}")

    print(f"\n{'═'*60}")
    print(f"  OSIRIS reduction pipeline")
    print(f"  OB directory : {ob_dir}")
    print(f"  Spectrograph : {args.spectrograph}")
    print(f"  Interactive  : {interactive}")
    print(f"{'═'*60}")

    # ── 1. pypeit_setup ──────────────────────────────────────────────────
    if not args.skip_setup:
        print(f"\n{'─'*50}\nStep 1 — pypeit_setup\n{'─'*50}")
        raw_dir = create_filtered_rawdir(ob_dir)
        run(['pypeit_setup', '-s', args.spectrograph,
             '-r', str(raw_dir), '-b', '-c', 'all'], cwd=ob_dir)
    else:
        print(f"\nStep 1 — pypeit_setup  [skipped]")

    all_config_dirs = find_config_dirs(ob_dir)
    if not all_config_dirs:
        sys.exit("ERROR: no config directories found (expected gtc_osiris*_A, *_B …)")

    # Only process configs that contain at least one science frame
    config_dirs = [cd for cd in all_config_dirs
                   if config_has_science(cd / f'{cd.name}.pypeit')]
    skipped = [cd.name for cd in all_config_dirs if cd not in config_dirs]
    if skipped:
        print(f"\n  Skipping calibration-only configs: {skipped}")
    if not config_dirs:
        sys.exit("ERROR: no configuration contains a science frame.")
    print(f"\n  Configurations to reduce: {[d.name for d in config_dirs]}")
    pypeit_files = [cd / f'{cd.name}.pypeit' for cd in config_dirs]

    # ── 2. Edit .pypeit files ────────────────────────────────────────────
    print(f"\n{'─'*50}\nStep 2 — editing .pypeit files\n{'─'*50}")
    for pf in pypeit_files:
        print(f"\n  {pf.name}")
        if not has_bias(pf):
            print(f"    No bias frames — searching other configs…")
            for other in pypeit_files:
                if other != pf and has_bias(other):
                    if copy_bias_from(other, pf):
                        print(f"    Copied bias rows from {other.name}")
                        break
            else:
                print(f"    WARNING: could not find bias in any configuration!")
        patch_pypeit_params(pf, args.maxnumber_sci, args.snr_thresh,
                            args.find_fwhm, args.find_min_max, args.spectrograph,
                            args.trace_npoly)
        print(f"    Grating: {get_dispname(pf)}")

    # ── 3. run_pypeit ────────────────────────────────────────────────────
    if not args.skip_pypeit:
        print(f"\n{'─'*50}\nStep 3 — run_pypeit\n{'─'*50}")
        for config_dir, pf in zip(config_dirs, pypeit_files):
            cmd = ['run_pypeit', pf.name] + (['-o'] if args.overwrite else [])
            run(cmd, cwd=config_dir)
    else:
        print(f"\nStep 3 — run_pypeit  [skipped]")

    # ── Steps 4–7: per configuration ────────────────────────────────────
    det = 2 if args.spectrograph == 'gtc_osiris' else 1

    for config_dir, pf in zip(config_dirs, pypeit_files):
        grating     = get_dispname(pf)
        cfg_letter  = config_dir.name[-1]
        science_dir = config_dir / 'Science'

        print(f"\n{'═'*60}")
        print(f"  Configuration {cfg_letter}  ({grating})")
        print(f"{'═'*60}")

        if not science_dir.is_dir():
            print(f"  WARNING: {science_dir} not found — skipping.")
            continue

        spec1d_all = sorted(science_dir.glob('spec1d_*.fits'))
        std_files  = [f for f in spec1d_all if 'SPSTD_' in f.name]
        sci_files  = [f for f in spec1d_all if 'SPSTD_' not in f.name]

        if not std_files:
            print(f"  WARNING: no standard star (SPSTD_) in {science_dir} — skipping.")
            continue
        if not sci_files:
            print(f"  WARNING: no science spectra in {science_dir} — skipping.")
            continue

        std_spec1d = std_files[0]
        print(f"\n  Standard : {std_spec1d.name}")
        for sf in sci_files:
            print(f"  Science  : {sf.name}")

        # ── 3b. Trace validation ─────────────────────────────────────────
        if interactive:
            print(f"\n{'─'*50}\nStep 3b — trace validation\n{'─'*50}")
            need_rerun = False

            def _validate_one(spec1d_file, label):
                """Validate trace for one spec1d; return True if rerun needed."""
                spec1d_path = science_dir / spec1d_file.name
                raw_m = re.match(r'spec1d_(\d+)-', spec1d_file.name)
                spec2d_path = None
                if raw_m:
                    cands = sorted(science_dir.glob(f'spec2d_{raw_m.group(1)}-*.fits'))
                    spec2d_path = cands[0] if cands else None
                if spec2d_path is None or not spec2d_path.exists():
                    print(f"  WARNING: spec2d not found for {spec1d_file.name} – skipping plot")
                    return False
                ok, manual_params = validate_trace(spec2d_path, spec1d_path, label)
                if not ok and manual_params is not None:
                    spat, spec_px, fwhm = manual_params
                    set_manual_extraction(pf, spec1d_file.name, spat, spec_px, fwhm, det)
                    return True
                return False

            # Standard star
            print(f"\n  --- Standard star ---")
            if _validate_one(std_spec1d, f'STD {std_spec1d.name}'):
                need_rerun = True

            # Science frames
            print(f"\n  --- Science frames ---")
            for sci_spec1d in sci_files:
                target, _ = parse_spec1d_meta(sci_spec1d.name)
                if _validate_one(sci_spec1d, target or sci_spec1d.name):
                    need_rerun = True

            if need_rerun:
                print(f"\n  Re-running PypeIt with manual extraction…")
                run(['run_pypeit', pf.name, '-o'], cwd=config_dir)
                # Refresh file lists after re-run
                spec1d_all = sorted(science_dir.glob('spec1d_*.fits'))
                std_files  = [f for f in spec1d_all if 'SPSTD_' in f.name]
                sci_files  = [f for f in spec1d_all if 'SPSTD_' not in f.name]
                std_spec1d = std_files[0] if std_files else std_spec1d

        # ── 4. pypeit_sensfunc ───────────────────────────────────────────
        print(f"\n{'─'*50}\nStep 4 — pypeit_sensfunc\n{'─'*50}")
        sensfunc_raw  = config_dir / f'sensfunc_{cfg_letter}.fits'
        sensfunc_smth = config_dir / f'sensfunc_{cfg_letter}_smoothed.fits'
        run(['pypeit_sensfunc', str(science_dir / std_spec1d.name),
             '-o', str(sensfunc_raw)], cwd=config_dir)

        # ── 5. Smooth sensfunc ────────────────────────────────────────────
        print(f"\n{'─'*50}\nStep 5 — smooth sensfunc\n{'─'*50}")
        smooth_sensfunc(sensfunc_raw, sensfunc_smth)

        # ── 4b/5b. Sensfunc validation ───────────────────────────────────
        if interactive:
            print(f"\n{'─'*50}\nStep 5b — sensfunc validation\n{'─'*50}")
            validate_sensfunc(sensfunc_raw, sensfunc_smth)

        # ── 6. Flux calibration ──────────────────────────────────────────
        print(f"\n{'─'*50}\nStep 6 — flux calibration\n{'─'*50}")
        flux_txt = science_dir / f'flux_{cfg_letter}.txt'
        write_flux_file(flux_txt, [f.name for f in sci_files],
                        f'../sensfunc_{cfg_letter}_smoothed.fits')
        run(['pypeit_flux_calib', flux_txt.name], cwd=science_dir)

        # ── 6b. Export ASCII ─────────────────────────────────────────────
        print(f"\n{'─'*50}\nStep 6b — export ASCII\n{'─'*50}")
        exported = []
        for spec1d in sci_files:
            target, date = parse_spec1d_meta(spec1d.name)
            if target is None:
                print(f"  WARNING: could not parse target/date from {spec1d.name}")
                continue
            out_txt = ob_dir / f'{target}_{grating}_{date}.txt'
            export_ascii(science_dir / spec1d.name, out_txt, ob_dir)
            exported.append((out_txt, target, date))

        if args.skip_telluric:
            continue

        # ── 7. Telluric correction ────────────────────────────────────────
        print(f"\n{'─'*50}\nStep 7 — telluric correction\n{'─'*50}")
        lam_tell, tell_corr = build_tell_correction(science_dir / std_spec1d.name, grating)

        # ── 7b. Telluric validation ───────────────────────────────────────
        if interactive:
            print(f"\n{'─'*50}\nStep 7b — telluric correction validation\n{'─'*50}")
            lam_tell, tell_corr = validate_telluric(
                science_dir / std_spec1d.name, lam_tell, tell_corr,
                grating, [txt for txt, _, _ in exported])

        corrected = []
        for out_txt, target, date in exported:
            tc_txt = ob_dir / f'{target}_{grating}_{date}_tellcorr.txt'
            apply_tell_correction(out_txt, lam_tell, tell_corr, tc_txt)
            corrected.append(tc_txt)

        if len(corrected) > 1:
            combined = ob_dir / f'{exported[0][1]}_{grating}_{exported[0][2]}_combined_tellcorr.txt'
            combine_spectra(corrected, combined)

    # ── Summary + final plot ─────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Pipeline complete!  Output: {ob_dir}")
    print(f"{'═'*60}")
    txt_outputs = sorted(ob_dir.glob('*.txt'))
    if txt_outputs:
        print("  Final spectra:")
        for f in txt_outputs:
            print(f"    {f.name}")
    plot_final_spectra(ob_dir)


if __name__ == '__main__':
    main()
