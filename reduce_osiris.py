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
    TARGET_GRATING_TIMESTAMP_RAWID.txt           – flux-calibrated (wave, flux, err)
    TARGET_GRATING_TIMESTAMP_RAWID_tellcorr.txt  – telluric-corrected
    TARGET_GRATING_YYYYMMDDThhmmss_combined_tellcorr.txt – averaged over exposures (if >1)
    TARGET_R1000BR_YYYYMMDDThhmmss_merged_tellcorr.txt   – stitched R1000B+R1000R spectrum (if both exist)

All ASCII files include a commented header with observation metadata.
Pass --no-interactive to run the full pipeline without pausing for inspection.
"""

import argparse
import copy
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
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
MERGE_CENTER = 6200.0
MERGE_WIDTH = 40.0


# ── Shell helper ───────────────────────────────────────────────────────────

def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=True)


# ── pypeit_setup helpers ───────────────────────────────────────────────────

def find_config_dirs(ob_dir: Path):
    return sorted(p for p in ob_dir.iterdir()
                  if p.is_dir() and re.match(r'gtc_osiris.*_[A-Z]$', p.name))


def detect_spectrograph(ob_dir: Path) -> str:
    """Infer the correct PypeIt spectrograph from raw FITS headers."""
    for sd in ['object', 'stds', 'flat', 'arc', 'bias']:
        files = sorted((ob_dir / sd).glob('*.fits')) + sorted((ob_dir / sd).glob('*.fits.gz'))
        for f in files:
            try:
                h = fits.getheader(f)
            except Exception:
                continue
            detector = str(h.get('DETECTOR', '')).strip()
            detsize = str(h.get('DETSIZE', '')).strip()
            if detector == 'E2V CCD44_82_BI' or detsize == '[1:4096,1:4102]':
                return 'gtc_osiris'
            if detector == 'E2V 231-84-0-E74' or detsize == '[1:4096,1:4112]':
                return 'gtc_osiris_plus'
    return 'gtc_osiris_plus'


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


def enforce_arc_frametypes(pypeit_file: Path, ob_dir: Path):
    """Force files originating from ob_dir/arc to use frametype arc,tilt."""
    arc_names = {f.name for f in (ob_dir / 'arc').glob('*.fits')}
    arc_names.update(f.name for f in (ob_dir / 'arc').glob('*.fits.gz'))
    if not arc_names:
        return 0

    text = pypeit_file.read_text()
    if 'data read' not in text or 'data end' not in text:
        return 0

    pre, rest = text.split('data read', 1)
    block, post = rest.split('data end', 1)
    lines = block.splitlines(keepends=True)

    hdr_idx = None
    hdr_names = None
    for i, line in enumerate(lines):
        if 'filename' in line and 'frametype' in line and '|' in line:
            hdr_idx = i
            hdr_names = [c.strip() for c in line.rstrip('\n').split('|')]
            break
    if hdr_idx is None:
        return 0

    try:
        filename_idx = hdr_names.index('filename')
        frametype_idx = hdr_names.index('frametype')
    except ValueError:
        return 0

    changed = 0
    for i, line in enumerate(lines):
        if i == hdr_idx:
            continue
        stripped = line.rstrip('\n')
        if '|' not in stripped:
            continue
        parts = stripped.split('|')
        if len(parts) <= max(filename_idx, frametype_idx):
            continue

        filename = parts[filename_idx].strip()
        if filename not in arc_names:
            continue

        new_type = ' arc,tilt '
        if parts[frametype_idx].strip() == 'arc,tilt':
            continue
        parts[frametype_idx] = new_type
        lines[i] = '|'.join(parts) + '\n'
        changed += 1

    if changed:
        pypeit_file.write_text(pre + 'data read' + ''.join(lines) + 'data end' + post)
    return changed


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


def _set_manual_extraction_by_raw_id(pypeit_file: Path, raw_id: str,
                                     spat: float, spec_px: float, fwhm: float,
                                     det: int = 1):
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
    return True


def set_manual_extraction(pypeit_file: Path, spec1d_name: str,
                          spat: float, spec_px: float, fwhm: float, det: int = 1):
    """Set manual extraction for the exposure that produced ``spec1d_name``."""
    m = re.match(r'spec1d_(\d+)-', spec1d_name)
    if not m:
        print(f"  WARNING: cannot parse raw ID from {spec1d_name}")
        return
    raw_id = m.group(1)
    _set_manual_extraction_by_raw_id(pypeit_file, raw_id, spat, spec_px, fwhm, det)
    print(f"  Manual extraction set: det={det}, spat={spat:.1f}, "
          f"spec={spec_px:.1f}, fwhm={fwhm:.1f}")


def seed_science_manual_extractions(pypeit_file: Path, spat: float,
                                    spec_px: float = 1024.0, fwhm: float = 4.0,
                                    det: int = 1):
    """Seed manual extraction for every science row in a .pypeit file."""
    text = pypeit_file.read_text()
    science_raw_ids = []
    for line in text.splitlines():
        if not re.search(r'\|\s+science\s+\|', line):
            continue
        m = re.match(r'\s*(\d+)-', line)
        if m:
            science_raw_ids.append(m.group(1))

    if not science_raw_ids:
        return 0

    for raw_id in science_raw_ids:
        _set_manual_extraction_by_raw_id(pypeit_file, raw_id, spat, spec_px, fwhm, det)

    print(f"    Seeded manual extraction at spat={spat:.1f}, spec={spec_px:.1f}, "
          f"fwhm={fwhm:.1f} for {len(science_raw_ids)} science exposure(s)")
    return len(science_raw_ids)


# ── Observation metadata ───────────────────────────────────────────────────

def _hdr_get(hdr, *keys, default='UNKNOWN'):
    for k in keys:
        v = hdr.get(k)
        if v is not None:
            return v
    return default


def _find_raw_science_file(spec1d_fits: Path, ob_dir: Path):
    m = re.match(r'spec1d_(\d+-\d{8}-OSIRIS-OsirisLongSlitSpectroscopy)-', spec1d_fits.name)
    if not m:
        return None

    candidates = (
        sorted((ob_dir / 'object').glob(f'{m.group(1)}*.fits')) +
        sorted((ob_dir / 'object').glob(f'{m.group(1)}*.fits.gz'))
    )
    return candidates[0] if candidates else None


def read_obs_meta(spec1d_fits: Path, ob_dir: Path) -> dict:
    """Read observation metadata, preferring the raw science FITS header."""
    raw_hdr = None
    raw_path = _find_raw_science_file(spec1d_fits, ob_dir)
    if raw_path is not None:
        with fits.open(raw_path) as h:
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


def export_ascii(spec1d_fits: Path, out_txt: Path, ob_dir: Path,
                 preferred_spat: float = None):
    with fits.open(spec1d_fits) as h:
        ext = _select_spec1d_hdu(spec1d_fits, h, preferred_spat=preferred_spat)
        data = h[ext].data
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
    print(f"  ASCII spectrum → {out_txt.name}  (from HDU {ext})")


def _build_fluxcal_from_reference(custom_spec, ref_spec):
    def _scaled_output(prefix: str):
        wave = getattr(custom_spec, f'{prefix}_WAVE', None)
        counts = getattr(custom_spec, f'{prefix}_COUNTS', None)
        counts_sig = getattr(custom_spec, f'{prefix}_COUNTS_SIG', None)
        mask_1d = getattr(custom_spec, f'{prefix}_MASK', None)
        ref_wave = getattr(ref_spec, f'{prefix}_WAVE', None)
        ref_counts = getattr(ref_spec, f'{prefix}_COUNTS', None)
        ref_flam = getattr(ref_spec, f'{prefix}_FLAM', None)
        if wave is None or counts is None or ref_wave is None or ref_counts is None or ref_flam is None:
            return None

        wave = np.asarray(wave, dtype=float)
        counts = np.asarray(counts, dtype=float)
        counts_sig = np.asarray(counts_sig, dtype=float) if counts_sig is not None else np.zeros_like(counts)
        mask_arr = np.asarray(mask_1d, dtype=bool) if mask_1d is not None else np.isfinite(wave)
        ref_wave = np.asarray(ref_wave, dtype=float)
        ref_counts = np.asarray(ref_counts, dtype=float)
        ref_flam = np.asarray(ref_flam, dtype=float) * 1e-17

        good_ref = np.isfinite(ref_wave) & np.isfinite(ref_counts) & np.isfinite(ref_flam) & (ref_counts != 0)
        if np.sum(good_ref) < 2:
            return None

        scale_ref = ref_flam[good_ref] / ref_counts[good_ref]
        scale = np.interp(wave, ref_wave[good_ref], scale_ref, left=np.nan, right=np.nan)
        flux = counts * scale
        err = counts_sig * np.abs(scale)
        good = mask_arr & np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (wave > 0)
        if np.sum(good) < 2:
            return None
        return wave[good], flux[good], err[good], prefix

    extracted = _scaled_output('OPT')
    if extracted is None:
        extracted = _scaled_output('BOX')
    return extracted


def _find_matching_spec2d(spec1d_fits: Path):
    raw_m = re.match(r'spec1d_(\d+)-', spec1d_fits.name)
    if not raw_m:
        return None
    science_dir = spec1d_fits.parent
    cands = sorted(science_dir.glob(f'spec2d_{raw_m.group(1)}-*.fits'))
    return cands[0] if cands else None


def _load_spec2d_arrays(spec2d_path: Path):
    sciimg = skymodel = ivarmodel = waveimg = bpmmask = None
    with fits.open(spec2d_path) as h:
        for ext in h:
            if ext.data is None or np.ndim(ext.data) == 0:
                continue
            name = ext.name.upper()
            if 'SCIIMG' in name and sciimg is None:
                sciimg = np.asarray(ext.data, dtype=float)
            elif 'SKYMODEL' in name and skymodel is None:
                skymodel = np.asarray(ext.data, dtype=float)
            elif 'IVARMODEL' in name and ivarmodel is None:
                ivarmodel = np.asarray(ext.data, dtype=float)
            elif 'WAVEIMG' in name and waveimg is None:
                waveimg = np.asarray(ext.data, dtype=float)
            elif 'BPMMASK' in name and bpmmask is None:
                bpmmask = np.asarray(ext.data)

    if sciimg is None or ivarmodel is None or waveimg is None:
        raise ValueError(f'could not find SCIIMG/IVARMODEL/WAVEIMG in {spec2d_path.name}')
    if skymodel is None:
        skymodel = np.zeros_like(sciimg)

    mask = np.isfinite(sciimg) & np.isfinite(ivarmodel) & np.isfinite(waveimg) & (ivarmodel > 0)
    if bpmmask is not None:
        mask &= (bpmmask == 0)
    return sciimg, skymodel, ivarmodel, waveimg, mask


def _gaussian_profile_from_trace(trace: np.ndarray, nspat: int, fwhm: float):
    sigma = max(float(fwhm), 1.0) / 2.354820045
    spat = np.arange(nspat, dtype=float)[None, :]
    profile = np.exp(-0.5 * ((spat - trace[:, None]) / sigma)**2)
    norm = np.sum(profile, axis=1, keepdims=True)
    norm[norm <= 0] = 1.0
    return profile / norm


def _format_trace_points(points):
    return '; '.join(f'({x:.1f},{y:.1f})' for x, y in points)


def _estimate_supermanual_residual_sky(residual_img: np.ndarray, mask: np.ndarray,
                                       trace: np.ndarray, fwhm: float, box_radius: float):
    """Estimate a local residual sky/background model around a custom trace.

    This keeps PypeIt's global sky model and measures any residual continuum
    locally from sidebands around the user-drawn trace.
    """
    nspec, nspat = residual_img.shape
    inner = max(float(box_radius) + 2.0, 2.0 * float(fwhm))
    outer = inner + max(8.0, 2.0 * float(fwhm))
    spat = np.arange(nspat, dtype=float)
    resid_sky = np.zeros_like(residual_img, dtype=float)

    for i in range(nspec):
        center = float(trace[i])
        li0 = max(0, int(np.floor(center - outer)))
        li1 = max(0, int(np.floor(center - inner)))
        ri0 = min(nspat, int(np.ceil(center + inner)))
        ri1 = min(nspat, int(np.ceil(center + outer)))

        left_good = np.zeros(nspat, dtype=bool)
        right_good = np.zeros(nspat, dtype=bool)
        if li1 > li0:
            left_good[li0:li1] = True
        if ri1 > ri0:
            right_good[ri0:ri1] = True
        left_good &= mask[i]
        right_good &= mask[i]

        left_vals = residual_img[i, left_good]
        right_vals = residual_img[i, right_good]

        if left_vals.size == 0 and right_vals.size == 0:
            continue

        if left_vals.size > 0:
            left_med = float(np.nanmedian(left_vals))
            left_x = float(np.nanmedian(spat[left_good]))
        else:
            left_med = None
            left_x = None
        if right_vals.size > 0:
            right_med = float(np.nanmedian(right_vals))
            right_x = float(np.nanmedian(spat[right_good]))
        else:
            right_med = None
            right_x = None

        if left_med is not None and right_med is not None and right_x != left_x:
            resid_sky[i] = np.interp(spat, [left_x, right_x], [left_med, right_med],
                                     left=left_med, right=right_med)
        else:
            level = left_med if left_med is not None else right_med
            resid_sky[i].fill(float(level))

    return resid_sky, inner, outer


def export_ascii_localbg(spec1d_fits: Path, spec2d_fits: Path, out_txt: Path, ob_dir: Path,
                         preferred_spat: float = None, mode_label: str = 'LOCALBG'):
    from pypeit import specobjs
    from pypeit.core import extract as pypeit_extract

    with fits.open(spec1d_fits) as h:
        ext = _select_spec1d_hdu(spec1d_fits, h, preferred_spat=preferred_spat)
        ref_name = h[ext].name
        ext_hdr = h[ext].header
        fwhm = float(ext_hdr.get('FWHM', 4.0) or 4.0)
        box_radius = float(ext_hdr.get('BOX_RADIUS', max(fwhm, 4.0)) or max(fwhm, 4.0))

    sobjs = specobjs.SpecObjs.from_fitsfile(str(spec1d_fits), chk_version=False)
    ref_spec = None
    for i in range(sobjs.nobj):
        if sobjs[i].NAME == ref_name:
            ref_spec = copy.deepcopy(sobjs[i])
            break
    if ref_spec is None:
        raise ValueError(f'could not match selected HDU {ref_name} in {spec1d_fits.name}')

    sciimg, skymodel, ivarmodel, waveimg, mask = _load_spec2d_arrays(spec2d_fits)
    residual_img = sciimg - skymodel
    nspec, nspat = residual_img.shape
    trace = np.asarray(ref_spec.TRACE_SPAT, dtype=float)
    if trace.shape[0] != nspec:
        raise ValueError(f'localbg trace length mismatch for {spec2d_fits.name}')

    custom_spec = copy.deepcopy(ref_spec)
    custom_spec.TRACE_SPAT = np.clip(trace, 0, nspat - 1)
    custom_spec.trace_spec = np.arange(nspec, dtype=int)
    custom_spec.SPAT_PIXPOS = float(custom_spec.TRACE_SPAT[nspec // 2])
    custom_spec.FWHM = float(fwhm)
    custom_spec.BOX_RADIUS = float(box_radius)
    custom_spec.maskwidth = 4.0 * float(fwhm)

    residual_sky, bg_inner, bg_outer = _estimate_supermanual_residual_sky(
        residual_img, mask, custom_spec.TRACE_SPAT, fwhm, box_radius
    )
    total_sky = skymodel + residual_sky
    imgminsky = sciimg - total_sky

    pypeit_extract.extract_boxcar(imgminsky, ivarmodel, mask, waveimg, total_sky, custom_spec)

    oprof = _gaussian_profile_from_trace(custom_spec.TRACE_SPAT, nspat, fwhm)
    oprof *= mask.astype(float)
    try:
        pypeit_extract.extract_optimal(
            imgminsky, ivarmodel, mask, waveimg, total_sky, mask, oprof, custom_spec
        )
    except Exception as exc:
        print(f"  WARNING: optimal local-background extraction failed for {spec1d_fits.name} ({exc}); using boxcar only")
        custom_spec.OPT_WAVE = None

    extracted = _build_fluxcal_from_reference(custom_spec, ref_spec)
    if extracted is None:
        raise ValueError(f'could not build flux-calibrated local-background spectrum for {spec1d_fits.name}')

    wave, flux, err, extract_prefix = extracted
    meta = read_obs_meta(spec1d_fits, ob_dir)
    hdr = build_header(meta, extra_lines=[
        f'EXTRACT   : {mode_label}',
        f'TRACE_MODE: rerun PypeIt trace + local residual background ({extract_prefix})',
        f'TRACE_FWHM: {float(fwhm):.2f} px',
        f'BOX_RADIUS: {float(box_radius):.2f} px',
        f'SKY_BANDS : residual sidebands {bg_inner:.1f}-{bg_outer:.1f} px from trace',
        f'REF_HDU   : {ref_name}',
    ])
    np.savetxt(str(out_txt), np.column_stack([wave, flux, err]),
               fmt=['%.3f', '%e', '%e'], header=hdr, comments='# ')
    print(f"  ASCII spectrum → {out_txt.name}  (local background from HDU {ext})")


def export_ascii_supermanual(spec1d_fits: Path, spec2d_fits: Path, out_txt: Path, ob_dir: Path,
                             trace: np.ndarray, points: list, fwhm: float, box_radius: float,
                             preferred_spat: float = None):
    from pypeit import specobjs
    from pypeit.core import extract as pypeit_extract

    with fits.open(spec1d_fits) as h:
        ext = _select_spec1d_hdu(spec1d_fits, h, preferred_spat=preferred_spat)
        ref_name = h[ext].name

    sobjs = specobjs.SpecObjs.from_fitsfile(str(spec1d_fits), chk_version=False)
    ref_spec = None
    for i in range(sobjs.nobj):
        if sobjs[i].NAME == ref_name:
            ref_spec = copy.deepcopy(sobjs[i])
            break
    if ref_spec is None:
        raise ValueError(f'could not match selected HDU {ref_name} in {spec1d_fits.name}')

    sciimg, skymodel, ivarmodel, waveimg, mask = _load_spec2d_arrays(spec2d_fits)
    residual_img = sciimg - skymodel
    nspec, nspat = residual_img.shape
    full_trace = np.asarray(trace, dtype=float)
    if full_trace.shape[0] != nspec:
        raise ValueError(f'supermanual trace length mismatch for {spec2d_fits.name}')

    custom_spec = copy.deepcopy(ref_spec)
    custom_spec.TRACE_SPAT = np.clip(full_trace, 0, nspat - 1)
    custom_spec.trace_spec = np.arange(nspec, dtype=int)
    custom_spec.SPAT_PIXPOS = float(custom_spec.TRACE_SPAT[nspec // 2])
    custom_spec.hand_extract_flag = True
    custom_spec.FWHM = float(fwhm)
    custom_spec.BOX_RADIUS = float(box_radius)
    custom_spec.maskwidth = 4.0 * float(fwhm)
    custom_spec.set_name()

    residual_sky, bg_inner, bg_outer = _estimate_supermanual_residual_sky(
        residual_img, mask, custom_spec.TRACE_SPAT, fwhm, box_radius
    )
    total_sky = skymodel + residual_sky
    imgminsky = sciimg - total_sky

    pypeit_extract.extract_boxcar(imgminsky, ivarmodel, mask, waveimg, total_sky, custom_spec)

    oprof = _gaussian_profile_from_trace(custom_spec.TRACE_SPAT, nspat, fwhm)
    oprof *= mask.astype(float)
    try:
        pypeit_extract.extract_optimal(
            imgminsky, ivarmodel, mask, waveimg, total_sky, mask, oprof, custom_spec
        )
    except Exception as exc:
        print(f"  WARNING: optimal supermanual extraction failed for {spec1d_fits.name} ({exc}); using boxcar only")
        custom_spec.OPT_WAVE = None

    extracted = _build_fluxcal_from_reference(custom_spec, ref_spec)
    if extracted is None:
        raise ValueError(f'could not build flux-calibrated supermanual spectrum for {spec1d_fits.name}')

    wave, flux, err, extract_prefix = extracted
    flux_offset = 0.0
    finite_flux = flux[np.isfinite(flux)]
    if finite_flux.size and np.nanmin(finite_flux) <= 0:
        min_flux = float(np.nanmin(finite_flux))
        eps = max(float(np.nanmedian(err[np.isfinite(err)])) * 0.01 if np.any(np.isfinite(err)) else 0.0,
                  abs(min_flux) * 1e-6, 1e-30)
        flux_offset = -min_flux + eps
        flux = flux + flux_offset

    meta = read_obs_meta(spec1d_fits, ob_dir)
    extra_lines = [
        'EXTRACT   : SUPERMANUAL',
        f'TRACE_MODE: piecewise linear through user clicks ({extract_prefix})',
        f'TRACE_FWHM: {float(fwhm):.2f} px',
        f'BOX_RADIUS: {float(box_radius):.2f} px',
        f'SKY_BANDS : residual sidebands {bg_inner:.1f}-{bg_outer:.1f} px from trace',
        f'TRACE_PTS : {_format_trace_points(points)}',
    ]
    if flux_offset != 0.0:
        extra_lines.append(f'FLUX_OFFSET: +{flux_offset:.6e} added to make spectrum positive')
    hdr = build_header(meta, extra_lines=extra_lines)
    np.savetxt(str(out_txt), np.column_stack([wave, flux, err]),
               fmt=['%.3f', '%e', '%e'], header=hdr, comments='# ')
    print(f"  ASCII spectrum → {out_txt.name}  (supermanual trace)")


def parse_spec1d_meta(filename: str):
    m = re.search(
        r'spec1d_(\d+)-(\d{8})-OSIRIS-OsirisLongSlitSpectroscopy-(.+?)_OSIRIS_'
        r'(\d{8}T\d{6})(?:\.\d+)?',
        os.path.basename(filename))
    if not m:
        return None
    return {
        'raw_id': m.group(1),
        'archive_date': m.group(2),
        'target': m.group(3),
        'timestamp': m.group(4),
    }


def _safe_stem_component(value: str) -> str:
    value = re.sub(r'\s+', '_', value.strip())
    value = re.sub(r'[^A-Za-z0-9._+-]+', '_', value)
    return value.strip('_') or 'UNKNOWN'


def _dateobs_to_token(date_obs) -> str:
    if date_obs in (None, 'UNKNOWN'):
        return None

    date_str = str(date_obs).strip()
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%Y%m%dT%H%M%S')
    except ValueError:
        match = re.search(r'(\d{8})T(\d{6})', date_str)
        if match:
            return f'{match.group(1)}T{match.group(2)}'
    return None


def build_output_info(spec1d_fits: Path, ob_dir: Path, grating: str) -> dict:
    meta = read_obs_meta(spec1d_fits, ob_dir)
    parsed = parse_spec1d_meta(spec1d_fits.name)

    target_name = (
        parsed['target'] if parsed is not None and parsed.get('target')
        else meta.get('object', 'UNKNOWN')
    )
    target_token = _safe_stem_component(target_name)
    timestamp = (
        _dateobs_to_token(meta.get('date_obs'))
        or (parsed.get('timestamp') if parsed is not None else None)
        or (parsed.get('archive_date') if parsed is not None else None)
        or 'UNKNOWN'
    )
    raw_id = parsed.get('raw_id') if parsed is not None else 'UNKNOWN'
    stem = f'{target_token}_{grating}_{timestamp}_{raw_id}'

    return {
        'meta': meta,
        'target': target_token,
        'grating': grating,
        'timestamp': timestamp,
        'raw_id': raw_id,
        'stem': stem,
        'group_key': (target_token, grating),
    }


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


def _header_value(path: Path, key: str):
    prefix = f'# {key}'
    for line in _read_header_lines(path):
        if line.startswith(prefix):
            parts = line.split(':', 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


def _iso_to_datetime(date_str: str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except ValueError:
        return None


def _header_exposure_count(path: Path) -> int:
    value = _header_value(path, 'N_EXPOSURES')
    if value is None:
        return 1
    try:
        return max(1, int(value))
    except ValueError:
        return 1


def _average_datetime(paths) -> datetime | None:
    dts = []
    weights = []
    for path in paths:
        dt = _iso_to_datetime(_header_value(path, 'DATE-OBS'))
        if dt is None:
            continue
        dts.append(dt)
        weights.append(_header_exposure_count(path))
    if not dts:
        return None

    base = dts[0]
    total_weight = float(sum(weights))
    weighted_seconds = sum(
        (dt - base).total_seconds() * weight
        for dt, weight in zip(dts, weights)
    ) / total_weight
    return base + timedelta(seconds=weighted_seconds)


def _average_datetime_token(paths):
    avg_dt = _average_datetime(paths)
    if avg_dt is None:
        return None
    return avg_dt.strftime('%Y%m%dT%H%M%S')


def _average_datetime_iso(paths):
    avg_dt = _average_datetime(paths)
    if avg_dt is None:
        return None
    return avg_dt.isoformat(timespec='seconds')


def _interp_with_nan(x_new, x_old, y_old):
    """Interpolate onto ``x_new`` and mark values outside the input range as NaN."""
    y_new = np.interp(x_new, x_old, y_old)
    outside = (x_new < x_old[0]) | (x_new > x_old[-1])
    y_new[outside] = np.nan
    return y_new


def _estimate_scale_factor(wave_ref, flux_ref, wave_other, flux_other,
                           center: float, width: float):
    w0 = center - width / 2.0
    w1 = center + width / 2.0
    mask = (wave_ref >= w0) & (wave_ref <= w1)
    if np.count_nonzero(mask) < 5:
        raise ValueError(f'not enough reference samples in merge window {w0:.1f}-{w1:.1f} A')

    other_on_ref = _interp_with_nan(wave_ref[mask], wave_other, flux_other)
    valid = np.isfinite(other_on_ref) & np.isfinite(flux_ref[mask]) & (other_on_ref != 0)
    if np.count_nonzero(valid) < 5:
        raise ValueError(f'not enough overlapping data in merge window {w0:.1f}-{w1:.1f} A')

    ratio = flux_ref[mask][valid] / other_on_ref[valid]
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        raise ValueError(f'cannot compute scale factor in merge window {w0:.1f}-{w1:.1f} A')
    return float(np.nanmedian(ratio))


def _parse_spec1d_summary_table(spec1d_txt: Path) -> list:
    """Parse the companion spec1d summary table written by PypeIt."""
    if not spec1d_txt.exists():
        return []

    lines = spec1d_txt.read_text().splitlines()
    if not lines:
        return []

    header = None
    rows = []
    for line in lines:
        if not line.startswith('|'):
            continue
        parts = [p.strip() for p in line.strip().strip('|').split('|')]
        if header is None:
            header = parts
            continue
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def _row_spat_pixpos(row: dict):
    value = row.get('spat_pixpos')
    if value in (None, ''):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _select_spec1d_hdu(spec1d_fits: Path, hdul=None, preferred_spat: float = None):
    """Choose the object HDU to use from a multi-object spec1d FITS file.

    Prefer the object flagged as ``manual_extract=True`` in the companion
    summary table.  Otherwise, if ``preferred_spat`` is given, choose the
    extracted object whose spatial position is closest to that column.
    Fall back to the first extracted object HDU.
    """
    close_hdul = False
    if hdul is None:
        hdul = fits.open(spec1d_fits)
        close_hdul = True

    try:
        object_hdus = {}
        first_idx = None
        for idx, hdu in enumerate(hdul[1:], start=1):
            if hdu.data is None or not hasattr(hdu.data, 'columns'):
                continue
            if 'TRACE_SPAT' not in hdu.columns.names:
                continue
            trace = np.asarray(hdu.data['TRACE_SPAT'], dtype=float)
            object_hdus[hdu.name] = {
                'idx': idx,
                'trace_median': float(np.nanmedian(trace)),
            }
            if first_idx is None:
                first_idx = idx

        if first_idx is None:
            raise ValueError(f'no extracted object HDUs found in {spec1d_fits.name}')

        summary_rows = _parse_spec1d_summary_table(spec1d_fits.with_suffix('.txt'))
        for row in summary_rows:
            if row.get('manual_extract', '').lower() == 'true':
                name = row.get('name')
                if name in object_hdus:
                    return object_hdus[name]['idx']

        if preferred_spat is not None:
            summary_matches = []
            for row in summary_rows:
                name = row.get('name')
                spat_pixpos = _row_spat_pixpos(row)
                if name not in object_hdus or spat_pixpos is None:
                    continue
                summary_matches.append((
                    abs(spat_pixpos - preferred_spat),
                    spat_pixpos,
                    object_hdus[name]['idx'],
                ))
            if summary_matches:
                summary_matches.sort(key=lambda item: item[0])
                best = summary_matches[0]
                print(f"  Preferring extracted object near spat={preferred_spat:.1f}: "
                      f"selected spat={best[1]:.1f} (HDU {best[2]}) from {spec1d_fits.name}")
                return best[2]

            hdu_matches = sorted(
                (
                    abs(info['trace_median'] - preferred_spat),
                    info['trace_median'],
                    info['idx'],
                )
                for info in object_hdus.values()
            )
            if hdu_matches:
                best = hdu_matches[0]
                print(f"  Preferring extracted object near spat={preferred_spat:.1f}: "
                      f"selected trace median={best[1]:.1f} (HDU {best[2]}) from {spec1d_fits.name}")
                return best[2]

        return first_idx
    finally:
        if close_hdul:
            hdul.close()


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
        ext = _select_spec1d_hdu(std_spec1d, h)
        lam  = h[ext].data['OPT_WAVE'].astype(float)
        flux = h[ext].data['OPT_COUNTS'].astype(float)
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
    wave = arrays[0][0]

    flux_stack = []
    err_stack = []
    resampled = []
    for path, (wave_i, flux_i, err_i) in zip(txt_files, arrays):
        if wave_i.shape != wave.shape or not np.allclose(wave, wave_i, rtol=0.0, atol=1e-3):
            flux_i = _interp_with_nan(wave, wave_i, flux_i)
            err_i = _interp_with_nan(wave, wave_i, err_i)
            resampled.append(path.name)
        flux_stack.append(flux_i)
        err_stack.append(err_i)

    flux_stack = np.vstack(flux_stack)
    err_stack = np.vstack(err_stack)
    valid = np.isfinite(flux_stack) & np.isfinite(err_stack)
    nvalid = np.sum(valid, axis=0)
    if np.any(nvalid == 0):
        raise ValueError(f"no overlapping wavelength coverage among spectra for {out_txt.name}")

    flux_sum = np.nansum(np.where(valid, flux_stack, 0.0), axis=0)
    err_sum_sq = np.nansum(np.where(valid, err_stack**2, 0.0), axis=0)
    flux_avg = flux_sum / nvalid
    err_avg = np.sqrt(err_sum_sq) / nvalid
    n = len(arrays)

    if resampled:
        print(f"  Resampled to common grid before combining: {', '.join(resampled)}")

    orig_hdr = _read_header_lines(txt_files[0])
    hdr_body = [l.lstrip('# ') for l in orig_hdr]
    avg_iso = _average_datetime_iso(txt_files)
    if avg_iso is not None:
        hdr_body = [l for l in hdr_body if not l.startswith('DATE-OBS')]
        hdr_body.insert(4, f'DATE-OBS  : {avg_iso}')
    hdr_body = [l for l in hdr_body if not l.startswith('N_EXPOSURES')]
    tell_idx = next((i for i, line in enumerate(hdr_body)
                     if line.startswith('TELLURIC')), len(hdr_body))
    hdr_body.insert(tell_idx, f'N_EXPOSURES: {n}')
    hdr_str = '\n'.join(hdr_body)
    np.savetxt(str(out_txt), np.column_stack([wave, flux_avg, err_avg]),
               fmt=['%.3f', '%e', '%e'], header=hdr_str, comments='# ')
    print(f"  Combined ({n} exposures) → {out_txt.name}")


def merge_blue_red_spectra(blue_txt: Path, red_txt: Path, out_txt: Path,
                           center: float = MERGE_CENTER, width: float = MERGE_WIDTH):
    wave_b, flux_b, err_b = _read_ascii3(blue_txt)
    wave_r, flux_r, err_r = _read_ascii3(red_txt)

    lo = max(wave_b[0], wave_r[0], center - width / 2.0)
    hi = min(wave_b[-1], wave_r[-1], center + width / 2.0)
    if hi <= lo:
        raise ValueError(
            f'no overlap around {center:.1f} A to merge {blue_txt.name} and {red_txt.name}'
        )

    scale = _estimate_scale_factor(wave_b, flux_b, wave_r, flux_r, center, width)
    flux_r_scaled = flux_r * scale
    err_r_scaled = err_r * abs(scale)

    keep_b = wave_b <= center
    keep_r = wave_r > center
    wave = np.concatenate([wave_b[keep_b], wave_r[keep_r]])
    flux = np.concatenate([flux_b[keep_b], flux_r_scaled[keep_r]])
    err = np.concatenate([err_b[keep_b], err_r_scaled[keep_r]])

    hdr_lines = _read_header_lines(blue_txt)
    hdr_body = [l.lstrip('# ') for l in hdr_lines]
    hdr_body = [l for l in hdr_body if not l.startswith('GRISM')]
    avg_iso = _average_datetime_iso([blue_txt, red_txt])
    if avg_iso is not None:
        hdr_body = [l for l in hdr_body if not l.startswith('DATE-OBS')]
        hdr_body.insert(4, f'DATE-OBS  : {avg_iso}')
    n_total = _header_exposure_count(blue_txt) + _header_exposure_count(red_txt)
    hdr_body = [l for l in hdr_body if not l.startswith('N_EXPOSURES')]
    hdr_body.append('GRISM     : R1000BR')
    hdr_body.append(f'N_EXPOSURES: {n_total}')
    hdr_body.append(f'MERGE_REF : {blue_txt.name}')
    hdr_body.append(f'MERGE_ADD : {red_txt.name}')
    hdr_body.append(f'MERGE_WIN : {center - width/2.0:.1f}-{center + width/2.0:.1f} A')
    hdr_body.append(f'MERGE_SPLIT: {center:.1f} A')
    hdr_body.append(f'MERGE_SCALE_R: {scale:.6f}')
    hdr_str = '\n'.join(hdr_body)

    np.savetxt(str(out_txt), np.column_stack([wave, flux, err]),
               fmt=['%.3f', '%e', '%e'], header=hdr_str, comments='# ')
    print(f"  Merged B+R spectrum → {out_txt.name}  (R scale {scale:.3f})")


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


def _manual_preview_trace(img: np.ndarray, spat: float, fwhm: float):
    """Return a preview trace anchored on a user-supplied spatial position."""
    nspec, _ = img.shape
    anchor = np.full(nspec, float(spat), dtype=float)
    half_window = max(20, int(np.ceil(3.0 * max(fwhm, 1.0))))
    custom = _compute_trace(img, anchor, half_window=half_window)
    if custom is not None:
        return custom[2]
    return anchor


def _trace_from_clicked_points(points, nspec: int, nspat: int):
    """Build a full trace by linearly interpolating user-clicked nodes."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return None, None
    in_bounds = (
        np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) &
        (pts[:, 0] >= 0) & (pts[:, 0] <= nspat - 1) &
        (pts[:, 1] >= 0) & (pts[:, 1] <= nspec - 1)
    )
    pts = pts[in_bounds]
    if pts.shape[0] < 2:
        return None, None

    order = np.argsort(pts[:, 1])
    pts = pts[order]
    y = pts[:, 1]
    x = pts[:, 0]
    keep = np.concatenate(([True], np.diff(y) > 1e-6))
    pts = pts[keep]
    y = pts[:, 1]
    x = pts[:, 0]
    if pts.shape[0] < 2:
        return None, None

    spec = np.arange(nspec, dtype=float)
    trace = np.interp(spec, y, x, left=x[0], right=x[-1])
    trace = np.clip(trace, 0, nspat - 1)
    return pts, trace


def _plot_trace_validation(fig, ax2d, axsp, img: np.ndarray, target: str,
                           spec1d_name: str, trace_spat=None, custom=None,
                           preview_trace=None, shifted_trace=None,
                           super_trace=None,
                           super_points=None):
    """Populate the trace-validation figure."""
    nspec, nspat = img.shape
    fig.suptitle(f'Trace validation – {target}  ({spec1d_name})', fontsize=11)

    vmin, vmax = np.nanpercentile(img, [2, 98])
    ax2d.imshow(img, origin='lower', aspect='auto', cmap='RdYlBu_r',
                vmin=vmin, vmax=vmax,
                extent=[0, nspat - 1, 0, nspec - 1])

    if trace_spat is not None:
        ax2d.plot(trace_spat, np.arange(nspec), color='lime',
                  lw=1.5, label=f'PypeIt trace (spat≈{np.nanmedian(trace_spat):.0f})')

    if custom is not None:
        sample_rows, sample_peaks, full_trace = custom
        ax2d.scatter(sample_peaks, sample_rows, s=12, color='orange',
                     zorder=5, label='peak samples')
        ax2d.plot(full_trace, np.arange(nspec), color='orange',
                  lw=1.5, ls='--', label=f'custom trace (spat≈{np.nanmedian(full_trace):.0f})')

    if preview_trace is not None:
        ax2d.plot(preview_trace, np.arange(nspec), color='magenta',
                  lw=1.7, label=f'manual preview (spat≈{np.nanmedian(preview_trace):.0f})')
    if shifted_trace is not None:
        ax2d.plot(shifted_trace, np.arange(nspec), color='deepskyblue',
                  lw=1.7, ls='-.',
                  label=f'shifted trace (spat≈{np.nanmedian(shifted_trace):.0f})')

    if super_points is not None:
        pts = np.asarray(super_points, dtype=float)
        ax2d.scatter(pts[:, 0], pts[:, 1], s=18, color='cyan',
                     zorder=6, label='supermanual clicks')
    if super_trace is not None:
        ax2d.plot(super_trace, np.arange(nspec), color='cyan',
                  lw=1.8, ls='-.', label=f'supermanual trace (spat≈{np.nanmedian(super_trace):.0f})')

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
        med_custom = int(np.nanmedian(custom[2]))
        axsp.axhline(med_custom, color='orange', lw=2, ls='--',
                     label=f'custom spat={med_custom}')
    if preview_trace is not None:
        med_preview = int(np.nanmedian(preview_trace))
        axsp.axhline(med_preview, color='magenta', lw=2, ls='-.',
                     label=f'manual spat={med_preview}')
    if shifted_trace is not None:
        med_shifted = int(np.nanmedian(shifted_trace))
        axsp.axhline(med_shifted, color='deepskyblue', lw=2, ls='-.',
                     label=f'shifted spat={med_shifted}')
    if super_trace is not None:
        med_super = int(np.nanmedian(super_trace))
        axsp.axhline(med_super, color='cyan', lw=2, ls='-.',
                     label=f'supermanual spat={med_super}')
    axsp.legend(fontsize=9)
    axsp.set_xlabel('Median counts')
    axsp.set_title('Spatial profile')
    axsp.yaxis.tick_right()


def validate_trace(spec2d_path: Path, spec1d_path: Path, target: str,
                   preferred_spat: float = None, allow_supermanual: bool = False) -> tuple:
    """Show 2D spectrum + spatial profile with extracted trace.

    Returns (action, payload), where action is one of:
      - 'accept'
      - 'rerun_manual'
      - 'rerun_manual_localbg'
      - 'supermanual'
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
    fwhm_guess = 4.0
    with fits.open(spec1d_path) as h:
        ext = _select_spec1d_hdu(spec1d_path, h, preferred_spat=preferred_spat)
        cols = [c.name.upper() for c in h[ext].columns]
        key  = next((c for c in cols if 'TRACE_SPAT' in c), None)
        if key:
            trace_spat = h[ext].data[key].astype(float)
        if 'FWHMFIT' in cols:
            fwhm_fit = np.asarray(h[ext].data['FWHMFIT'], dtype=float)
            good = np.isfinite(fwhm_fit) & (fwhm_fit > 0)
            if np.any(good):
                fwhm_guess = float(np.nanmedian(fwhm_fit[good]))

    # ── plot ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f'Trace validation – {target}  ({spec1d_path.name})', fontsize=11)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax2d = fig.add_subplot(gs[0])
    axsp = fig.add_subplot(gs[1], sharey=None)

    custom = _compute_trace(img, trace_spat)
    _plot_trace_validation(fig, ax2d, axsp, img, target, spec1d_path.name,
                           trace_spat=trace_spat, custom=custom)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    has_custom = custom is not None
    has_shift = trace_spat is not None
    if has_custom and has_shift and allow_supermanual:
        prompt = f"\n  [{target}] Which trace to use? [g]reen, [o]range, s[h]ift, [m]anual or [s]upermanual? "
    elif has_custom and has_shift:
        prompt = f"\n  [{target}] Which trace to use? [g]reen, [o]range, s[h]ift or [m]anual? "
    elif has_shift and allow_supermanual:
        prompt = f"\n  [{target}] Which trace to use? [g]reen, s[h]ift, [m]anual or [s]upermanual? "
    elif has_shift:
        prompt = f"\n  [{target}] Which trace to use? [g]reen, s[h]ift or [m]anual? "
    else:
        prompt = f"\n  [{target}] Which trace to use? [g]reen or [m]anual? "

    ans = _ask(prompt, default='g')
    plt.close('all')

    # ── green: accept PypeIt trace as-is ──────────────────────────────────
    if ans in ('g', '', 'k', 'keep', 'skip'):
        return 'accept', None
    if ans == 's' and not allow_supermanual:
        return 'accept', None

    # ── orange: use custom peak-finding trace ──────────────────────────────
    if ans == 'o' and has_custom:
        full_trace = custom[2]
        spat = float(np.nanmedian(full_trace))
        print(f"  Using custom (orange) trace preview around spat={spat:.1f}; "
              "continuing without re-running PypeIt")
        return 'accept', None

    # ── shift: apply a constant spatial shift to the green trace ───────────
    if ans == 'h' and has_shift:
        try:
            shift_in = input("    Shift in spatial pixels (positive moves right): ").strip()
            shift = float(shift_in)
            sp_def = nspec // 2
            sp_in = input(f"    Spectral pixel to anchor (Enter = {sp_def}): ").strip()
            spec_px = float(sp_in) if sp_in else float(sp_def)
            fw_in = input(f"    FWHM in pixels (Enter = {fwhm_guess:.1f}): ").strip()
            fwhm = float(fw_in) if fw_in else float(fwhm_guess)
        except ValueError:
            print("  Invalid shift input")
            return 'accept', None

        shifted_trace = np.clip(trace_spat + shift, 0, nspat - 1)
        fig = plt.figure(figsize=(16, 6))
        gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
        ax2d = fig.add_subplot(gs[0])
        axsp = fig.add_subplot(gs[1], sharey=None)
        _plot_trace_validation(fig, ax2d, axsp, img, target, spec1d_path.name,
                               trace_spat=trace_spat, custom=custom,
                               shifted_trace=shifted_trace)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
        confirm = _ask("  Use this shifted trace preview? [y]/n: ", default='y')
        plt.close('all')
        if confirm == 'n':
            print("  Shifted trace cancelled")
            return 'accept', None

        spec_idx = int(np.clip(round(spec_px), 0, nspec - 1))
        spat = float(shifted_trace[spec_idx])
        return 'rerun_manual_localbg', (spat, float(spec_idx), fwhm)

    # ── supermanual: user clicks a traced path ─────────────────────────────
    if ans == 's' and allow_supermanual:
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
        ax2d = fig.add_subplot(gs[0])
        axsp = fig.add_subplot(gs[1], sharey=None)
        _plot_trace_validation(fig, ax2d, axsp, img, target, spec1d_path.name,
                               trace_spat=trace_spat, custom=custom)
        plt.tight_layout()
        print("  Click points along the desired trace from blue to red.")
        print("  Press Enter when you are done.")
        plt.show(block=False)
        plt.pause(0.5)
        clicked = plt.ginput(n=-1, timeout=0, show_clicks=True)
        plt.close('all')

        points, super_trace = _trace_from_clicked_points(clicked, nspec, nspat)
        if super_trace is None:
            print("  Need at least two clicked points for supermanual tracing")
            return 'accept', None

        try:
            fw_in = input("    FWHM in pixels for supermanual extraction (Enter = 4.0): ").strip()
            fwhm = float(fw_in) if fw_in else 4.0
            br_def = max(float(fwhm), 4.0)
            br_in = input(f"    Boxcar radius in pixels (Enter = {br_def:.1f}): ").strip()
            box_radius = float(br_in) if br_in else br_def
        except ValueError:
            print("  Invalid supermanual extraction input")
            return 'accept', None

        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
        ax2d = fig.add_subplot(gs[0])
        axsp = fig.add_subplot(gs[1], sharey=None)
        _plot_trace_validation(fig, ax2d, axsp, img, target, spec1d_path.name,
                               trace_spat=trace_spat, custom=custom,
                               super_trace=super_trace, super_points=points)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
        confirm = _ask("  Use this supermanual trace? [y]/n: ", default='y')
        plt.close('all')
        if confirm == 'n':
            print("  Supermanual trace cancelled")
            return 'accept', None

        payload = {
            'trace': super_trace,
            'points': [tuple(map(float, p)) for p in points],
            'fwhm': float(fwhm),
            'box_radius': float(box_radius),
        }
        return 'supermanual', payload

    # ── manual: user types in the spatial position ─────────────────────────
    print("  Enter the position of the correct object (read from the plot):")
    try:
        spat    = float(input("    Spatial pixel : "))
        sp_def  = nspec // 2
        sp_in   = input(f"    Spectral pixel (Enter = {sp_def}): ").strip()
        spec_px = float(sp_in) if sp_in else float(sp_def)
        fw_in   = input(f"    FWHM in pixels (Enter = {fwhm_guess:.1f}): ").strip()
        fwhm    = float(fw_in) if fw_in else float(fwhm_guess)
    except ValueError:
        print("  Invalid input – skipping manual extraction")
        return 'accept', None

    preview_trace = _manual_preview_trace(img, spat, fwhm)
    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax2d = fig.add_subplot(gs[0])
    axsp = fig.add_subplot(gs[1], sharey=None)
    _plot_trace_validation(fig, ax2d, axsp, img, target, spec1d_path.name,
                           trace_spat=trace_spat, custom=custom,
                           preview_trace=preview_trace)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
    confirm = _ask("  Use this manual trace preview? [y]/n: ", default='y')
    plt.close('all')
    if confirm == 'n':
        print("  Manual extraction cancelled")
        return 'accept', None

    return 'rerun_manual', (spat, spec_px, fwhm)


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
        ext = _select_spec1d_hdu(std_spec1d, h)
        lam_std  = h[ext].data['OPT_WAVE'].astype(float)
        flux_std = h[ext].data['OPT_COUNTS'].astype(float)
        # Read extraction metadata from the spec1d header
        hdr0 = h[0].header
        hdr_obj = h[ext].header
        obj_name  = hdr0.get('TARGET',  hdr0.get('OBJECT', '?'))
        spat_pixl = hdr_obj.get('SPAT_PIXPOS', hdr0.get('SPAT_PIX', None))
        snr       = hdr_obj.get('OPT_SNR', hdr0.get('OPT_SNR', None))

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

    # Title shows exactly which file / object / spatial position was extracted
    std_info = f'source: {Path(std_spec1d).name}'
    if obj_name and obj_name != '?':
        std_info += f'  |  object: {obj_name}'
    if spat_pixl is not None:
        std_info += f'  |  spat_pix: {spat_pixl:.1f}'
    if snr is not None:
        std_info += f'  |  S/N: {snr:.1f}'
    ax.set_title(f'Standard star OPT_COUNTS\n{std_info}\n'
                 f'(shaded = telluric band, hatched = continuum windows)',
                 fontsize=9)
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
                if re.search(r'_R\d{4}[BR]_\d{8}T\d{6}_\d+\.txt$', f.name)]
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

        label = txt.stem
        if label.endswith('_tellcorr'):
            label = label[:-9]
        if label.endswith('_combined'):
            label = label[:-9] + '  (combined)'
        label = label.replace('_', ' ')

        merge_ref = _header_value(txt, 'MERGE_REF')
        merge_add = _header_value(txt, 'MERGE_ADD')
        merge_scale = _header_value(txt, 'MERGE_SCALE_R')

        if merge_ref and merge_add and merge_scale:
            ref_path = ob_dir / merge_ref
            add_path = ob_dir / merge_add
            try:
                wave_b, flux_b, err_b = _read_ascii3(ref_path)
                wave_r, flux_r, err_r = _read_ascii3(add_path)
                scale_r = float(merge_scale)

                ax.plot(wave_b, flux_b, color='royalblue', lw=1.0, alpha=0.65,
                        label='R1000B original')
                ax.plot(wave_r, flux_r * scale_r, color='darkorange', lw=1.0, alpha=0.65,
                        label=f'R1000R scaled x {scale_r:.3f}')
                ax.plot(wave, flux, color='black', lw=1.4, alpha=0.95,
                        label='Merged spectrum')
                ax.fill_between(wave, flux - err, flux + err,
                                color='0.35', alpha=0.18)
            except Exception as e:
                print(f"  WARNING: could not overlay merged inputs for {txt.name} ({e})")
                ax.plot(wave, flux, color='steelblue', lw=1.2, label=label)
                ax.fill_between(wave, flux - err, flux + err,
                                color='steelblue', alpha=0.2)
        else:
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

def infer_ccdsum(header) -> str | None:
    """Infer CCDSUM from alternate OSIRIS header cards when missing."""
    raw = header.get('CCDSUM')
    if raw is not None:
        raw = str(raw).strip()
        if raw and raw.upper() != 'NONE':
            return raw

    p_binning = str(header.get('P_BINNING', '')).strip()
    match = re.search(r'(\d+)\s*x\s*(\d+)', p_binning)
    if match:
        return f'{match.group(1)} {match.group(2)}'

    return None


def infer_rdnoise(header) -> float | None:
    """Infer RDNOISE for OSIRIS+ headers that omit the standard card."""
    raw = header.get('RDNOISE')
    if raw is not None:
        try:
            return float(raw)
        except Exception:
            pass

    detector = str(header.get('DETECTOR', '')).strip()
    detsize = str(header.get('DETSIZE', '')).strip()
    if detector == 'E2V 231-84-0-E74' or detsize == '[1:4096,1:4112]':
        return 4.3
    if detector == 'E2V CCD44_82_BI' or detsize == '[1:4096,1:4102]':
        return 4.5

    return None


def stage_for_pypeit(src: Path, dst: Path) -> bool:
    """Stage one FITS file into _raw_pypeit, patching missing metadata if needed.

    Returns True when the staged file was written as a patched copy, False when
    a simple symlink was sufficient.
    """
    try:
        with fits.open(src) as hdul:
            header = hdul[0].header
            ccdsum = infer_ccdsum(header)
            rdnoise = infer_rdnoise(header)
            needs_ccdsum = ccdsum is not None and not str(header.get('CCDSUM', '')).strip()
            needs_rdnoise = rdnoise is not None and header.get('RDNOISE') is None
            needs_patch = needs_ccdsum or needs_rdnoise
            if not needs_patch:
                dst.symlink_to(src.resolve())
                return False

            if needs_ccdsum:
                header['CCDSUM'] = (ccdsum, 'Injected by reduce_osiris.py for PypeIt')
                if not str(header.get('BINNING', '')).strip():
                    header['BINNING'] = (ccdsum, 'Injected by reduce_osiris.py for PypeIt')
            if needs_rdnoise:
                header['RDNOISE'] = (rdnoise, 'Injected by reduce_osiris.py for PypeIt')
            hdul.writeto(dst, overwrite=True)
            return True
    except Exception:
        dst.symlink_to(src.resolve())
        return False


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

    n_kept = n_excl = n_patched = 0
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
                staged = raw_dir / f.name
                if not staged.exists():
                    if stage_for_pypeit(f, staged):
                        n_patched += 1
                n_kept += 1

    print(f"  {n_kept} files kept, {n_excl} GRISM=OPEN acquisition images excluded")
    if n_patched:
        print(f"  Patched {n_patched} staged file(s) with inferred metadata for PypeIt")
    return raw_dir


def config_has_science(pypeit_file: Path) -> bool:
    """Return True if this config contains at least one science frame."""
    return bool(re.search(r'\|\s+science\s+\|', pypeit_file.read_text()))


def config_has_trace_frames(pypeit_file: Path) -> bool:
    """Return True if this config contains at least one trace/pixelflat frame."""
    return bool(re.search(r'\|\s+[^|]*trace[^|]*\|', pypeit_file.read_text()))


def config_has_arc_frames(pypeit_file: Path) -> bool:
    """Return True if this config contains at least one arc frame."""
    return bool(re.search(r'\|\s+[^|]*arc[^|]*\|', pypeit_file.read_text()))


# ── Main pipeline ──────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ob_dir',
                    help='OB directory (contains arc/, bias/, flat/, object/, stds/)')
    ap.add_argument('--spectrograph', default='auto',
                    choices=['auto', 'gtc_osiris_plus', 'gtc_osiris'],
                    help='PypeIt spectrograph name (default: auto-detect from raw headers)')
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
    ap.add_argument('--science-spat', type=float, default=None,
                    help='Prefer the extracted science object nearest this spatial column')
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
    ap.add_argument('--merge-center', type=float, default=MERGE_CENTER,
                    help='Wavelength [A] where R1000B/R1000R are stitched (default: 6200)')
    ap.add_argument('--merge-width', type=float, default=MERGE_WIDTH,
                    help='Width [A] of the scaling window around --merge-center (default: 40)')
    return ap.parse_args()


def main():
    args = parse_args()

    ob_dir = Path(args.ob_dir).resolve()
    if not ob_dir.is_dir():
        sys.exit(f"ERROR: {ob_dir} is not a directory")
    for sd in RAW_SUBDIRS:
        if not (ob_dir / sd).is_dir():
            sys.exit(f"ERROR: missing expected subdirectory: {ob_dir / sd}")

    detected_spectrograph = detect_spectrograph(ob_dir)
    if args.spectrograph == 'auto':
        args.spectrograph = detected_spectrograph
    elif args.spectrograph != detected_spectrograph:
        sys.exit(
            f"ERROR: requested spectrograph {args.spectrograph} does not match the raw "
            f"headers in {ob_dir.name} (detected {detected_spectrograph}). "
            "Use --spectrograph auto or the detected value."
        )

    interactive = not args.no_interactive
    det = 2 if args.spectrograph == 'gtc_osiris' else 1

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

    all_config_dirs = [cd for cd in find_config_dirs(ob_dir)
                       if cd.name.startswith(args.spectrograph)]
    if not all_config_dirs:
        sys.exit(
            f"ERROR: no config directories found for {args.spectrograph} "
            f"(expected {args.spectrograph}_A, {args.spectrograph}_B …)"
        )

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
    corrected_groups = defaultdict(list)

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
        n_arc_fixed = enforce_arc_frametypes(pf, ob_dir)
        if n_arc_fixed:
            print(f"    Corrected frametype to arc,tilt for {n_arc_fixed} arc file(s)")
        patch_pypeit_params(pf, args.maxnumber_sci, args.snr_thresh,
                            args.find_fwhm, args.find_min_max, args.spectrograph,
                            args.trace_npoly)
        if args.science_spat is not None:
            seed_science_manual_extractions(
                pf, args.science_spat, spec_px=1024.0, fwhm=4.0, det=det
            )
        print(f"    Grating: {get_dispname(pf)}")

    # ── 3. run_pypeit ────────────────────────────────────────────────────
    if not args.skip_pypeit:
        print(f"\n{'─'*50}\nStep 3 — run_pypeit\n{'─'*50}")
        for config_dir, pf in zip(config_dirs, pypeit_files):
            if not config_has_trace_frames(pf):
                dispname = get_dispname(pf)
                sys.exit(
                    f"ERROR: {pf.name} contains science data for {dispname} but no "
                    "trace/pixelflat frames. This OB is missing the matching spectral "
                    "flat calibration for that configuration, so run_pypeit cannot proceed."
                )
            if not config_has_arc_frames(pf):
                dispname = get_dispname(pf)
                sys.exit(
                    f"ERROR: {pf.name} contains science data for {dispname} but no "
                    "arc frames. PypeIt likely mis-typed the arc lamps; inspect the "
                    ".pypeit file or rerun setup with arc frametype correction enabled."
                )
            cmd = ['run_pypeit', pf.name] + (['-o'] if args.overwrite else [])
            run(cmd, cwd=config_dir)
    else:
        print(f"\nStep 3 — run_pypeit  [skipped]")

    # ── Steps 4–7: per configuration ────────────────────────────────────
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

        supermanual_traces = {}
        localbg_exports = set()

        # ── 3b. Trace validation ─────────────────────────────────────────
        if interactive:
            print(f"\n{'─'*50}\nStep 3b — trace validation\n{'─'*50}")
            while True:
                need_rerun = False

                spec1d_all = sorted(science_dir.glob('spec1d_*.fits'))
                std_files  = [f for f in spec1d_all if 'SPSTD_' in f.name]
                sci_files  = [f for f in spec1d_all if 'SPSTD_' not in f.name]
                std_spec1d = std_files[0] if std_files else std_spec1d

                def _validate_one(spec1d_file, label, preferred_spat=None):
                    """Validate trace for one spec1d; return True if rerun needed."""
                    if spec1d_file.name in supermanual_traces:
                        print(f"  Using stored supermanual trace for {spec1d_file.name}")
                        return False
                    spec1d_path = science_dir / spec1d_file.name
                    spec2d_path = _find_matching_spec2d(spec1d_path)
                    if spec2d_path is None or not spec2d_path.exists():
                        print(f"  WARNING: spec2d not found for {spec1d_file.name} – skipping plot")
                        return False
                    action, payload = validate_trace(
                        spec2d_path, spec1d_path, label, preferred_spat=preferred_spat,
                        allow_supermanual=('SPSTD_' not in spec1d_file.name)
                    )
                    if action in ('rerun_manual', 'rerun_manual_localbg') and payload is not None:
                        spat, spec_px, fwhm = payload
                        set_manual_extraction(pf, spec1d_file.name, spat, spec_px, fwhm, det)
                        supermanual_traces.pop(spec1d_file.name, None)
                        if action == 'rerun_manual_localbg':
                            localbg_exports.add(spec1d_file.name)
                        else:
                            localbg_exports.discard(spec1d_file.name)
                        return True
                    if action == 'supermanual' and payload is not None:
                        localbg_exports.discard(spec1d_file.name)
                        supermanual_traces[spec1d_file.name] = payload
                        print(f"  Stored supermanual trace for {spec1d_file.name}")
                    return False

                print(f"\n  --- Standard star ---")
                if _validate_one(std_spec1d, f'STD {std_spec1d.name}'):
                    need_rerun = True

                print(f"\n  --- Science frames ---")
                for sci_spec1d in sci_files:
                    parsed = parse_spec1d_meta(sci_spec1d.name)
                    label = parsed['target'] if parsed is not None else sci_spec1d.name
                    if _validate_one(sci_spec1d, label, preferred_spat=args.science_spat):
                        need_rerun = True

                if not need_rerun:
                    break

                print("\n  Manual/custom trace selected — re-running run_pypeit -o "
                      "to rebuild the extraction and sky subtraction…")
                print("  After this rerun, the validation plot will open again.")
                print("  If the green trace is now correct, accept it with [g] or Enter.")
                print("  Do not choose [h] again unless you want to apply another shift.")
                run(['run_pypeit', pf.name, '-o'], cwd=config_dir)

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
            info = build_output_info(science_dir / spec1d.name, ob_dir, grating)
            out_txt = ob_dir / f"{info['stem']}.txt"
            spec1d_path = science_dir / spec1d.name
            if spec1d.name in supermanual_traces:
                spec2d_path = _find_matching_spec2d(spec1d_path)
                if spec2d_path is None or not spec2d_path.exists():
                    sys.exit(f"ERROR: spec2d not found for supermanual extraction of {spec1d.name}")
                export_ascii_supermanual(
                    spec1d_path, spec2d_path, out_txt, ob_dir,
                    preferred_spat=args.science_spat, **supermanual_traces[spec1d.name]
                )
            elif spec1d.name in localbg_exports:
                spec2d_path = _find_matching_spec2d(spec1d_path)
                if spec2d_path is None or not spec2d_path.exists():
                    sys.exit(f"ERROR: spec2d not found for local-background extraction of {spec1d.name}")
                export_ascii_localbg(
                    spec1d_path, spec2d_path, out_txt, ob_dir,
                    preferred_spat=args.science_spat, mode_label='HSHIFT_LOCALBG'
                )
            else:
                export_ascii(spec1d_path, out_txt, ob_dir,
                             preferred_spat=args.science_spat)
            exported.append({'txt': out_txt, 'info': info})

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
                grating, [item['txt'] for item in exported])

        corrected = []
        for item in exported:
            out_txt = item['txt']
            info = item['info']
            tc_txt = ob_dir / f"{info['stem']}_tellcorr.txt"
            apply_tell_correction(out_txt, lam_tell, tell_corr, tc_txt)
            corrected.append(tc_txt)
            corrected_groups[info['group_key']].append(tc_txt)

    if not args.skip_telluric:
        for (target, grating), group_files in sorted(corrected_groups.items()):
            if len(group_files) <= 1:
                continue
            avg_token = _average_datetime_token(group_files) or 'UNKNOWN'
            combined = ob_dir / f'{target}_{grating}_{avg_token}_combined_tellcorr.txt'
            try:
                combine_spectra(group_files, combined)
            except ValueError as exc:
                sys.exit(f"ERROR: {exc}")

    # ── Optional B/R stitching ───────────────────────────────────────────
    merged_candidates = {}
    suffix = '_tellcorr.txt' if not args.skip_telluric else '.txt'
    for txt in sorted(ob_dir.glob(f'*{suffix}')):
        stem = txt.name[:-len(suffix)]
        if '_R1000B_' in stem:
            target = stem.split('_R1000B_')[0]
            merged_candidates.setdefault(target, {})['B'] = txt
        elif '_R1000R_' in stem:
            target = stem.split('_R1000R_')[0]
            merged_candidates.setdefault(target, {})['R'] = txt

    for target, pair in sorted(merged_candidates.items()):
        if 'B' not in pair or 'R' not in pair:
            continue
        avg_token = _average_datetime_token([pair['B'], pair['R']]) or 'UNKNOWN'
        out_name = f'{target}_R1000BR_{avg_token}_merged{suffix}'
        out_txt = ob_dir / out_name
        try:
            merge_blue_red_spectra(pair['B'], pair['R'], out_txt,
                                   center=args.merge_center, width=args.merge_width)
        except ValueError as exc:
            print(f"  WARNING: could not stitch R1000B/R1000R for {target} ({exc})")

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
