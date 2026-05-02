#!/usr/bin/env python3
"""
reduce_imaging.py — OSIRIS/GTC broadband imaging reduction pipeline.

Steps
-----
1. Overscan-subtract and trim every raw frame.
2. Build master bias (median combine).
3. Build master flat (bias-subtract → normalise → sigma-clip median combine).
4. Reduce each science frame (bias-subtract → flat-divide).
5. Align all reduced science frames to the first using WCS (reproject).
6. Combine with sigma-clipped mean → write final FITS.

Usage
-----
    python reduce_imaging.py OB0010 [--output-dir PATH] [--no-align]

The OB directory must contain the sub-directories:
    bias/   flat/   object/   (stds/ is ignored for imaging)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.wcs import WCS


# ── Section parsing ────────────────────────────────────────────────────────────

def _parse_fits_section(sec: str):
    """Return (x0, x1, y0, y1) slice indices (0-based, exclusive end) from
    a FITS section string like '[51:4146,1:4112]'."""
    sec = sec.strip().strip('[]')
    xpart, ypart = sec.split(',')
    x0, x1 = (int(v) for v in xpart.split(':'))
    y0, y1 = (int(v) for v in ypart.split(':'))
    # FITS is 1-indexed; convert to numpy 0-indexed
    return x0 - 1, x1, y0 - 1, y1   # data[y0:y1, x0:x1]


def overscan_subtract_and_trim(data: np.ndarray, header) -> np.ndarray:
    """Subtract overscan level and trim to TRIMSEC.

    Uses BIASSEC to measure the overscan level (median per row), then
    trims to TRIMSEC.  Returns float32 array.
    """
    data = data.astype(np.float32)

    biassec = header.get('BIASSEC')
    trimsec = header.get('TRIMSEC')

    if biassec:
        bx0, bx1, by0, by1 = _parse_fits_section(biassec)
        overscan = np.median(data[by0:by1, bx0:bx1], axis=1)  # (ny,)
        data -= overscan[:, np.newaxis]

    if trimsec:
        tx0, tx1, ty0, ty1 = _parse_fits_section(trimsec)
        data = data[ty0:ty1, tx0:tx1]

    return data


# ── Master calibrations ────────────────────────────────────────────────────────

def build_master_bias(bias_files: list) -> np.ndarray:
    """Median-combine overscan-corrected bias frames."""
    print(f"  Building master bias from {len(bias_files)} frames …")
    stack = []
    for f in bias_files:
        with fits.open(f) as hdul:
            arr = overscan_subtract_and_trim(hdul[0].data, hdul[0].header)
        stack.append(arr)
    master = np.median(np.stack(stack, axis=0), axis=0)
    print(f"    Master bias shape: {master.shape}  "
          f"median={np.median(master):.2f}  std={np.std(master):.2f}")
    return master.astype(np.float32)


def build_master_flat(flat_files: list, master_bias: np.ndarray) -> np.ndarray:
    """Build a normalised master flat.

    Each flat is overscan-corrected, bias-subtracted, then normalised by
    its own median before sigma-clip stacking.
    """
    print(f"  Building master flat from {len(flat_files)} frames …")
    stack = []
    for f in flat_files:
        with fits.open(f) as hdul:
            arr = overscan_subtract_and_trim(hdul[0].data, hdul[0].header)
        arr -= master_bias
        med = np.median(arr)
        if med <= 0:
            print(f"    WARNING: {Path(f).name} has non-positive median after bias sub, skipping")
            continue
        stack.append(arr / med)

    cube = np.stack(stack, axis=0)
    # Sigma-clip along the frame axis
    clipped = sigma_clip(cube, sigma=3.0, axis=0, masked=True)
    master = np.ma.mean(clipped, axis=0).filled(np.nan)
    # Renormalise to 1
    norm = np.nanmedian(master)
    master /= norm
    # Replace any bad pixels (NaN, ≤0) with 1 so division is safe
    bad = ~np.isfinite(master) | (master <= 0)
    if bad.sum():
        print(f"    Replacing {bad.sum()} bad flat pixels with 1.0")
        master[bad] = 1.0
    print(f"    Master flat shape: {master.shape}  "
          f"median={np.nanmedian(master):.4f}  std={np.nanstd(master):.4f}")
    return master.astype(np.float32)


# ── Science reduction ──────────────────────────────────────────────────────────

def reduce_frame(fits_path: Path, master_bias: np.ndarray,
                 master_flat: np.ndarray) -> tuple:
    """Return (reduced_data, header) for a single science frame."""
    with fits.open(fits_path) as hdul:
        header = hdul[0].header.copy()
        data = overscan_subtract_and_trim(hdul[0].data, hdul[0].header)
    data -= master_bias
    data /= master_flat
    return data.astype(np.float32), header


# ── Alignment ─────────────────────────────────────────────────────────────────

def _trim_wcs_header(header, ref_shape):
    """Strip TRIMSEC and update CRPIX to the trimmed coordinate system."""
    trimsec = header.get('TRIMSEC')
    if trimsec is None:
        return header
    tx0, tx1, ty0, ty1 = _parse_fits_section(trimsec)
    header = header.copy()
    if 'CRPIX1' in header:
        header['CRPIX1'] = float(header['CRPIX1']) - tx0
    if 'CRPIX2' in header:
        header['CRPIX2'] = float(header['CRPIX2']) - ty0
    return header


def _wcs_shift(ref_header, src_header) -> tuple:
    """Compute (dy, dx) pixel shift from src to ref using WCS.

    Projects the RA/Dec of the src frame centre into the ref WCS.
    Returns (dy, dx) in pixels (ref_pixel - src_pixel convention for
    scipy.ndimage.shift).
    """
    ref_hdr = _trim_wcs_header(ref_header, None)
    src_hdr = _trim_wcs_header(src_header, None)

    ref_wcs = WCS(ref_hdr)
    src_wcs = WCS(src_hdr)

    # Sky position of the source frame centre
    ny, nx = src_hdr['NAXIS2'], src_hdr['NAXIS1']
    cx, cy = nx / 2.0, ny / 2.0
    sky = src_wcs.pixel_to_world(cx, cy)

    # Where does that sky position land in the reference frame?
    ref_x, ref_y = ref_wcs.world_to_pixel(sky)

    dy = float(ref_y - cy)
    dx = float(ref_x - cx)
    return dy, dx


def align_to_reference(science_frames: list) -> list:
    """Align all frames to the first using WCS-derived pixel shifts.

    Uses each frame's WCS to compute the integer+sub-pixel offset relative
    to the reference frame, then applies it with scipy.ndimage.shift
    (cubic spline interpolation, NaN-filled borders).

    Falls back to phase cross-correlation if WCS is absent or degenerate.
    """
    from scipy.ndimage import shift as ndshift
    from skimage.registration import phase_cross_correlation

    ref_data, ref_header = science_frames[0]
    aligned = [science_frames[0]]

    for data, header in science_frames[1:]:
        # Try WCS-based shift first
        try:
            dy, dx = _wcs_shift(ref_header, header)
            # Sanity check: if shift is larger than half the image something is wrong
            if abs(dy) > ref_data.shape[0] / 2 or abs(dx) > ref_data.shape[1] / 2:
                raise ValueError(f"WCS shift ({dy:.1f}, {dx:.1f}) implausibly large")
            method = 'WCS'
        except Exception as e:
            print(f"    WCS alignment failed ({e}), using cross-correlation")
            # Use central 512×512 patch to speed up cross-correlation
            cy, cx = ref_data.shape[0] // 2, ref_data.shape[1] // 2
            half = 256
            ref_patch = ref_data[cy - half:cy + half, cx - half:cx + half]
            src_patch = data[cy - half:cy + half, cx - half:cx + half]
            shift_yx, _, _ = phase_cross_correlation(ref_patch, src_patch,
                                                     upsample_factor=10)
            dy, dx = float(shift_yx[0]), float(shift_yx[1])
            method = 'cross-corr'

        shifted = ndshift(data, shift=(dy, dx), order=3, mode='constant',
                          cval=np.nan)
        aligned.append((shifted.astype(np.float32), ref_header))
        print(f"    Aligned via {method}: dy={dy:+.2f} dx={dx:+.2f} px")

    return aligned


# ── Combination ───────────────────────────────────────────────────────────────

def combine_frames(frames: list) -> np.ndarray:
    """Sigma-clipped mean combination."""
    print(f"  Combining {len(frames)} frames …")
    cube = np.stack([d for d, _ in frames], axis=0)
    if len(frames) == 1:
        return cube[0]
    clipped = sigma_clip(cube, sigma=3.0, axis=0, masked=True)
    combined = np.ma.mean(clipped, axis=0).filled(np.nan)
    return combined.astype(np.float32)


# ── Header helpers ─────────────────────────────────────────────────────────────

def _get_filter(header) -> str:
    """Return a clean filter name from FILTER2 or fallback keywords."""
    for key in ('FILTER2', 'FILTER1', 'FILTER'):
        val = header.get(key, '')
        if val and val.upper() not in ('', 'OPEN', '--'):
            return val.replace(' ', '_')
    return 'unknown'


def _get_object(header) -> str:
    return str(header.get('OBJECT', 'OBJECT')).replace(' ', '_')


def _get_date(header) -> str:
    d = str(header.get('DATE-OBS', ''))
    return d[:10].replace('-', '') if d else 'unknown'


def build_output_header(ref_header, n_frames: int, bias_files: list,
                        flat_files: list, science_files: list) -> fits.Header:
    """Build a clean output header with WCS and provenance keywords."""
    header = _trim_wcs_header(ref_header, None)
    hdr = fits.Header()

    # Provenance
    hdr['OBJECT']   = header.get('OBJECT', 'OBJECT')
    hdr['INSTRUME'] = header.get('INSTRUME', 'OSIRIS')
    hdr['FILTER']   = _get_filter(header)
    hdr['DATE-OBS'] = header.get('DATE-OBS', '')
    hdr['EXPTIME']  = header.get('EXPTIME', 0.0)
    hdr['AIRMASS']  = header.get('AIRMASS', 0.0)
    hdr['GAIN']     = header.get('GAIN', 0.0)
    hdr['RDNOISE']  = header.get('RDNOISE', 0.0)
    hdr['NFRAMES']  = (n_frames, 'Number of science frames combined')
    hdr['NBIAS']    = (len(bias_files), 'Number of bias frames used')
    hdr['NFLAT']    = (len(flat_files), 'Number of flat frames used')
    hdr.add_comment('Reduced by reduce_imaging.py')

    # WCS
    for key in ('CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2',
                'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                'RADESYS', 'EQUINOX'):
        if key in header:
            hdr[key] = header[key]

    return hdr


# ── Main ──────────────────────────────────────────────────────────────────────

def reduce_ob(ob_dir: Path, output_dir: Path, no_align: bool = False) -> None:
    ob_dir = ob_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    bias_files = sorted((ob_dir / 'bias').glob('*.fits'))
    flat_files  = sorted((ob_dir / 'flat').glob('*.fits'))
    sci_files   = sorted((ob_dir / 'object').glob('*.fits'))

    if not bias_files:
        sys.exit(f"ERROR: no bias frames found in {ob_dir / 'bias'}")
    if not flat_files:
        sys.exit(f"ERROR: no flat frames found in {ob_dir / 'flat'}")
    if not sci_files:
        sys.exit(f"ERROR: no science frames found in {ob_dir / 'object'}")

    print(f"\n{'='*60}")
    print(f"  OB directory : {ob_dir}")
    print(f"  Bias frames  : {len(bias_files)}")
    print(f"  Flat frames  : {len(flat_files)}")
    print(f"  Science frames: {len(sci_files)}")
    print(f"{'='*60}\n")

    # ── Calibrations
    master_bias = build_master_bias(bias_files)
    master_flat = build_master_flat(flat_files, master_bias)

    # Save calibrations
    fits.writeto(str(output_dir / 'master_bias.fits'), master_bias,
                 overwrite=True)
    fits.writeto(str(output_dir / 'master_flat.fits'), master_flat,
                 overwrite=True)
    print()

    # ── Reduce science frames
    print(f"  Reducing {len(sci_files)} science frame(s) …")
    science_frames = []
    for f in sci_files:
        data, hdr = reduce_frame(f, master_bias, master_flat)
        science_frames.append((data, hdr))
        print(f"    {f.name}  median={np.nanmedian(data):.1f}")

    # Grab metadata from first science frame for output naming
    ref_header = science_frames[0][1]
    obj_name  = _get_object(ref_header)
    filt_name = _get_filter(ref_header)
    date_str  = _get_date(ref_header)
    print()

    # ── Align (optional)
    if not no_align and len(science_frames) > 1:
        print("  Aligning frames to reference WCS …")
        science_frames = align_to_reference(science_frames)
        print()
    elif no_align and len(science_frames) > 1:
        print("  Skipping WCS alignment (--no-align)")
        print()

    # ── Combine
    combined = combine_frames(science_frames)

    # ── Write output
    out_hdr = build_output_header(ref_header, len(science_frames),
                                  bias_files, flat_files, sci_files)
    out_name = f"{obj_name}_{filt_name}_{date_str}.fits"
    out_path = output_dir / out_name
    fits.writeto(str(out_path), combined, header=out_hdr, overwrite=True)

    print(f"\n  Combined image saved → {out_path}")
    _, med, std = sigma_clipped_stats(combined[np.isfinite(combined)])
    print(f"  Stats: median={med:.2f}  std={std:.2f}  shape={combined.shape}")


def parse_args():
    p = argparse.ArgumentParser(
        description='Reduce OSIRIS/GTC broadband imaging for one OB directory.')
    p.add_argument('ob_dir', help='OB directory (contains bias/, flat/, object/)')
    p.add_argument('--output-dir', default=None,
                   help='Directory for output files (default: OB_DIR/Reduced/)')
    p.add_argument('--no-align', action='store_true',
                   help='Skip WCS alignment before combining')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ob_path = Path(args.ob_dir)
    out_path = Path(args.output_dir) if args.output_dir else ob_path / 'Reduced'
    reduce_ob(ob_path, out_path, no_align=args.no_align)
