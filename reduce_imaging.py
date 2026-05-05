#!/usr/bin/env python3
"""
reduce_imaging.py — OSIRIS/GTC broadband imaging reduction pipeline.

Steps
-----
1. Overscan-subtract and trim every raw frame.
2. Build master bias (median combine).
3. Build master flat (bias-subtract → normalise → sigma-clip median combine).
4. Reduce each science frame (bias-subtract → flat-divide).
5. Align all reduced science frames to the first using image-based registration.
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
from photutils.detection import DAOStarFinder


# ── Section parsing ────────────────────────────────────────────────────────────

SHARED_CAL_DIR = Path(__file__).resolve().parent / 'img_cal'

def _parse_fits_section(sec: str):
    """Return (x0, x1, y0, y1) slice indices (0-based, exclusive end) from
    a FITS section string like '[51:4146,1:4112]'."""
    sec = sec.strip().strip('[]')
    xpart, ypart = sec.split(',')
    x0, x1 = (int(v) for v in xpart.split(':'))
    y0, y1 = (int(v) for v in ypart.split(':'))
    # FITS is 1-indexed; convert to numpy 0-indexed
    return x0 - 1, x1, y0 - 1, y1   # data[y0:y1, x0:x1]


def _repair_overscan_rows(overscan: np.ndarray) -> tuple:
    """Repair pathological overscan rows via interpolation.

    Some OSIRIS imaging frames contain short runs of wildly high overscan
    values for a handful of rows. Subtracting those rows verbatim imprints a
    dark horizontal band in the trimmed science image. Detect strong outliers
    in the row-wise overscan vector and replace them by linear interpolation
    between neighboring good rows.
    """
    overscan = np.asarray(overscan, dtype=np.float32)
    finite = np.isfinite(overscan)
    if not np.any(finite):
        return overscan, np.zeros_like(overscan, dtype=bool)

    center = float(np.nanmedian(overscan[finite]))
    mad = float(1.4826 * np.nanmedian(np.abs(overscan[finite] - center)))
    if not np.isfinite(mad) or mad == 0:
        mad = float(np.nanstd(overscan[finite]))
    threshold = max(25.0 * mad, 100.0)
    bad = (~finite) | (np.abs(overscan - center) > threshold)
    if not np.any(bad):
        return overscan, bad

    good_idx = np.flatnonzero(~bad)
    if good_idx.size < 2:
        return overscan, np.zeros_like(overscan, dtype=bool)

    repaired = overscan.copy()
    bad_idx = np.flatnonzero(bad)
    repaired[bad_idx] = np.interp(bad_idx, good_idx, overscan[good_idx]).astype(np.float32)
    return repaired, bad


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
        overscan, bad_rows = _repair_overscan_rows(overscan)
        if np.any(bad_rows):
            print(f"    Repaired {int(np.sum(bad_rows))} overscan row(s)")
        data -= overscan[:, np.newaxis]

    if trimsec:
        tx0, tx1, ty0, ty1 = _parse_fits_section(trimsec)
        data = data[ty0:ty1, tx0:tx1]

    return data


def _collect_fits_files(directory: Path) -> list:
    """Collect FITS/FITS.GZ files once, preferring uncompressed copies."""
    files = {}
    for path in sorted(directory.glob('*.fits')):
        files[path.name] = path
    for path in sorted(directory.glob('*.fits.gz')):
        key = path.name[:-3]
        files.setdefault(key, path)
    return sorted(files.values())


def _collect_raw_fits_files(directory: Path) -> list:
    """Collect raw FITS files, excluding prebuilt master calibrations."""
    return [path for path in _collect_fits_files(directory)
            if not path.name.startswith('master_')]


def _get_ccdsum(header) -> str:
    """Return CCDSUM as a normalized string like '2 2'."""
    return ' '.join(str(header.get('CCDSUM', '')).split()) or 'unknown'


def _get_file_ccdsum(path: Path) -> str:
    return _get_ccdsum(fits.getheader(path))


def _filter_files_by_ccdsum(files: list, ccdsum: str) -> list:
    return [path for path in files if _get_file_ccdsum(path) == ccdsum]


def _ccdsum_token(ccdsum: str) -> str:
    return ccdsum.replace(' ', 'x')


def _shared_master_bias_path(shared_dir: Path, ccdsum: str) -> Path:
    return shared_dir / f'master_bias_{_ccdsum_token(ccdsum)}.fits'


def _shared_master_flat_path(shared_dir: Path, filt_name: str, ccdsum: str) -> Path:
    return shared_dir / f'master_flat_{filt_name}_{_ccdsum_token(ccdsum)}.fits'


def _load_shared_master_bias(shared_dir: Path, ccdsum: str) -> tuple:
    """Return (data, path, n_bias) for a shared master bias, or (None, None, 0)."""
    path = _shared_master_bias_path(shared_dir, ccdsum)
    if not path.exists():
        legacy = shared_dir / 'master_bias.fits'
        path = legacy if legacy.exists() else None
    if path is None or not path.exists():
        return None, None, 0

    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
        n_bias = int(hdul[0].header.get('NBIAS', 1))
    return data, path, n_bias


def _load_shared_master_flats(shared_dir: Path, ccdsum: str) -> dict:
    """Return {filter: (data, path, n_flat)} for shared master flats."""
    token = _ccdsum_token(ccdsum)
    masters = {}
    for path in sorted(shared_dir.glob(f'master_flat_*_{token}.fits')):
        stem = path.stem
        prefix = 'master_flat_'
        suffix = f'_{token}'
        if not stem.startswith(prefix) or not stem.endswith(suffix):
            continue
        filt_name = stem[len(prefix):-len(suffix)]
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
            n_flat = int(hdul[0].header.get('NFLAT', 1))
        masters[filt_name] = (data, path, n_flat)
    return masters


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


def _registration_image(data: np.ndarray) -> np.ndarray:
    """Return a background-subtracted image that emphasises compact sources."""
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    image = np.asarray(data, dtype=np.float32) - np.float32(med)
    floor = 2.0 * float(std)
    image[image < floor] = 0.0
    return image


def _detect_registration_sources(data: np.ndarray, max_sources: int = 40) -> np.ndarray:
    """Detect bright sources for translation-only image registration."""
    _, _, std = sigma_clipped_stats(data, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return np.empty((0, 2), dtype=np.float32)

    finder = DAOStarFinder(fwhm=4.0, threshold=8.0 * std, exclude_border=True)
    table = finder(_registration_image(data))
    if table is None or len(table) == 0:
        return np.empty((0, 2), dtype=np.float32)

    table.sort('flux')
    table = table[::-1][:max_sources]
    return np.column_stack((np.asarray(table['xcentroid'], dtype=np.float32),
                            np.asarray(table['ycentroid'], dtype=np.float32)))


def _source_match_shift(ref_data: np.ndarray, src_data: np.ndarray,
                        tolerance: float = 3.0) -> tuple:
    """Estimate a pure translation from detected source centroids."""
    ref_xy = _detect_registration_sources(ref_data)
    src_xy = _detect_registration_sources(src_data)
    if len(ref_xy) < 3 or len(src_xy) < 3:
        raise ValueError('not enough detected sources')

    shifts = (ref_xy[:, None, :] - src_xy[None, :, :]).reshape(-1, 2)
    rounded = np.rint(shifts).astype(int)
    unique, counts = np.unique(rounded, axis=0, return_counts=True)
    best_dx, best_dy = unique[np.argmax(counts)]
    keep = ((np.abs(shifts[:, 0] - best_dx) <= tolerance) &
            (np.abs(shifts[:, 1] - best_dy) <= tolerance))
    matched = shifts[keep]
    if len(matched) < 3:
        raise ValueError('source matching produced too few consistent pairs')

    dx, dy = np.median(matched, axis=0)
    return float(dy), float(dx), int(len(matched))


def _cross_correlation_shift(ref_data: np.ndarray, src_data: np.ndarray) -> tuple:
    """Fallback translation from phase cross-correlation on source-enhanced images."""
    from skimage.registration import phase_cross_correlation

    ref_image = _registration_image(ref_data)
    src_image = _registration_image(src_data)
    shift_yx, _, _ = phase_cross_correlation(ref_image, src_image,
                                             upsample_factor=20)
    return float(shift_yx[0]), float(shift_yx[1])


def align_to_reference(science_frames: list) -> list:
    """Align all frames to the first using source-based image registration."""
    from scipy.ndimage import shift as ndshift

    ref_data, ref_header = science_frames[0]
    aligned = [science_frames[0]]

    for data, header in science_frames[1:]:
        try:
            dy, dx, nmatches = _source_match_shift(ref_data, data)
            method = f'source-match ({nmatches} pairs)'
        except Exception as e:
            print(f"    Source matching failed ({e}), using cross-correlation")
            dy, dx = _cross_correlation_shift(ref_data, data)
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


def _group_files_by_filter(files: list) -> dict:
    groups = {}
    for path in files:
        with fits.open(path) as hdul:
            filt = _get_filter(hdul[0].header)
        groups.setdefault(filt, []).append(path)
    return groups


def _science_ccdsum(sci_files: list) -> str:
    ccdsums = sorted({_get_file_ccdsum(path) for path in sci_files})
    if len(ccdsums) != 1:
        raise ValueError(f'multiple science CCDSUM values found: {ccdsums}')
    return ccdsums[0]


def build_output_header(ref_header, n_frames: int, n_bias: int,
                        n_flat: int, science_files: list) -> fits.Header:
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
    hdr['NBIAS']    = (n_bias, 'Number of bias frames used')
    hdr['NFLAT']    = (n_flat, 'Number of flat frames used')
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
    shared_bias_dir = SHARED_CAL_DIR / 'bias'
    shared_flat_dir = SHARED_CAL_DIR / 'flats'

    sci_files = _collect_fits_files(ob_dir / 'object')
    if not sci_files:
        sys.exit(f"ERROR: no science frames found in {ob_dir / 'object'}")

    try:
        science_ccdsum = _science_ccdsum(sci_files)
    except ValueError as exc:
        sys.exit(f'ERROR: {exc}')

    local_flat_files = _filter_files_by_ccdsum(_collect_raw_fits_files(ob_dir / 'flat'),
                                               science_ccdsum)
    local_bias_files = _filter_files_by_ccdsum(_collect_raw_fits_files(ob_dir / 'bias'),
                                               science_ccdsum)
    shared_bias_master, shared_bias_path, shared_bias_count = _load_shared_master_bias(
        shared_bias_dir, science_ccdsum
    )
    shared_bias_files = _filter_files_by_ccdsum(_collect_raw_fits_files(shared_bias_dir),
                                                science_ccdsum)
    shared_flat_masters = _load_shared_master_flats(shared_flat_dir, science_ccdsum)
    shared_flat_files = _filter_files_by_ccdsum(_collect_raw_fits_files(shared_flat_dir),
                                                science_ccdsum)

    sci_groups = _group_files_by_filter(sci_files)
    local_flat_groups = _group_files_by_filter(local_flat_files) if local_flat_files else {}
    shared_flat_groups = _group_files_by_filter(shared_flat_files) if shared_flat_files else {}

    if local_bias_files:
        bias_source = f'local bias frames ({len(local_bias_files)})'
        bias_count = len(local_bias_files)
        bias_files = local_bias_files
        master_bias = None
    elif shared_bias_master is not None:
        print(f"  Using shared master bias {shared_bias_path.name} for CCDSUM={science_ccdsum}")
        bias_source = f'shared master bias ({shared_bias_count})'
        bias_count = shared_bias_count
        bias_files = [shared_bias_path]
        master_bias = shared_bias_master
    elif shared_bias_files:
        print(f"  Using shared bias frames from {shared_bias_dir} for CCDSUM={science_ccdsum}")
        bias_source = f'shared bias frames ({len(shared_bias_files)})'
        bias_count = len(shared_bias_files)
        bias_files = shared_bias_files
        master_bias = None
    else:
        sys.exit(
            f"ERROR: no bias calibration with CCDSUM={science_ccdsum} found in "
            f"{ob_dir / 'bias'} or {shared_bias_dir}"
        )

    flat_available = set(local_flat_groups) | set(shared_flat_masters) | set(shared_flat_groups)
    if not flat_available:
        sys.exit(
            f"ERROR: no flat calibration with CCDSUM={science_ccdsum} found in "
            f"{ob_dir / 'flat'} or {shared_flat_dir}"
        )

    print(f"\n{'='*60}")
    print(f"  OB directory : {ob_dir}")
    print(f"  Science CCDSUM: {science_ccdsum}")
    print(f"  Bias source  : {bias_source}")
    print(f"  Flat sources : local={ {k: len(v) for k, v in local_flat_groups.items()} }")
    if shared_flat_masters:
        print(f"  Shared flat masters: { {k: v[2] for k, v in shared_flat_masters.items()} }")
    if shared_flat_groups:
        print(f"  Shared flat frames : { {k: len(v) for k, v in shared_flat_groups.items()} }")
    print(f"  Science frames: {len(sci_files)}")
    print(f"  Sci filters  : { {k: len(v) for k, v in sci_groups.items()} }")
    print(f"{'='*60}\n")

    # ── Calibrations
    if master_bias is None:
        master_bias = build_master_bias(bias_files)

    # Save calibrations
    fits.writeto(str(output_dir / 'master_bias.fits'), master_bias,
                 overwrite=True)
    master_flats = {}
    flat_counts = {}
    for filt_name in sorted(flat_available):
        if filt_name in local_flat_groups:
            filt_flat_files = local_flat_groups[filt_name]
            print(f"  Filter {filt_name} (local flats):")
            master_flat = build_master_flat(filt_flat_files, master_bias)
            flat_counts[filt_name] = len(filt_flat_files)
        elif filt_name in shared_flat_masters:
            master_flat, master_path, n_flat = shared_flat_masters[filt_name]
            print(f"  Filter {filt_name} (shared master flat: {master_path.name}, NFLAT={n_flat})")
            flat_counts[filt_name] = n_flat
        else:
            filt_flat_files = shared_flat_groups[filt_name]
            print(f"  Filter {filt_name} (shared flats):")
            master_flat = build_master_flat(filt_flat_files, master_bias)
            flat_counts[filt_name] = len(filt_flat_files)
        master_flats[filt_name] = master_flat
        fits.writeto(str(output_dir / f'master_flat_{filt_name}.fits'),
                     master_flat, overwrite=True)
    if len(master_flats) == 1:
        only_flat = next(iter(master_flats.values()))
        fits.writeto(str(output_dir / 'master_flat.fits'), only_flat,
                     overwrite=True)
    print()

    # ── Reduce science frames per filter
    for filt_name, filt_sci_files in sorted(sci_groups.items()):
        if filt_name not in master_flats:
            print(f"  WARNING: no flats found for filter {filt_name}; skipping science combination")
            continue

        print(f"  Reducing {len(filt_sci_files)} science frame(s) in filter {filt_name} …")
        science_frames = []
        for f in filt_sci_files:
            data, hdr = reduce_frame(f, master_bias, master_flats[filt_name])
            science_frames.append((data, hdr))
            print(f"    {f.name}  median={np.nanmedian(data):.1f}")

        # Grab metadata from first science frame for output naming
        ref_header = science_frames[0][1]
        obj_name  = _get_object(ref_header)
        date_str  = _get_date(ref_header)
        print()

        # ── Align (optional)
        if not no_align and len(science_frames) > 1:
            print("  Aligning frames to reference image …")
            science_frames = align_to_reference(science_frames)
            print()
        elif no_align and len(science_frames) > 1:
            print("  Skipping image registration (--no-align)")
            print()

        # ── Combine
        combined = combine_frames(science_frames)

        # ── Write output
        out_hdr = build_output_header(ref_header, len(science_frames),
                                      bias_count, flat_counts[filt_name], filt_sci_files)
        out_name = f"{obj_name}_{filt_name}_{date_str}.fits"
        out_path = output_dir / out_name
        fits.writeto(str(out_path), combined, header=out_hdr, overwrite=True)

        print(f"\n  Combined image saved → {out_path}")
        _, med, std = sigma_clipped_stats(combined[np.isfinite(combined)])
        print(f"  Stats: median={med:.2f}  std={std:.2f}  shape={combined.shape}")
        print()


def parse_args():
    p = argparse.ArgumentParser(
        description='Reduce OSIRIS/GTC broadband imaging for one OB directory.')
    p.add_argument('ob_dir', help='OB directory (contains bias/, flat/, object/)')
    p.add_argument('--output-dir', default=None,
                   help='Directory for output files (default: OB_DIR/Reduced/)')
    p.add_argument('--no-align', action='store_true',
                   help='Skip image registration before combining')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ob_path = Path(args.ob_dir)
    out_path = Path(args.output_dir) if args.output_dir else ob_path / 'Reduced'
    reduce_ob(ob_path, out_path, no_align=args.no_align)
