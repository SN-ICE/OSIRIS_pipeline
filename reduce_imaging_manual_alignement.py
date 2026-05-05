#!/usr/bin/env python3
"""
Manual alignment and combination for imaging OBs already calibrated by
reduce_imaging.py.

This script reuses the existing master bias and master flat products in
OB_DIR/Reduced/, applies them to the raw science frames in OB_DIR/object/,
and then performs only:

1. Manual source selection in each science frame
2. Local centroid refinement around each click
3. Rotation+translation alignment from 3 matched sources
4. Final sigma-clipped combination

Typical usage:
    python reduce_imaging_manual_alignement.py OB0008a --overwrite

If an OB contains multiple filters, pass one explicitly:
    python reduce_imaging_manual_alignement.py OB0034 --filter Sloan_u --overwrite
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.visualization import AsinhStretch, ImageNormalize
from photutils.centroids import centroid_2dg, centroid_com
from skimage.transform import EuclideanTransform, warp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Manually align and combine imaging frames for one OB/filter.'
    )
    parser.add_argument('ob_dir',
                        help='OB directory containing object/ and Reduced/')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for aligned products (default: OB_DIR/ManualAlign/)')
    parser.add_argument('--filter', default=None,
                        help='Filter to align (required if multiple science filters are present)')
    parser.add_argument('--crop-size', type=int, default=300,
                        help='Half-size of the zoom cutout shown after the first click')
    parser.add_argument('--centroid-box', type=int, default=15,
                        help='Half-size of the local box used to centroid around each click')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing outputs')
    return parser.parse_args()


def _collect_fits_files(directory: Path) -> list[Path]:
    files = {}
    for path in sorted(directory.glob('*.fits')):
        files[path.name] = path
    for path in sorted(directory.glob('*.fits.gz')):
        files.setdefault(path.name[:-3], path)
    return sorted(files.values())


def _parse_fits_section(sec: str) -> tuple[int, int, int, int]:
    sec = sec.strip().strip('[]')
    xpart, ypart = sec.split(',')
    x0, x1 = (int(v) for v in xpart.split(':'))
    y0, y1 = (int(v) for v in ypart.split(':'))
    return x0 - 1, x1, y0 - 1, y1


def overscan_subtract_and_trim(data: np.ndarray, header: fits.Header) -> np.ndarray:
    data = data.astype(np.float32)

    biassec = header.get('BIASSEC')
    trimsec = header.get('TRIMSEC')

    if biassec:
        bx0, bx1, by0, by1 = _parse_fits_section(biassec)
        overscan = np.median(data[by0:by1, bx0:bx1], axis=1)
        data -= overscan[:, np.newaxis]

    if trimsec:
        tx0, tx1, ty0, ty1 = _parse_fits_section(trimsec)
        data = data[ty0:ty1, tx0:tx1]

    return data


def _trim_wcs_header(header: fits.Header) -> fits.Header:
    trimsec = header.get('TRIMSEC')
    if trimsec is None:
        return header.copy()

    tx0, _, ty0, _ = _parse_fits_section(trimsec)
    new_header = header.copy()
    if 'CRPIX1' in new_header:
        new_header['CRPIX1'] = float(new_header['CRPIX1']) - tx0
    if 'CRPIX2' in new_header:
        new_header['CRPIX2'] = float(new_header['CRPIX2']) - ty0
    return new_header


def _get_filter(header: fits.Header) -> str:
    for key in ('FILTER2', 'FILTER1', 'FILTER'):
        value = str(header.get(key, '')).strip()
        if value and value.upper() not in {'OPEN', '--'}:
            return value.replace(' ', '_')
    return 'unknown'


def _get_date(header: fits.Header) -> str:
    date_obs = str(header.get('DATE-OBS', ''))
    return date_obs[:10].replace('-', '') if date_obs else 'unknown'


def _get_object(header: fits.Header) -> str:
    return str(header.get('OBJECT', 'OBJECT')).replace(' ', '_')


def _group_files_by_filter(files: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in files:
        filt = _get_filter(fits.getheader(path))
        groups.setdefault(filt, []).append(path)
    return groups


def _resolve_filter(object_files: list[Path], requested_filter: str | None) -> tuple[str, list[Path]]:
    groups = _group_files_by_filter(object_files)
    if requested_filter is not None:
        if requested_filter not in groups:
            raise ValueError(
                f'filter {requested_filter!r} not found in object/ '
                f'(available: {sorted(groups)})'
            )
        return requested_filter, groups[requested_filter]

    if len(groups) != 1:
        raise ValueError(
            f'multiple science filters found {sorted(groups)}; use --filter to choose one'
        )

    only_filter = next(iter(groups))
    return only_filter, groups[only_filter]


def load_reduced_frames(ob_dir: Path, filter_name: str | None) -> tuple[list[tuple[np.ndarray, fits.Header, Path]], Path, Path, str]:
    object_files = _collect_fits_files(ob_dir / 'object')
    if not object_files:
        sys.exit(f'ERROR: no FITS files found in {ob_dir / "object"}')

    chosen_filter, obj_files = _resolve_filter(object_files, filter_name)

    reduced_dir = ob_dir / 'Reduced'
    bias_path = reduced_dir / 'master_bias.fits'
    if not bias_path.exists():
        sys.exit(f'ERROR: missing master bias: {bias_path}')

    flat_path = reduced_dir / f'master_flat_{chosen_filter}.fits'
    if not flat_path.exists():
        fallback_flat = reduced_dir / 'master_flat.fits'
        if fallback_flat.exists():
            flat_path = fallback_flat
        else:
            sys.exit(f'ERROR: missing master flat: {flat_path}')

    master_bias = fits.getdata(bias_path).astype(np.float32)
    master_flat = fits.getdata(flat_path).astype(np.float32)

    frames = []
    for path in obj_files:
        with fits.open(path) as hdul:
            header = hdul[0].header.copy()
            data = overscan_subtract_and_trim(hdul[0].data, hdul[0].header)
        reduced = (data - master_bias) / master_flat
        frames.append((reduced.astype(np.float32), header, path))

    return frames, bias_path, flat_path, chosen_filter


def _display_limits(data: np.ndarray) -> tuple[float, float]:
    _, med, std = sigma_clipped_stats(data, sigma=3.0)
    vmin = med - 2.0 * std
    vmax = med + 8.0 * std
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = np.nanpercentile(data, [5, 99.5])
    return float(vmin), float(vmax)


def _refine_click_to_centroid(data: np.ndarray, x_click: float, y_click: float,
                              half_size: int) -> tuple[float, float]:
    xi = int(round(x_click))
    yi = int(round(y_click))
    x0 = max(0, xi - half_size)
    x1 = min(data.shape[1], xi + half_size + 1)
    y0 = max(0, yi - half_size)
    y1 = min(data.shape[0], yi + half_size + 1)

    cutout = np.array(data[y0:y1, x0:x1], dtype=float, copy=True)
    if cutout.size == 0:
        return x_click, y_click

    finite = np.isfinite(cutout)
    if not np.any(finite):
        return x_click, y_click

    local_bg = np.nanmedian(cutout[finite])
    cutout = cutout - local_bg
    cutout[~np.isfinite(cutout)] = 0.0
    cutout[cutout < 0] = 0.0
    if np.all(cutout == 0):
        return x_click, y_click

    try:
        xcen, ycen = centroid_2dg(cutout)
    except Exception:
        xcen, ycen = np.nan, np.nan

    if not np.isfinite(xcen) or not np.isfinite(ycen):
        try:
            xcen, ycen = centroid_com(cutout)
        except Exception:
            xcen, ycen = np.nan, np.nan

    if not np.isfinite(xcen) or not np.isfinite(ycen):
        ypeak, xpeak = np.unravel_index(np.nanargmax(cutout), cutout.shape)
        xcen = float(xpeak)
        ycen = float(ypeak)

    return x0 + float(xcen), y0 + float(ycen)


def click_sources(data: np.ndarray, title: str,
                  ref_points: list[tuple[float, float]] | None,
                  crop_size: int, centroid_box: int,
                  n_points: int = 3) -> list[tuple[float, float]]:
    vmin, vmax = _display_limits(data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    fig.suptitle(title)

    axes[0].imshow(data, origin='lower', cmap='gray', norm=norm)
    axes[0].set_title('Full frame')
    axes[0].set_xlabel('X [pix]')
    axes[0].set_ylabel('Y [pix]')

    if ref_points is None:
        cx = data.shape[1] / 2.0
        cy = data.shape[0] / 2.0
    else:
        ref_arr = np.asarray(ref_points, dtype=float)
        cx = float(np.median(ref_arr[:, 0]))
        cy = float(np.median(ref_arr[:, 1]))
        for idx, (rx, ry) in enumerate(ref_points, start=1):
            axes[0].plot(rx, ry, marker='+', color='tab:red', ms=14, mew=2)
            axes[0].text(rx + 10, ry + 10, str(idx), color='tab:red', fontsize=11)

    x0 = max(0, int(round(cx - crop_size)))
    x1 = min(data.shape[1], int(round(cx + crop_size)))
    y0 = max(0, int(round(cy - crop_size)))
    y1 = min(data.shape[0], int(round(cy + crop_size)))

    axes[1].imshow(data, origin='lower', cmap='gray', norm=norm)
    axes[1].set_xlim(x0, x1)
    axes[1].set_ylim(y0, y1)
    axes[1].set_title('Zoom view')
    axes[1].set_xlabel('X [pix]')
    axes[1].set_ylabel('Y [pix]')
    if ref_points is not None:
        for idx, (rx, ry) in enumerate(ref_points, start=1):
            axes[1].plot(rx, ry, marker='+', color='tab:red', ms=14, mew=2)
            axes[1].text(rx + 10, ry + 10, str(idx), color='tab:red', fontsize=11)

    click_state: dict[str, list[tuple[float, float]]] = {'xy': []}

    def onclick(event):
        if event.inaxes not in axes or event.xdata is None or event.ydata is None:
            return
        click_number = len(click_state['xy']) + 1
        x_refined, y_refined = _refine_click_to_centroid(
            data, float(event.xdata), float(event.ydata), centroid_box
        )
        click_state['xy'].append((x_refined, y_refined))
        for ax in axes:
            ax.plot(event.xdata, event.ydata, marker='o', color='tab:cyan',
                    ms=8, mew=1.2, fillstyle='none', alpha=0.7)
            ax.plot(x_refined, y_refined, marker='x', color='tab:cyan', ms=12, mew=2)
            ax.text(x_refined + 10, y_refined + 10, str(click_number),
                    color='tab:cyan', fontsize=11)
        fig.canvas.draw_idle()
        if len(click_state['xy']) >= n_points:
            plt.close(fig)

    def onkey(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    print(
        f'  Click near the same {n_points} sources in order; '
        f'the script will centroid each one locally. Press q to abort this frame.'
    )
    plt.show()

    if len(click_state['xy']) != n_points:
        raise RuntimeError('Not enough clicks recorded')
    return click_state['xy']


def combine_frames(frames: list[np.ndarray]) -> np.ndarray:
    cube = np.stack(frames, axis=0)
    if cube.shape[0] == 1:
        return cube[0].astype(np.float32)
    clipped = sigma_clip(cube, sigma=3.0, axis=0, masked=True)
    return np.ma.mean(clipped, axis=0).filled(np.nan).astype(np.float32)


def transform_frame(data: np.ndarray, transform: EuclideanTransform,
                    output_shape: tuple[int, int]) -> np.ndarray:
    warped = warp(
        data,
        inverse_map=transform.inverse,
        order=3,
        mode='constant',
        cval=np.nan,
        preserve_range=True,
        output_shape=output_shape,
    )
    return warped.astype(np.float32)


def build_output_header(ref_header: fits.Header, n_frames: int,
                        bias_path: Path, flat_path: Path) -> fits.Header:
    header = _trim_wcs_header(ref_header)
    out = fits.Header()
    for key in ('OBJECT', 'INSTRUME', 'FILTER1', 'FILTER2', 'DATE-OBS',
                'EXPTIME', 'AIRMASS', 'GAIN', 'RDNOISE',
                'CTYPE1', 'CTYPE2', 'CRPIX1', 'CRPIX2',
                'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                'RADESYS', 'EQUINOX'):
        if key in header:
            out[key] = header[key]
    out['FILTER'] = _get_filter(header)
    out['NFRAMES'] = (n_frames, 'Number of frames combined')
    out['MBIAS'] = (bias_path.name, 'Master bias used')
    out['MFLAT'] = (flat_path.name, 'Master flat used')
    out.add_comment('Combined with reduce_imaging_manual_alignement.py')
    out.add_comment('Frame transforms derived from manual 3-point clicks')
    return out


def write_offsets_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                'file',
                'p1_x', 'p1_y',
                'p2_x', 'p2_y',
                'p3_x', 'p3_y',
                'shift_dx', 'shift_dy', 'rotation_deg'
            ]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ob_dir = Path(args.ob_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ob_dir / 'ManualAlign'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        frames, bias_path, flat_path, chosen_filter = load_reduced_frames(ob_dir, args.filter)
    except ValueError as exc:
        sys.exit(f'ERROR: {exc}')
    ref_data, ref_header, _ = frames[0]

    print(f'Loaded {len(frames)} frame(s) from {ob_dir / "object"}')
    print(f'Filter: {chosen_filter}')
    print(f'Using bias: {bias_path}')
    print(f'Using flat: {flat_path}')
    print()

    clicks: list[list[tuple[float, float]]] = []
    aligned_frames: list[np.ndarray] = []
    offsets_rows: list[dict[str, object]] = []

    for idx, (data, header, path) in enumerate(frames, start=1):
        title = f'Frame {idx}/{len(frames)}: {path.name}'
        ref_points = clicks[0] if clicks else None
        try:
            points = click_sources(data, title, ref_points=ref_points,
                                   crop_size=args.crop_size,
                                   centroid_box=args.centroid_box,
                                   n_points=3)
        except RuntimeError:
            sys.exit(f'Aborted on {path.name}')

        clicks.append(points)
        if idx == 1:
            dx = 0.0
            dy = 0.0
            rotation_deg = 0.0
            shifted = data
        else:
            src = np.asarray(points, dtype=float)
            dst = np.asarray(clicks[0], dtype=float)
            transform = EuclideanTransform()
            ok = transform.estimate(src, dst)
            if not ok:
                sys.exit(f'Could not solve transform for {path.name}')
            dx = float(transform.translation[0])
            dy = float(transform.translation[1])
            rotation_deg = float(np.degrees(transform.rotation))
            shifted = transform_frame(data, transform, ref_data.shape)

        aligned_frames.append(shifted.astype(np.float32))
        flat_points = [coord for point in points for coord in point]
        offsets_rows.append({
            'file': path.name,
            'p1_x': f'{flat_points[0]:.3f}',
            'p1_y': f'{flat_points[1]:.3f}',
            'p2_x': f'{flat_points[2]:.3f}',
            'p2_y': f'{flat_points[3]:.3f}',
            'p3_x': f'{flat_points[4]:.3f}',
            'p3_y': f'{flat_points[5]:.3f}',
            'shift_dx': f'{dx:.3f}',
            'shift_dy': f'{dy:.3f}',
            'rotation_deg': f'{rotation_deg:.4f}',
        })
        print(f'  {path.name}: dx={dx:+.2f} dy={dy:+.2f} rot={rotation_deg:+.3f} deg')

    combined = combine_frames(aligned_frames)

    obj_name = _get_object(ref_header)
    date_str = _get_date(ref_header)
    out_base = f'{obj_name}_{chosen_filter}_{date_str}_manual_align'
    fits_path = output_dir / f'{out_base}.fits'
    csv_path = output_dir / f'{out_base}_offsets.csv'
    png_path = output_dir / f'{out_base}.png'

    if not args.overwrite:
        for out_path in (fits_path, csv_path, png_path):
            if out_path.exists():
                sys.exit(f'ERROR: output exists: {out_path} (use --overwrite)')

    out_header = build_output_header(ref_header, len(aligned_frames), bias_path, flat_path)
    fits.writeto(fits_path, combined, header=out_header, overwrite=args.overwrite)
    write_offsets_csv(csv_path, offsets_rows)

    vmin, vmax = _display_limits(combined)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(combined, origin='lower', cmap='gray', norm=norm)
    ax.set_title(out_base)
    ax.set_xlabel('X [pix]')
    ax.set_ylabel('Y [pix]')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print()
    print(f'Wrote combined FITS: {fits_path}')
    print(f'Wrote offsets table: {csv_path}')
    print(f'Wrote preview PNG : {png_path}')


if __name__ == '__main__':
    main()
