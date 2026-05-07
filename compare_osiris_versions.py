#!/usr/bin/env python3
"""Step through matched spectra from new and old OSIRIS directories."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATE_TOKEN_RE = re.compile(r"^\d{8}(?:T\d{6})?$")
GRISM_TOKEN_RE = re.compile(r"^R\d{4}[A-Z]*$", re.IGNORECASE)
IGNORE_TOKENS = {"tellcorr", "combined", "merged", "clean", "gtc", "osiris"}
TELLCORR_SUFFIX_RE = re.compile(r"_tellcorr(?:_v\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class SpectrumEntry:
    path: Path
    name: str
    name_key: str
    observed_date: date
    grism: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot spectra from a new OSIRIS directory against the closest-date "
            "match from an old OSIRIS directory."
        )
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path("/Users/lluisgalbany/Downloads/_DATA/OSIRIS"),
        help="Directory containing the new spectra.",
    )
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("/Users/lluisgalbany/Downloads/_DATA/OSIRIS_old"),
        help="Directory containing the old spectra.",
    )
    parser.add_argument(
        "--name",
        action="append",
        help="Optional substring filter on the SN/object name. Repeat to include multiple objects.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of matched pairs to show.",
    )
    parser.add_argument(
        "--start-at",
        type=int,
        default=1,
        help="1-based matched-pair index to start from after sorting and deduplication.",
    )
    parser.add_argument(
        "--resume-from",
        help="Start from the first matched pair whose new filename or object name contains this text.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Optional directory where comparison PNGs should be saved.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save plots without opening interactive windows.",
    )
    return parser.parse_args()


def is_metadata_token(token: str) -> bool:
    token_lower = token.lower()
    return (
        bool(DATE_TOKEN_RE.fullmatch(token))
        or bool(GRISM_TOKEN_RE.fullmatch(token))
        or token_lower in IGNORE_TOKENS
        or token_lower == "sloan"
        or (token.startswith("OB") and token[2:].isdigit())
    )


def extract_name(stem: str) -> str:
    tokens = stem.split("_")
    name_tokens: list[str] = []
    for token in tokens:
        if is_metadata_token(token):
            break
        name_tokens.append(token)
    if not name_tokens:
        name_tokens = [tokens[0]]
    return "_".join(name_tokens)


def extract_date(stem: str) -> date | None:
    for token in stem.split("_"):
        if DATE_TOKEN_RE.fullmatch(token):
            return datetime.strptime(token[:8], "%Y%m%d").date()
    return None


def extract_grism(stem: str) -> str | None:
    for token in stem.split("_"):
        if GRISM_TOKEN_RE.fullmatch(token):
            return token.upper()
    return None


def build_entry(path: Path) -> SpectrumEntry | None:
    if path.suffix.lower() != ".txt" or path.name.startswith("."):
        return None
    observed_date = extract_date(path.stem)
    if observed_date is None:
        return None
    name = extract_name(path.stem)
    return SpectrumEntry(
        path=path,
        name=name,
        name_key=name.lower(),
        observed_date=observed_date,
        grism=extract_grism(path.stem),
    )


def strip_tellcorr_suffix(stem: str) -> str:
    return TELLCORR_SUFFIX_RE.sub("", stem)


def find_new_versions(path: Path) -> tuple[Path | None, Path | None]:
    """Return companion original and telluric-corrected files when present."""
    base_stem = strip_tellcorr_suffix(path.stem)
    original_path = path.with_name(f"{base_stem}.txt")
    tellcorr_candidates = sorted(path.parent.glob(f"{base_stem}_tellcorr*.txt"))
    tellcorr_path = tellcorr_candidates[0] if tellcorr_candidates else None

    if not original_path.exists():
        original_path = None
    if tellcorr_path is not None and not tellcorr_path.exists():
        tellcorr_path = None
    return original_path, tellcorr_path


def load_entries(directory: Path, name_filters: list[str] | None) -> list[SpectrumEntry]:
    entries: list[SpectrumEntry] = []
    filter_keys = [item.lower() for item in name_filters] if name_filters else []
    for path in sorted(directory.iterdir()):
        entry = build_entry(path)
        if entry is None:
            continue
        if filter_keys and not any(filter_key in entry.name_key for filter_key in filter_keys):
            continue
        entries.append(entry)
    return entries


def grism_penalty(new_grism: str | None, old_grism: str | None) -> int:
    if not new_grism or not old_grism:
        return 1
    if new_grism == old_grism:
        return 0
    if "BR" in {new_grism, old_grism}:
        return 1
    return 2


def match_entries(
    new_entries: list[SpectrumEntry],
    old_entries: list[SpectrumEntry],
) -> tuple[list[tuple[SpectrumEntry, SpectrumEntry]], list[SpectrumEntry]]:
    old_by_name: dict[str, list[SpectrumEntry]] = defaultdict(list)
    for entry in old_entries:
        old_by_name[entry.name_key].append(entry)

    matches: list[tuple[SpectrumEntry, SpectrumEntry]] = []
    unmatched: list[SpectrumEntry] = []

    for new_entry in sorted(new_entries, key=lambda item: (item.observed_date, item.name_key, item.path.name)):
        candidates = old_by_name.get(new_entry.name_key, [])
        if not candidates:
            unmatched.append(new_entry)
            continue
        best_old = min(
            candidates,
            key=lambda old_entry: (
                abs((old_entry.observed_date - new_entry.observed_date).days),
                grism_penalty(new_entry.grism, old_entry.grism),
                old_entry.path.name,
            ),
        )
        matches.append((new_entry, best_old))

    return matches, unmatched


def deduplicate_matches(
    matches: list[tuple[SpectrumEntry, SpectrumEntry]],
) -> list[tuple[SpectrumEntry, SpectrumEntry]]:
    """Keep one display entry per new-spectrum base filename."""
    deduped: list[tuple[SpectrumEntry, SpectrumEntry]] = []
    seen_keys: set[tuple[str, str]] = set()

    for new_entry, old_entry in matches:
        display_key = (str(new_entry.path.parent), strip_tellcorr_suffix(new_entry.path.stem))
        if display_key in seen_keys:
            continue
        seen_keys.add(display_key)
        deduped.append((new_entry, old_entry))

    return deduped


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rows: list[tuple[float, float, float | None]] = []

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                wavelength = float(parts[0])
                flux = float(parts[1])
                flux_err = float(parts[2]) if len(parts) >= 3 else None
            except ValueError:
                continue
            rows.append((wavelength, flux, flux_err))

    if not rows:
        raise ValueError(f"{path} does not contain any readable spectral rows.")

    wavelength = np.array([row[0] for row in rows], dtype=float)
    flux = np.array([row[1] for row in rows], dtype=float)
    has_errors = any(row[2] is not None for row in rows)
    flux_err = (
        np.array(
            [np.nan if row[2] is None else row[2] for row in rows],
            dtype=float,
        )
        if has_errors
        else None
    )

    mask = np.isfinite(wavelength) & np.isfinite(flux)
    if flux_err is not None:
        finite_err_mask = np.isfinite(flux_err)
        if np.any(finite_err_mask):
            mask &= finite_err_mask
            flux_err = flux_err[mask]
        else:
            flux_err = None

    return wavelength[mask], flux[mask], flux_err


def plot_pair(
    new_entry: SpectrumEntry,
    old_entry: SpectrumEntry,
    index: int,
    total: int,
    save_dir: Path | None = None,
    save_only: bool = False,
) -> str:
    old_wave, old_flux, _ = read_spectrum(old_entry.path)
    new_original_path, new_tellcorr_path = find_new_versions(new_entry.path)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(old_wave, old_flux, lw=1.0, color="#d95f02", label=f"Old: {old_entry.path.name}")
    plotted_new = False

    if new_original_path is not None:
        new_wave, new_flux, new_flux_err = read_spectrum(new_original_path)
        ax.plot(new_wave, new_flux, lw=1.0, color="#7570b3", label=f"New original: {new_original_path.name}")
        if new_flux_err is not None:
            ax.fill_between(
                new_wave,
                new_flux - new_flux_err,
                new_flux + new_flux_err,
                color="#7570b3",
                alpha=0.14,
                linewidth=0,
                label="New original uncertainty",
            )
        plotted_new = True

    if new_tellcorr_path is not None and new_tellcorr_path != new_original_path:
        new_wave, new_flux, new_flux_err = read_spectrum(new_tellcorr_path)
        ax.plot(
            new_wave,
            new_flux,
            lw=1.0,
            color="#1b9e77",
            label=f"New tellcorr: {new_tellcorr_path.name}",
        )
        if new_flux_err is not None:
            ax.fill_between(
                new_wave,
                new_flux - new_flux_err,
                new_flux + new_flux_err,
                color="#1b9e77",
                alpha=0.18,
                linewidth=0,
                label="New tellcorr uncertainty",
            )
        plotted_new = True

    if not plotted_new:
        new_wave, new_flux, new_flux_err = read_spectrum(new_entry.path)
        ax.plot(new_wave, new_flux, lw=1.0, color="#1b9e77", label=f"New: {new_entry.path.name}")
        if new_flux_err is not None:
            ax.fill_between(
                new_wave,
                new_flux - new_flux_err,
                new_flux + new_flux_err,
                color="#1b9e77",
                alpha=0.18,
                linewidth=0,
                label="New uncertainty",
            )
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
    ax.set_title(
        f"{new_entry.name}  [{index}/{total}]\n"
        f"Date match: {new_entry.observed_date.isoformat()} vs "
        f"{old_entry.observed_date.isoformat()} "
        f"({abs((old_entry.observed_date - new_entry.observed_date).days)} d)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    if save_dir is not None:
        out_png = save_dir / f"{index:03d}_{new_entry.path.stem}_comparison.png"
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        print(f"  Saved comparison plot → {out_png}")

    if save_only:
        plt.close(fig)
        return "next"

    print(
        f"\n[{index}/{total}] {new_entry.path.name}\n"
        f"  old match: {old_entry.path.name}\n"
        f"  new original: {new_original_path.name if new_original_path is not None else 'not available'}\n"
        f"  new tellcorr: {new_tellcorr_path.name if new_tellcorr_path is not None else 'not available'}\n"
        "  Press Enter in the terminal for the next spectrum, or q to quit."
    )
    plt.show(block=False)
    plt.pause(0.1)
    try:
        response = input("Next spectrum? [Enter/q]: ").strip().lower()
    except EOFError:
        response = "q"
    plt.close(fig)
    return "quit" if response in {"q", "quit", "exit"} else "next"


def main() -> int:
    args = parse_args()

    if not args.new_dir.is_dir():
        print(f"New directory not found: {args.new_dir}", file=sys.stderr)
        return 1
    if not args.old_dir.is_dir():
        print(f"Old directory not found: {args.old_dir}", file=sys.stderr)
        return 1
    if args.save_only and args.save_dir is None:
        print("--save-only requires --save-dir.", file=sys.stderr)
        return 1
    if args.save_dir is not None and not args.save_dir.is_dir():
        print(f"Save directory not found: {args.save_dir}", file=sys.stderr)
        return 1

    new_entries = load_entries(args.new_dir, args.name)
    old_entries = load_entries(args.old_dir, args.name)

    if not new_entries:
        print("No new `.txt` spectra found with the requested filter.", file=sys.stderr)
        return 1
    if not old_entries:
        print("No old `.txt` spectra found with the requested filter.", file=sys.stderr)
        return 1

    matches, unmatched = match_entries(new_entries, old_entries)
    matches = deduplicate_matches(matches)

    print(f"Loaded {len(new_entries)} new spectra and {len(old_entries)} old spectra.")
    print(f"Matched {len(matches)} new spectra by SN/object name and nearest filename date.")
    if unmatched:
        print(f"Skipped {len(unmatched)} new spectra with no old match.")
        for entry in unmatched[:10]:
            print(f"  - {entry.path.name}")
        if len(unmatched) > 10:
            print("  - ...")

    if not matches:
        print("No matched spectra to display.", file=sys.stderr)
        return 1

    total_matches = len(matches)
    start_index = max(args.start_at, 1)
    if args.resume_from:
        resume_key = args.resume_from.lower()
        matched_index = next(
            (
                idx
                for idx, (new_entry, _) in enumerate(matches, start=1)
                if resume_key in new_entry.path.name.lower() or resume_key in new_entry.name_key
            ),
            None,
        )
        if matched_index is None:
            print(
                f"No matched spectrum found for --resume-from={args.resume_from!r}.",
                file=sys.stderr,
            )
            return 1
        start_index = matched_index

    if start_index > total_matches:
        print(
            f"--start-at={start_index} is beyond the available matched spectra ({total_matches}).",
            file=sys.stderr,
        )
        return 1

    matches = matches[start_index - 1 :]
    if args.limit is not None:
        matches = matches[: args.limit]

    for index, (new_entry, old_entry) in enumerate(matches, start=1):
        display_index = start_index + index - 1
        action = plot_pair(
            new_entry,
            old_entry,
            display_index,
            total_matches,
            save_dir=args.save_dir,
            save_only=args.save_only,
        )
        if action == "quit":
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
