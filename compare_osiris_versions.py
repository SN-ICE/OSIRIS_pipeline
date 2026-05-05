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
        help="Optional substring filter on the SN/object name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of matched pairs to show.",
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


def load_entries(directory: Path, name_filter: str | None) -> list[SpectrumEntry]:
    entries: list[SpectrumEntry] = []
    filter_key = name_filter.lower() if name_filter else None
    for path in sorted(directory.iterdir()):
        entry = build_entry(path)
        if entry is None:
            continue
        if filter_key and filter_key not in entry.name_key:
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


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 2:
        raise ValueError(f"{path} does not contain at least two columns.")
    wavelength = data[:, 0]
    flux = data[:, 1]
    mask = np.isfinite(wavelength) & np.isfinite(flux)
    return wavelength[mask], flux[mask]


def plot_pair(
    new_entry: SpectrumEntry,
    old_entry: SpectrumEntry,
    index: int,
    total: int,
) -> str:
    new_wave, new_flux = read_spectrum(new_entry.path)
    old_wave, old_flux = read_spectrum(old_entry.path)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(old_wave, old_flux, lw=1.0, color="#d95f02", label=f"Old: {old_entry.path.name}")
    ax.plot(new_wave, new_flux, lw=1.0, color="#1b9e77", label=f"New: {new_entry.path.name}")
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

    print(
        f"\n[{index}/{total}] {new_entry.path.name}\n"
        f"  old match: {old_entry.path.name}\n"
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

    new_entries = load_entries(args.new_dir, args.name)
    old_entries = load_entries(args.old_dir, args.name)

    if not new_entries:
        print("No new `.txt` spectra found with the requested filter.", file=sys.stderr)
        return 1
    if not old_entries:
        print("No old `.txt` spectra found with the requested filter.", file=sys.stderr)
        return 1

    matches, unmatched = match_entries(new_entries, old_entries)
    if args.limit is not None:
        matches = matches[: args.limit]

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

    for index, (new_entry, old_entry) in enumerate(matches, start=1):
        action = plot_pair(new_entry, old_entry, index, len(matches))
        if action == "quit":
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
