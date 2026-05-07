#!/usr/bin/env python3
"""Migrate old-pipeline OSIRIS ASCII spectra into the main OSIRIS folder."""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


TELLURIC_BANDS = [
    (6855, 6940),
    (7155, 7332),
    (7580, 7690),
    (8110, 8357),
]

GRISM_RE = re.compile(r"^R1000[BR]$", re.IGNORECASE)
DATE_RE = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class RawEntry:
    path: Path
    raw_id: str
    object_name: str
    object_key: str
    grism: str
    raw_date: str | None
    date_obs: str | None
    date_obs_dt: datetime | None
    exptime: str
    airmass: str
    telescope: str
    instrument: str


@dataclass(frozen=True)
class OldSpectrum:
    path: Path
    object_name: str
    object_key: str
    date_token: str
    grism: str | None
    occurrence: int | None
    tellcorr: bool
    clean: bool


@dataclass(frozen=True)
class MatchResult:
    old: OldSpectrum
    rows: np.ndarray
    metadata_source: str
    metadata_note: str | None
    telescope: str
    instrument: str
    object_name: str
    grism: str
    date_obs: str
    exptime: str
    airmass: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("/Users/lluisgalbany/Downloads/_DATA/OSIRIS_old/old"),
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path("/Users/lluisgalbany/Downloads/_DATA/OSIRIS"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/Users/lluisgalbany/Downloads/_DATA/_gtc_osiris"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/Users/lluisgalbany/Downloads/_DATA/OSIRIS_old/old_migration_report.txt"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not move or write files; just print the planned actions.",
    )
    return parser.parse_args()


def normalize_object_name(name: str) -> str:
    key = name.lower().strip()
    key = re.sub(r"_\d+$", "", key)
    return key


def read_numeric_rows(path: Path) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                wave = float(parts[0])
                flux = float(parts[1])
            except ValueError:
                continue
            rows.append((wave, flux))
    if not rows:
        raise ValueError(f"{path} does not contain readable spectral rows.")
    return np.asarray(rows, dtype=float)


def parse_old_spectrum(path: Path) -> OldSpectrum:
    tokens = path.stem.split("_")
    date_index = next((idx for idx, token in enumerate(tokens) if DATE_RE.fullmatch(token)), None)
    if date_index is None:
        raise ValueError(f"Could not find YYYYMMDD token in {path.name}")

    object_tokens = tokens[:date_index]
    if object_tokens and GRISM_RE.fullmatch(object_tokens[-1]):
        object_tokens = object_tokens[:-1]
    object_name = "_".join(object_tokens)
    object_key = normalize_object_name(object_name)

    grism = next((token.upper() for token in tokens if GRISM_RE.fullmatch(token)), None)
    occurrence = None
    if date_index + 1 < len(tokens) and tokens[date_index + 1].isdigit():
        occurrence = int(tokens[date_index + 1])

    stem_lower = path.stem.lower()
    return OldSpectrum(
        path=path,
        object_name=object_name,
        object_key=object_key,
        date_token=tokens[date_index],
        grism=grism,
        occurrence=occurrence,
        tellcorr="tellcorr" in stem_lower,
        clean=stem_lower.endswith("_clean"),
    )


def load_raw_index(raw_root: Path) -> dict[str, list[RawEntry]]:
    index: dict[str, list[RawEntry]] = {}
    for path in sorted(raw_root.rglob("object/*.fits")):
        match = re.match(r"(\d+)-(\d{8})-OSIRIS-", path.name)
        raw_id = match.group(1) if match else path.stem
        raw_date = match.group(2) if match else None
        try:
            with fits.open(path) as hdul:
                header = hdul[0].header
        except Exception:
            continue

        object_name = str(header.get("OBJECT") or header.get("TARGET") or header.get("OBJNAME") or "").strip()
        grism = str(header.get("GRISM") or header.get("DISPNAME") or "").strip().upper()
        date_obs = header.get("DATE-OBS")
        date_obs_dt = None
        if date_obs:
            try:
                date_obs_dt = datetime.fromisoformat(str(date_obs))
            except ValueError:
                date_obs_dt = None
        if not object_name or not grism.startswith("R1000"):
            continue

        entry = RawEntry(
            path=path,
            raw_id=raw_id,
            object_name=object_name,
            object_key=normalize_object_name(object_name),
            grism=grism,
            raw_date=raw_date,
            date_obs=str(date_obs) if date_obs else None,
            date_obs_dt=date_obs_dt,
            exptime=str(header.get("EXPTIME", "UNKNOWN")),
            airmass=str(header.get("AIRMASS", "UNKNOWN")),
            telescope=str(header.get("TELESCOP") or header.get("TELESCOPE") or "GTC"),
            instrument=str(header.get("INSTRUME") or header.get("INSTRUMENT") or "OSIRIS"),
        )
        index.setdefault(entry.object_key, []).append(entry)

    for entries in index.values():
        entries.sort(key=lambda item: (item.date_obs_dt or datetime.min, item.raw_id))
    return index


def infer_grism_from_rows(rows: np.ndarray) -> str | None:
    wave_min = float(rows[:, 0].min())
    wave_max = float(rows[:, 0].max())
    if wave_min < 4500 and wave_max > 9000:
        return "R1000BR"
    if wave_max < 8000:
        return "R1000B"
    if wave_min > 5000:
        return "R1000R"
    return None


def candidates_for(old: OldSpectrum, raw_index: dict[str, list[RawEntry]]) -> list[RawEntry]:
    raw_entries = raw_index.get(old.object_key, [])
    return [
        entry
        for entry in raw_entries
        if old.date_token in {entry.raw_date, (entry.date_obs or "")[:10].replace("-", "")}
    ]


def select_arm_candidate(
    old: OldSpectrum,
    entries: list[RawEntry],
    grism: str,
) -> RawEntry | None:
    arm_entries = [entry for entry in entries if entry.grism == grism]
    if not arm_entries:
        return None
    if old.occurrence is not None and 1 <= old.occurrence <= len(arm_entries):
        return arm_entries[old.occurrence - 1]
    return arm_entries[0]


def merged_date_obs(entries: list[RawEntry]) -> str:
    dated = [entry.date_obs_dt for entry in entries if entry.date_obs_dt is not None]
    if not dated:
        return "UNKNOWN"
    avg_ts = sum(item.timestamp() for item in dated) / len(dated)
    return datetime.fromtimestamp(avg_ts).replace(microsecond=0).isoformat()


def build_match_result(old: OldSpectrum, rows: np.ndarray, raw_index: dict[str, list[RawEntry]]) -> MatchResult:
    inferred_grism = old.grism or infer_grism_from_rows(rows) or "UNKNOWN"
    entries = candidates_for(old, raw_index)

    if inferred_grism == "R1000BR":
        blue = select_arm_candidate(old, entries, "R1000B")
        red = select_arm_candidate(old, entries, "R1000R")
        if blue is not None and red is not None:
            return MatchResult(
                old=old,
                rows=rows,
                metadata_source="raw",
                metadata_note=None,
                telescope=blue.telescope,
                instrument=blue.instrument,
                object_name=blue.object_name,
                grism="R1000BR",
                date_obs=merged_date_obs([blue, red]),
                exptime=blue.exptime,
                airmass=blue.airmass,
            )

    if inferred_grism in {"R1000B", "R1000R"}:
        arm = select_arm_candidate(old, entries, inferred_grism)
        if arm is not None:
            return MatchResult(
                old=old,
                rows=rows,
                metadata_source="raw",
                metadata_note=None,
                telescope=arm.telescope,
                instrument=arm.instrument,
                object_name=arm.object_name,
                grism=inferred_grism,
                date_obs=arm.date_obs or old.date_token,
                exptime=arm.exptime,
                airmass=arm.airmass,
            )

    fallback_date = f"{old.date_token[:4]}-{old.date_token[4:6]}-{old.date_token[6:]}"
    return MatchResult(
        old=old,
        rows=rows,
        metadata_source="fallback",
        metadata_note=f"No raw match found under {Path('/Users/lluisgalbany/Downloads/_DATA/_gtc_osiris')}",
        telescope="GTC",
        instrument="OSIRIS",
        object_name=old.object_name,
        grism=inferred_grism,
        date_obs=fallback_date,
        exptime="UNKNOWN",
        airmass="UNKNOWN",
    )


def build_header(result: MatchResult) -> str:
    lines = [
        f"TELESCOPE : {result.telescope}",
        f"INSTRUMENT: {result.instrument}",
        f"OBJECT    : {result.object_name}",
        f"DATE-OBS  : {result.date_obs}",
        f"EXPTIME   : {result.exptime} s" if result.exptime != "UNKNOWN" else "EXPTIME   : UNKNOWN",
        f"AIRMASS   : {result.airmass}",
        "COLUMNS   : wavelength[AA]  flux[erg/s/cm2/AA]",
    ]
    if result.old.tellcorr:
        lines.append("TELLURIC  : corrected")
    if result.grism != "UNKNOWN":
        lines.append(f"GRISM     : {result.grism}")
    if result.metadata_note:
        lines.append(f"RAW_MATCH : {result.metadata_note}")
    return "\n".join(lines)


def write_spectrum(path: Path, result: MatchResult) -> None:
    header = build_header(result)
    np.savetxt(path, result.rows, fmt="%.6f %.6e", header=header)


def plot_spectrum(txt_path: Path, result: MatchResult) -> None:
    rows = result.rows
    wave = rows[:, 0]
    flux = rows[:, 1]

    fig, ax = plt.subplots(figsize=(14, 5))
    color = "seagreen" if result.old.tellcorr else "steelblue"
    label = txt_path.stem.replace("_", " ")
    ax.plot(wave, flux, color=color, lw=1.2, label=label)
    for idx, (w0, w1) in enumerate(TELLURIC_BANDS):
        ax.axvspan(w0, w1, alpha=0.08, color="steelblue", label="Telluric bands" if idx == 0 else "")
    ax.set_xlabel("Wavelength [Å]", fontsize=12)
    ax.set_ylabel("Flux [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]", fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(txt_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def ensure_no_conflicts(destinations: list[Path]) -> None:
    conflicts = [path for path in destinations if path.exists()]
    if conflicts:
        names = ", ".join(path.name for path in conflicts[:10])
        raise FileExistsError(f"Destination files already exist: {names}")


def write_report(report_path: Path, moved: list[MatchResult], conflicts: list[str]) -> None:
    raw_count = sum(item.metadata_source == "raw" for item in moved)
    fallback_count = sum(item.metadata_source == "fallback" for item in moved)
    lines = [
        f"Moved spectra: {len(moved)}",
        f"Metadata from raw headers: {raw_count}",
        f"Fallback metadata: {fallback_count}",
        "",
        "Files:",
    ]
    for item in moved:
        lines.append(
            f"- {item.old.path.name} [{item.metadata_source}]"
            + (f" | {item.metadata_note}" if item.metadata_note else "")
        )
    if conflicts:
        lines.extend(["", "Conflicts:", *[f"- {name}" for name in conflicts]])
    report_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()

    raw_index = load_raw_index(args.raw_root)
    old_files = sorted(args.old_dir.glob("*.txt"))
    if not old_files:
        print(f"No old .txt files found in {args.old_dir}")
        return 1

    old_specs = [parse_old_spectrum(path) for path in old_files]
    results = [build_match_result(spec, read_numeric_rows(spec.path), raw_index) for spec in old_specs]

    destinations = [args.new_dir / item.old.path.name for item in results]
    if not args.dry_run:
        ensure_no_conflicts(destinations)

    raw_count = sum(item.metadata_source == "raw" for item in results)
    fallback_count = sum(item.metadata_source == "fallback" for item in results)
    print(f"Prepared {len(results)} spectra.")
    print(f"  Raw metadata matches : {raw_count}")
    print(f"  Fallback metadata    : {fallback_count}")

    if args.dry_run:
        for item in results[:20]:
            print(f"{item.old.path.name} -> {item.metadata_source} ({item.grism}, {item.date_obs})")
        return 0

    moved: list[MatchResult] = []
    conflicts: list[str] = []

    for result in results:
        source = result.old.path
        dest = args.new_dir / source.name
        if dest.exists():
            conflicts.append(dest.name)
            continue
        shutil.move(str(source), str(dest))
        moved_result = MatchResult(
            old=OldSpectrum(
                path=dest,
                object_name=result.old.object_name,
                object_key=result.old.object_key,
                date_token=result.old.date_token,
                grism=result.old.grism,
                occurrence=result.old.occurrence,
                tellcorr=result.old.tellcorr,
                clean=result.old.clean,
            ),
            rows=result.rows,
            metadata_source=result.metadata_source,
            metadata_note=result.metadata_note,
            telescope=result.telescope,
            instrument=result.instrument,
            object_name=result.object_name,
            grism=result.grism,
            date_obs=result.date_obs,
            exptime=result.exptime,
            airmass=result.airmass,
        )
        write_spectrum(dest, moved_result)
        plot_spectrum(dest, moved_result)
        moved.append(moved_result)

    write_report(args.report, moved, conflicts)

    print(f"Moved {len(moved)} spectra into {args.new_dir}")
    print(f"Report written to {args.report}")
    if conflicts:
        print(f"Skipped {len(conflicts)} conflicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
