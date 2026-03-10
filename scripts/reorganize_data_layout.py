#!/usr/bin/env python3
"""Reorganize dataset folders to strict layout.

Target layout:
  data/<dataset>/raw       - source files
  data/<dataset>/processed - processed files
  data/<dataset>/catalogs  - hashed catalog cache dirs

Actions:
1) Move top-level entries (except reserved dirs) into raw/.
2) Move source/ contents into raw/ and remove source/.
3) Move raw/<8-hex-hash>/ dirs into catalogs/.

Conflicting files are not overwritten; they are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

HASH_DIR_RE = re.compile(r"^[0-9a-f]{8}$")
RESERVED_NAMES = {"raw", "processed", "catalogs", "source"}


@dataclass
class Stats:
    moved: list[tuple[Path, Path]] = field(default_factory=list)
    dedup_removed: list[tuple[Path, Path]] = field(default_factory=list)
    conflicts: list[tuple[Path, Path, str]] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def same_file(a: Path, b: Path) -> bool:
    return (
        a.is_file()
        and b.is_file()
        and a.stat().st_size == b.stat().st_size
        and sha256(a) == sha256(b)
    )


def _move_file(src: Path, dst: Path, stats: Stats, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        if not dry_run:
            shutil.move(str(src), str(dst))
        stats.moved.append((src, dst))
        return

    if dst.is_file() and same_file(src, dst):
        if not dry_run:
            src.unlink()
        stats.dedup_removed.append((src, dst))
        return

    stats.conflicts.append((src, dst, "file_exists_different"))


def move_entry(src: Path, dst: Path, stats: Stats, dry_run: bool) -> None:
    if not src.exists():
        return

    if src.is_file():
        _move_file(src, dst, stats, dry_run)
        return

    if src.is_dir():
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dry_run:
                shutil.move(str(src), str(dst))
            stats.moved.append((src, dst))
            return

        if not dst.is_dir():
            stats.conflicts.append((src, dst, "dst_not_dir"))
            return

        # Merge recursively.
        for child in sorted(src.iterdir()):
            move_entry(child, dst / child.name, stats, dry_run)

        if not dry_run:
            try:
                src.rmdir()
            except OSError:
                pass
        return

    stats.skipped.append(src)


def move_source_to_raw(dataset_dir: Path, stats: Stats, dry_run: bool) -> None:
    source = dataset_dir / "source"
    raw = dataset_dir / "raw"
    if not source.exists():
        return
    raw.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        move_entry(child, raw / child.name, stats, dry_run)
    if not dry_run:
        try:
            source.rmdir()
        except OSError:
            pass


def move_top_level_to_raw(dataset_dir: Path, stats: Stats, dry_run: bool) -> None:
    raw = dataset_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for entry in sorted(dataset_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.name in RESERVED_NAMES:
            continue
        move_entry(entry, raw / entry.name, stats, dry_run)


def move_hash_dirs_raw_to_catalogs(dataset_dir: Path, stats: Stats, dry_run: bool) -> None:
    raw = dataset_dir / "raw"
    if not raw.exists():
        return
    catalogs = dataset_dir / "catalogs"
    catalogs.mkdir(parents=True, exist_ok=True)
    for entry in sorted(raw.iterdir()):
        if entry.is_dir() and HASH_DIR_RE.fullmatch(entry.name):
            move_entry(entry, catalogs / entry.name, stats, dry_run)


def iter_dataset_dirs(data_root: Path, datasets: list[str] | None) -> list[Path]:
    if datasets:
        return [data_root / name for name in datasets if (data_root / name).is_dir()]

    def looks_like_dataset_dir(p: Path) -> bool:
        return any((p / x).exists() for x in ("raw", "processed", "catalogs"))

    return sorted([p for p in data_root.iterdir() if p.is_dir() and looks_like_dataset_dir(p)])


def reorganize_dataset(dataset_dir: Path, stats: Stats, dry_run: bool) -> None:
    move_source_to_raw(dataset_dir, stats, dry_run)
    move_top_level_to_raw(dataset_dir, stats, dry_run)
    move_hash_dirs_raw_to_catalogs(dataset_dir, stats, dry_run)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reorganize data/* into raw/processed/catalogs layout.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Path to data root directory (default: data)",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names, e.g. FORGE2022 PNR_1z",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without moving files.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data root does not exist: {data_root}")

    dataset_dirs = iter_dataset_dirs(data_root, args.datasets)
    stats = Stats()

    for ds in dataset_dirs:
        reorganize_dataset(ds, stats, dry_run=args.dry_run)

    print(f"data_root: {data_root}")
    print(f"datasets: {[p.name for p in dataset_dirs]}")
    print(f"dry_run: {args.dry_run}")
    print(f"moved: {len(stats.moved)}")
    print(f"dedup_removed: {len(stats.dedup_removed)}")
    print(f"conflicts: {len(stats.conflicts)}")
    print(f"skipped: {len(stats.skipped)}")

    if stats.conflicts:
        print("\n[conflicts]")
        for src, dst, reason in stats.conflicts:
            print(f"- {reason}: {src} -> {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
