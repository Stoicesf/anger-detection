#!/usr/bin/env python3
"""Migrate anger-detection repo to standard src layout (v2).

Safety:
  - Copies the tree (excluding .git) to ../backup_anger_detection_<unix_ts>/
  - Aborts on destination conflicts before any move
  - Prefer `git mv` when inside a git work tree; else shutil.move

Usage:
  python tools/migrate_to_v2_layout.py --dry-run
  python tools/migrate_to_v2_layout.py
  python tools/migrate_to_v2_layout.py --skip-backup   # only if backup already exists
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# New empty dirs to ensure exist (even if no moves land there yet)
ENSURE_DIRS = [
    "src/anger_detection",
    "src/anger_detection/features",
    "src/anger_detection/models",
    "src/anger_detection/losses",
    "src/anger_detection/decision",
    "src/anger_detection/inference",
    "apps/streamlit_demo",
    "scripts/data",
    "scripts/train",
    "scripts/eval",
    "scripts/export",
    "data/raw",
    "data/processed",
    "data/curated",
    "artifacts/models",
    "artifacts/reports",
    "tests",
    "configs",
    "tools",
]

# (relative_src, relative_dst) — files and directories
MOVES: list[tuple[str, str]] = [
    # Core library
    ("scripts/common.py", "src/anger_detection/common.py"),
    ("scripts/model.py", "src/anger_detection/models/model_tf.py"),
    ("scripts/model_anger.py", "src/anger_detection/models/model_anger.py"),
    ("scripts/losses.py", "src/anger_detection/losses/focal.py"),
    ("demo/decision.py", "src/anger_detection/decision/decision.py"),
    ("demo/live_monitor.py", "src/anger_detection/decision/live_monitor.py"),
    # Apps / tests
    ("demo/realtime_emotion_demo.py", "apps/streamlit_demo/realtime_emotion_demo.py"),
    ("demo/test_decision.py", "tests/test_decision.py"),
    # scripts/data
    ("scripts/download_data.py", "scripts/data/download_data.py"),
    ("scripts/download_chinese_data.py", "scripts/data/download_chinese_data.py"),
    ("scripts/prepare_data.py", "scripts/data/prepare_data.py"),
    ("scripts/prepare_data_chinese.py", "scripts/data/prepare_data_chinese.py"),
    ("scripts/build_hard_negative.py", "scripts/data/build_hard_negative.py"),
    ("scripts/retry_esd.py", "scripts/data/retry_esd.py"),
    ("scripts/run_download_chinese.cmd", "scripts/data/run_download_chinese.cmd"),
    ("scripts/run_retry_esd.cmd", "scripts/data/run_retry_esd.cmd"),
    # scripts/train
    ("scripts/train.py", "scripts/train/train.py"),
    ("scripts/train_torch.py", "scripts/train/train_torch.py"),
    ("scripts/train_anger_binary.py", "scripts/train/train_anger_binary.py"),
    ("scripts/finetune_casia.py", "scripts/train/finetune_casia.py"),
    ("scripts/finetune_hard_negative.py", "scripts/train/finetune_hard_negative.py"),
    ("scripts/run_train.cmd", "scripts/train/run_train.cmd"),
    ("scripts/run_train_torch.cmd", "scripts/train/run_train_torch.cmd"),
    # scripts/eval
    ("scripts/eval_anger.py", "scripts/eval/eval_anger.py"),
    ("scripts/eval_realtime_metrics.py", "scripts/eval/eval_realtime_metrics.py"),
    ("scripts/evaluate_tflite.py", "scripts/eval/evaluate_tflite.py"),
    ("scripts/predict_wav.py", "scripts/eval/predict_wav.py"),
    ("scripts/test_mic.py", "scripts/eval/test_mic.py"),
    # scripts/export
    ("scripts/export_tflite_torch.py", "scripts/export/export_tflite_torch.py"),
    ("scripts/quantize.py", "scripts/export/quantize.py"),
    # curated datasets
    ("dataset/hard_negative", "data/curated/hard_negative"),
    ("dataset/home_anger", "data/curated/home_anger"),
]

# Directory tree moves (entire folder contents via rename)
DIR_MOVES: list[tuple[str, str]] = [
    ("models", "artifacts/models"),
    ("output", "artifacts/reports"),
]


def in_git_repo(path: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"
    except OSError:
        return False


def backup_tree(root: Path, dry_run: bool) -> Path:
    ts = int(time.time())
    dest = root.parent / f"backup_anger_detection_{ts}"
    print(f"[backup] {root} -> {dest}")
    if dry_run:
        return dest

    def ignore(directory: str, names: list[str]) -> set[str]:
        skip = {".git", "__pycache__", ".numba_cache"}
        return {n for n in names if n in skip or n.endswith(".pyc")}

    shutil.copytree(root, dest, ignore=ignore)
    print(f"[backup] done ({dest})")
    return dest


def collect_moves() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for src_rel, dst_rel in MOVES:
        pairs.append((ROOT / src_rel, ROOT / dst_rel))
    for src_rel, dst_rel in DIR_MOVES:
        pairs.append((ROOT / src_rel, ROOT / dst_rel))
    return pairs


def check_conflicts(pairs: list[tuple[Path, Path]]) -> list[str]:
    errors: list[str] = []
    for src, dst in pairs:
        if not src.exists():
            print(f"[skip-missing] {src.relative_to(ROOT)}")
            continue
        if not dst.exists():
            continue
        if src.resolve() == dst.resolve():
            continue
        if dst.is_file():
            errors.append(f"DST exists (file): {dst.relative_to(ROOT)}  (from {src.relative_to(ROOT)})")
        elif dst.is_dir():
            children = [c for c in dst.iterdir() if c.name != ".gitkeep"]
            if children:
                errors.append(
                    f"DST exists (non-empty dir): {dst.relative_to(ROOT)} "
                    f"children={[c.name for c in children[:5]]} "
                    f"(from {src.relative_to(ROOT)})"
                )
            # empty dir (or only .gitkeep) is OK — will be replaced on move
    return errors


def ensure_dirs(dry_run: bool) -> None:
    for rel in ENSURE_DIRS:
        p = ROOT / rel
        print(f"[mkdir] {rel}")
        if not dry_run:
            p.mkdir(parents=True, exist_ok=True)


def move_one(src: Path, dst: Path, use_git: bool, dry_run: bool) -> None:
    if not src.exists():
        return
    rel_s = src.relative_to(ROOT)
    rel_d = dst.relative_to(ROOT)
    print(f"[mv] {rel_s} -> {rel_d}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.is_dir():
        for gk in dst.glob(".gitkeep"):
            gk.unlink(missing_ok=True)
        if not any(dst.iterdir()):
            dst.rmdir()
    if use_git:
        # git mv works for tracked; for untracked fall back
        r = subprocess.run(
            ["git", "mv", "-f", str(src), str(dst)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return
        # untracked or rename across — plain move then git add
        shutil.move(str(src), str(dst))
        subprocess.run(["git", "add", "-A", str(dst)], cwd=ROOT, check=False)
    else:
        shutil.move(str(src), str(dst))


def cleanup_empty(paths: list[str], dry_run: bool) -> None:
    for rel in paths:
        p = ROOT / rel
        if not p.exists() or not p.is_dir():
            continue
        leftover = list(p.iterdir())
        # drop pycache
        for c in list(leftover):
            if c.name == "__pycache__":
                if not dry_run:
                    shutil.rmtree(c, ignore_errors=True)
                leftover = [x for x in leftover if x != c]
        if not leftover:
            print(f"[rmdir] {rel}")
            if not dry_run:
                p.rmdir()


def write_gitkeeps(dry_run: bool) -> None:
    keep_dirs = [
        "data/processed",
        "data/curated",
        "artifacts/reports",
        "configs",
        "src/anger_detection/inference",
        "src/anger_detection/features",
    ]
    for rel in keep_dirs:
        p = ROOT / rel / ".gitkeep"
        print(f"[gitkeep] {p.relative_to(ROOT)}")
        if not dry_run:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Print plan only")
    ap.add_argument("--skip-backup", action="store_true", help="Skip tree backup")
    args = ap.parse_args()
    dry = args.dry_run

    print(f"ROOT = {ROOT}")
    if not args.skip_backup:
        backup_tree(ROOT, dry_run=dry)
    else:
        print("[backup] skipped")

    ensure_dirs(dry)
    pairs = collect_moves()

    # Filter to existing sources for conflict check
    existing = [(s, d) for s, d in pairs if s.exists()]
    conflicts = check_conflicts(existing)
    if conflicts:
        print("\n[CONFLICT] abort — fix these first:")
        for c in conflicts:
            print(f"  - {c}")
        return 1

    use_git = in_git_repo(ROOT)
    print(f"[git] use_git_mv={use_git}")

    for src, dst in existing:
        move_one(src, dst, use_git=use_git, dry_run=dry)

    write_gitkeeps(dry)
    cleanup_empty(["demo", "dataset", "scripts", "models", "output"], dry)

    print("\n[done] migration mapping applied." + (" (dry-run)" if dry else ""))
    print("Next: fix imports, pyproject.toml, docs (see plan).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
