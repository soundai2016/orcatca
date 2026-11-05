#!/usr/bin/env python3
"""Publish OrcaTCA results, runs, state (except caches), and all best.pt files.

Sources are never moved or overwritten. The release contains artifacts and
model weights, not training or figure-reproduction scripts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "soundai2016/orcatca"
DIRECTORIES = ("result", "runs", "state")
CANONICAL = Path("result/analysis/transfer_review")
RESULT_FILES = (
    "crossfit_summary.csv", "crossfit_transfer.csv", "crossfit_operating.csv",
    "joint_differences.csv", "calibration_support.csv", "analysis.json",
    "transfer_gap_summary.csv", "transfer_label_summary.csv",
    "transfer_original_operating.csv", "transfer_original_summary.csv",
    "transfer_original_labels.csv", "transfer_threshold_diagnostics.csv",
    "transfer_block_influence.csv", "transfer_bootstrap_audit.json",
    "data_flow.csv", "protocol_audit.json", "main_results.csv",
    "support_sensitivity.csv", "site_metrics.csv", "fit_metrics.csv",
)
DERIVED = (
    "label_error_rates.csv", "transfer_gap_estimates.csv",
    "original_operating_summary.csv", "available_evidence.json",
    "label_gap_contributions.csv", "paired_operating_differences.csv",
    "ranking_sensitivity.csv", "source_check_label_rates.csv",
    "mixture_bounds.csv", "label_mix_sensitivity.csv", "full_label_contrasts.csv",
    "pauc_gain.csv", "transfer_configuration_points.csv", "paired_site_gains.csv",
)
FIGURES = ("figures/Fig1.pdf", "figures/Fig1.png", "figures/Fig2.pdf", "figures/Fig2.png")
SOURCE_EXAMPLE_DIR = Path("result/acoustic_examples")
EXAMPLE_FILES = ("examples.json", "features.npz")
MANIFEST = "release_manifest.json"
MODEL_INDEX = "model_index.csv"
FORMAT = "orcatca-release-v5"
MAX_FILE_SIZE = 50 * 1024**3
# Skip staging areas and Git/Python infrastructure, not legitimate result files.
SKIP_DIRS = {".git", ".hg", ".svn", ".cache", "__pycache__", ".huggingface",
             ".hf_release", ".venv", "venv", "node_modules"}
SENSITIVE_NAMES = {".env", "token", "token.txt", "token.json", ".netrc",
                   "credentials.json", "id_rsa", "id_ed25519", "hf_token"}
SEED_RE = re.compile(r"_s(\d+)$")
FOLD_RE = re.compile(r"fold\d+$")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_file(root: Path, path: Path) -> Path:
    raw = (root / "dclde_2027_killer_whales").resolve()
    if (not path.is_file() or path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or path.resolve().is_relative_to(raw)):
        raise ValueError("Unsafe or missing release input: " + str(path))
    if path.name.lower() in SENSITIVE_NAMES or path.suffix.lower() in {".pem", ".key"}:
        raise ValueError("Potential credential found; refusing public release: " + str(path))
    if path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError("File exceeds 50 GiB; shard before publishing: " + str(path))
    return path


def is_snapshot(path: Path) -> bool:
    name = path.name
    return (name in SKIP_DIRS or name.startswith(".hf-stage-")
            or name.startswith(".hf_release") or name.startswith("hf_release")
            or (path / MANIFEST).is_file())


def safe_output(root: Path, out: Path) -> None:
    raw = (root / "dclde_2027_killer_whales").resolve()
    if out == root or root.is_relative_to(out):
        raise ValueError("Release staging cannot replace or contain the project root")
    if out == raw or out.is_relative_to(raw) or raw.is_relative_to(out):
        raise ValueError("Release staging cannot touch the raw dataset")
    for name in DIRECTORIES:
        source = (root / name).resolve()
        if out == source or out.is_relative_to(source):
            raise ValueError("Staging must be outside " + name + "/")


def collect_tree(root: Path, folder: str) -> tuple[dict[str, Path], int]:
    """Mirror a complete source tree. Only state/ excludes cache directories."""
    base = root / folder
    if not base.is_dir() or base.is_symlink():
        raise FileNotFoundError("Required source directory missing: " + str(base))
    files: dict[str, Path] = {}
    excluded = 0
    for current, dirs, names in os.walk(base, topdown=True, followlinks=False):
        current_path = Path(current)
        kept = []
        for dirname in sorted(dirs):
            child = current_path / dirname
            # Exclude any cache directory beneath state, at any depth.
            if (is_snapshot(child) or (folder == "state" and dirname.lower() == "cache")):
                excluded += 1
                continue
            if child.is_symlink():
                raise ValueError("Symlinked source directory is not publishable: " + str(child))
            kept.append(dirname)
        dirs[:] = kept
        for name in sorted(names):
            path = current_path / name
            if path.is_symlink():
                raise ValueError("Symlinked source file is not publishable: " + str(path))
            files[path.relative_to(root).as_posix()] = input_file(root, path)
    return files, excluded


def find_other_weights(root: Path, mirrored: dict[str, Path], out: Path) -> dict[str, Path]:
    """Find best.pt elsewhere without visiting the raw archive or upload snapshots."""
    found: dict[str, Path] = {}
    raw = (root / "dclde_2027_killer_whales").resolve()
    for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept = []
        for dirname in sorted(dirs):
            child = current_path / dirname
            if (dirname in DIRECTORIES and current_path == root
                    or child.resolve() == raw or child.resolve() == out
                    or out.is_relative_to(child.resolve()) and child.name.startswith(".hf")
                    or is_snapshot(child) or child.is_symlink()):
                continue
            kept.append(dirname)
        dirs[:] = kept
        if "best.pt" not in names:
            continue
        path = current_path / "best.pt"
        key = path.relative_to(root).as_posix()
        if key not in mirrored:
            found[key] = input_file(root, path)
    return found


def figure_files(root: Path, evidence: dict) -> None:
    raw_hashes = evidence.get("figure_sha256", {})
    if not isinstance(raw_hashes, dict):
        raise ValueError("figure_sha256 must be a dictionary")
    expected = {}
    for name, digest in raw_hashes.items():
        key = str(name).replace("\\", "/").removeprefix("./").removeprefix("result/")
        if key in {Path(item).name for item in FIGURES}:
            key = "figures/" + key
        expected[key] = str(digest)
    if set(expected) != set(FIGURES):
        raise ValueError("Incomplete Fig1/Fig2 hashes")
    for name in FIGURES:
        path = input_file(root, root / "result" / name)
        if sha(path).lower() != expected[name].lower():
            raise ValueError("Figure checksum mismatch: " + name)


def verify_completed_study(root: Path, *, source_assets: bool) -> bool:
    """Verify saved transfer results when a completed canonical analysis is present."""
    source = root / CANONICAL
    if not (source / "analysis.json").is_file():
        return False
    data = json.loads((source / "analysis.json").read_text(encoding="utf-8"))
    if data.get("status") != "completed":
        raise ValueError("Canonical transfer analysis is not completed")
    derived = root / "result/figure_data/derived"
    for name in RESULT_FILES:
        input_file(root, source / name)
    for name in DERIVED:
        input_file(root, derived / name)
    evidence = json.loads((derived / "available_evidence.json").read_text(encoding="utf-8"))
    if (evidence.get("mode") != "completed_analysis" or evidence.get("alpha") != .01
            or evidence.get("input_directory") != CANONICAL.as_posix()):
        raise ValueError("Incomplete transfer-review evidence")
    figure_files(root, evidence)
    hashes = evidence.get("input_sha256", {})
    required = {"analysis.json", "main_results.csv", "transfer_gap_summary.csv",
                "transfer_original_labels.csv", "transfer_original_summary.csv",
                "joint_differences.csv", "transfer_label_summary.csv", "crossfit_operating.csv",
                "crossfit_transfer.csv", "site_metrics.csv", "support_sensitivity.csv"}
    if (not isinstance(hashes, dict) or not required.issubset(hashes)
            or any(name not in RESULT_FILES or sha(source / name) != digest
                   for name, digest in hashes.items())):
        raise ValueError("Figure input checksums do not match saved analysis")
    if source_assets:
        context = evidence.get("acoustic_context", {})
        if context.get("mode") == "measured":
            asset_dir = Path(context.get("asset_directory") or SOURCE_EXAMPLE_DIR)
            if asset_dir.is_absolute() or ".." in asset_dir.parts:
                raise ValueError("Unsafe acoustic asset directory")
            if not (root / asset_dir).is_dir():
                asset_dir = SOURCE_EXAMPLE_DIR
            for name in EXAMPLE_FILES:
                digest = context.get("asset_sha256", {}).get(name)
                if not digest or sha(input_file(root, root / asset_dir / name)) != digest:
                    raise ValueError("Acoustic input checksum mismatch: " + name)
        check = root / "scripts/check_label_mixtures.py"
        if check.is_file():
            with TemporaryDirectory() as temporary:
                target = Path(temporary) / "mixture_bounds.csv"
                subprocess.run([sys.executable, str(check), "--analysis", str(source),
                                "--output", str(target)], check=True)
                for name in ("mixture_bounds.csv", "label_mix_sensitivity.csv"):
                    if sha(Path(temporary) / name) != sha(derived / name):
                        raise ValueError("Mixture table checksum mismatch: " + name)
    return True


def weight_rows(stage: Path, hashes: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for name in sorted(hashes):
        path = Path(name)
        if path.name != "best.pt":
            continue
        parts = path.parts
        suite = parts[1] if len(parts) > 2 and parts[0] == "runs" else ""
        fold = next((p for p in parts[:-1] if FOLD_RE.fullmatch(p)), "")
        seed = next((match.group(1) for p in reversed(parts[:-1])
                     if (match := SEED_RE.search(p))), "")
        rows.append({"weight": name, "source": parts[0], "suite": suite,
                     "fold": fold, "seed": seed,
                     "size_mib": f"{(stage / name).stat().st_size / 2**20:.2f}",
                     "sha256": hashes[name]})
    return rows


def write_index(stage: Path, rows: list[dict[str, str]]) -> None:
    with (stage / MODEL_INDEX).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("weight", "source", "suite", "fold",
                                                     "seed", "size_mib", "sha256"))
        writer.writeheader()
        writer.writerows(rows)


def md_cell(value: object) -> str:
    value = str(value if value is not None else "").replace("\r", " ").replace("\n", " ")
    return value.replace("|", "\\|").replace("`", "'")[:90]


def csv_preview(file: Path, max_rows=5, max_columns=7) -> str:
    if not file.is_file():
        return ""
    with file.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        rows = [row for _, row in zip(range(max_rows), reader)]
    if not rows:
        return ""
    priority = ("site", "target", "source", "model", "method", "label", "pauc",
                "auc", "gain", "delta", "difference", "fpr", "tpr", "precision")
    ranked = sorted(headers, key=lambda name: (next((i for i, term in enumerate(priority)
                     if term in name.lower()), len(priority)), headers.index(name)))
    selected = [name for name in headers if name in ranked[:max_columns]]
    output = "| " + " | ".join(map(md_cell, selected)) + " |\n"
    output += "| " + " | ".join("---" for _ in selected) + " |\n"
    for row in rows:
        output += "| " + " | ".join(md_cell(row.get(name, "")) for name in selected) + " |\n"
    return output


def card(stage: Path, rows: list[dict[str, str]], counts: dict[str, int],
         total_bytes: int, audited: bool) -> str:
    lines = ["---", "language: en", "tags:", "- bioacoustics", "- killer-whale",
             "- underwater-acoustics", "---", "", "# OrcaTCA — results and model weights", "",
             "Killer whale acoustic detection and transfer analysis using the",
             "[DCLDE archive](https://doi.org/10.25921/15ey-mh50).",
             "Error rates refer to annotated five-second windows; false-positive rate",
             "is not the number of alarms per hour.", "", "## Contents", "",
             f"- [`result/`](result/): {counts['result']:,} files (figures, metrics and analyses).",
             f"- [`runs/`](runs/): {counts['runs']:,} files (training runs and checkpoints).",
             f"- [`state/`](state/): {counts['state']:,} files (excluding `cache/` directories).",
             f"- **{len(rows):,} `best.pt` checkpoints**, including any found outside those directories.",
             f"- **{total_bytes / 2**30:,.2f} GiB** of original published artifacts.",
             "- [`model_index.csv`](model_index.csv): weight locations, fold/seed and SHA256.",
             "- [`release_manifest.json`](release_manifest.json): complete file checksums.", "",
             "## Model weights", "",
             "Weights are original project-specific PyTorch checkpoints. Per-run metrics",
             "are stored alongside their runs when present. A single checkpoint does not",
             "represent the study-wide mean across runs or seeds.", "",
             "| Location | Weights |", "| --- | ---: |"]
    categories: dict[str, int] = {}
    for row in rows:
        key = row["suite"] if row["suite"] else row["source"]
        categories[key] = categories.get(key, 0) + 1
    for key, count in sorted(categories.items()):
        lines.append(f"| `{md_cell(key)}` | {count} |")
    lines.extend(["", "## Results", ""])
    if audited:
        lines.extend(["The saved 1% source-FPR transfer analysis and its published figure",
                      "inputs were checked against their recorded SHA256 hashes.", ""])
    previews = (("Main results", "result/analysis/transfer_review/main_results.csv"),
                ("Transfer gap", "result/analysis/transfer_review/transfer_gap_summary.csv"),
                ("Paired pAUC", "result/figure_data/derived/pauc_gain.csv"))
    for title, name in previews:
        preview = csv_preview(stage / name)
        if preview:
            lines.extend([f"### {title}", "", f"[`{name}`]({name})", "", preview.rstrip(), ""])
    lines.extend(["The tables are previews only. Source recordings are not included.",
                  "Release paths preserve the original project layout.", ""])
    return "\n".join(lines)


def collect(root: Path, out: Path) -> tuple[dict[str, Path], dict[str, int], int]:
    files: dict[str, Path] = {}
    counts: dict[str, int] = {}
    excluded = 0
    for folder in DIRECTORIES:
        collected, skipped = collect_tree(root, folder)
        files.update(collected)
        counts[folder] = len(collected)
        excluded += skipped
    other = find_other_weights(root, files, out)
    files.update(other)
    return files, counts, excluded


def prepare(root: Path, out: Path, *, hardlink: bool = False) -> dict:
    root = root.resolve(strict=True)
    out = out.resolve()
    safe_output(root, out)
    if out.exists():
        raise FileExistsError(f"Release stage already exists: {out}; choose a new --out")
    files, counts, excluded = collect(root, out)
    if not files:
        raise ValueError("No release artifacts found")
    weights = [key for key in files if Path(key).name == "best.pt"]
    if not weights:
        raise ValueError("No best.pt checkpoints found in or outside result/runs/state")
    audited = verify_completed_study(root, source_assets=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".hf-stage-", dir=out.parent) as temporary:
        stage = Path(temporary)
        for name, source in sorted(files.items()):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if hardlink:
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            else:
                shutil.copy2(source, target)
        hashes = {name: sha(stage / name) for name in sorted(files)}
        rows = weight_rows(stage, hashes)
        write_index(stage, rows)
        total_bytes = sum(source.stat().st_size for source in files.values())
        (stage / "README.md").write_text(card(stage, rows, counts, total_bytes, audited),
                                         encoding="utf-8")
        hashes["README.md"] = sha(stage / "README.md")
        hashes[MODEL_INDEX] = sha(stage / MODEL_INDEX)
        manifest = {"format": FORMAT, "directory_files": counts,
                    "other_weights": len([key for key in weights if key.split("/", 1)[0]
                                           not in DIRECTORIES]),
                    "checkpoint_count": len(rows), "excluded_directories": excluded,
                    "verified_transfer_review": audited, "source_bytes": total_bytes,
                    "sha256": dict(sorted(hashes.items()))}
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
        stage.rename(out)
    return manifest


def verify(out: Path) -> dict:
    manifest_path = out / MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("No release manifest in " + str(out))
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = data.get("sha256", {})
    if (data.get("format") != FORMAT or not isinstance(hashes, dict)
            or not {"README.md", MODEL_INDEX}.issubset(hashes)):
        raise ValueError("Unsupported or incomplete release manifest")
    paths = list(out.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Symlink detected in staged release")
    actual = {path.relative_to(out).as_posix() for path in paths if path.is_file()}
    if actual != set(hashes) | {MANIFEST}:
        raise ValueError("Stage file list differs from manifest")
    counts = {folder: 0 for folder in DIRECTORIES}
    weights: set[str] = set()
    for name in hashes:
        parts = Path(name).parts
        if (not parts or Path(name).is_absolute() or ".." in parts
                or name != Path(*parts).as_posix()):
            raise ValueError("Unsafe path in release manifest: " + name)
        if name in {"README.md", MODEL_INDEX}:
            continue
        if parts[0] in DIRECTORIES and len(parts) > 1:
            folder = parts[0]
            if (folder == "state" and any(part.lower() == "cache" for part in parts[1:-1])):
                raise ValueError("state/cache content must never be published: " + name)
            if any(part in SKIP_DIRS or part.startswith(".hf-stage-")
                   or part.startswith(".hf_release") or part.startswith("hf_release")
                   for part in parts[1:-1]):
                raise ValueError("Upload snapshot/cache staged: " + name)
            counts[folder] += 1
        elif parts[-1] != "best.pt":
            raise ValueError("Outside result/runs/state only best.pt is allowed: " + name)
        if parts[-1] == "best.pt":
            weights.add(name)
    if counts != data.get("directory_files") or len(weights) != data.get("checkpoint_count"):
        raise ValueError("Release file or checkpoint counts do not match manifest")
    if len([x for x in weights if x.split("/", 1)[0] not in DIRECTORIES]) != data.get("other_weights"):
        raise ValueError("Standalone checkpoint count mismatch")
    with (out / MODEL_INDEX).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != len(weights) or {row.get("weight") for row in rows} != weights
            or any(row.get("sha256") != hashes.get(row.get("weight")) for row in rows)):
        raise ValueError("Model index does not match checkpoint files")
    for name, digest in hashes.items():
        if sha(out / name) != digest:
            raise ValueError("SHA256 mismatch: " + name)
    if data.get("verified_transfer_review"):
        verify_completed_study(out, source_assets=False)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="OrcaTCA project root (default parent of scripts/)")
    parser.add_argument("--out", type=Path,
                        help="Fresh release stage (default PROJECT/.hf_release)")
    parser.add_argument("--hardlink", action="store_true",
                        help="Use hardlinks when possible; only for frozen finished results")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Verify existing stage offline")
    mode.add_argument("--upload", action="store_true", help="Prepare, verify and upload")
    mode.add_argument("--upload-only", action="store_true", help="Verify and upload existing stage")
    parser.add_argument("--repo", default=REPOSITORY)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--clean-legacy-layout", action="store_true",
                        help="Also delete old flat analysis/figures/scripts paths on Hugging Face")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = (args.out or root / ".hf_release").resolve()
    if (args.verify_only or args.upload_only) and args.hardlink:
        parser.error("--hardlink applies only when building a new release")
    if not (args.verify_only or args.upload_only):
        built = prepare(root, out, hardlink=args.hardlink)
        print(f"Staged: result={built['directory_files']['result']}, "
              f"runs={built['directory_files']['runs']}, "
              f"state={built['directory_files']['state']}, "
              f"standalone best.pt={built['other_weights']}; "
              f"checkpoints={built['checkpoint_count']}; "
              f"excluded directories={built['excluded_directories']}")
    manifest = verify(out)
    total_bytes = sum((out / path).stat().st_size for path in manifest["sha256"])
    print(f"Verified {len(manifest['sha256']) + 1} files ({total_bytes / 2**30:.2f} GiB) in {out}")
    if not (args.upload or args.upload_only):
        print("Local verification completed; no upload performed")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
    deletions = ["scripts/**"]
    if args.clean_legacy_layout:
        deletions.extend(["analysis/**", "analysis_extra/**", "derived/**",
                          "figure_data/**", "figures/**", "training_records/**",
                          "assets/acoustic_examples/**"])
    api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="model",
                      delete_patterns=deletions,
                      commit_message="Publish OrcaTCA results, runs, state and model weights")
    print("Uploaded to https://huggingface.co/" + args.repo)


if __name__ == "__main__":
    main()
