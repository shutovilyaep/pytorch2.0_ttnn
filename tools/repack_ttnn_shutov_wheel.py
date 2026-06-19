#!/usr/bin/env python3
"""Repack an internal ttnn wheel as ttnn-shutov for public PyPI upload."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_SHA256 = "d68ddb1fd83f558f43b908c10094ebe016664646da9d795844de07a15aedbf09"
DEFAULT_VERSION = "0.62.0.dev20250916"
DEFAULT_WHEEL = f"ttnn-{DEFAULT_VERSION}-cp310-cp310-manylinux_2_34_x86_64.whl"
DEFAULT_URL = f"https://pypi.eng.aws.tenstorrent.com/ttnn/{DEFAULT_WHEEL}"
PROVENANCE = (
    "Provenance build: repackaged from tt-metal ttnn " f"{DEFAULT_VERSION} (Apache-2.0). Import package remains ttnn."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-wheel", type=Path, help="Existing ttnn wheel path")
    parser.add_argument("--download-url", default=DEFAULT_URL, help="Wheel download URL")
    parser.add_argument(
        "--expected-sha256",
        default=DEFAULT_SHA256,
        help="Expected SHA256 of the source wheel",
    )
    parser.add_argument("--version", default=DEFAULT_VERSION, help="ttnn version to repack")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist"),
        help="Directory for the repacked wheel",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_wheel(url: str, destination: Path) -> None:
    import urllib.request

    with urllib.request.urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())


def rewrite_metadata(metadata_path: Path) -> None:
    text = metadata_path.read_text(encoding="utf-8")
    if not text.startswith("Metadata-Version:"):
        raise RuntimeError(f"Unexpected METADATA header in {metadata_path}")

    header, _, body = text.partition("\n\n")
    lines = header.splitlines()
    rewritten_header: list[str] = []
    name_replaced = False
    summary_replaced = False

    for line in lines:
        if line.startswith("Name: "):
            rewritten_header.append("Name: ttnn-shutov")
            name_replaced = True
            continue
        if line.startswith("Summary: "):
            rewritten_header.append(f"Summary: {line.removeprefix('Summary: ')} | {PROVENANCE}")
            summary_replaced = True
            continue
        rewritten_header.append(line)

    if not name_replaced:
        raise RuntimeError(f"Name field not found in {metadata_path}")
    if not summary_replaced:
        rewritten_header.insert(3, f"Summary: {PROVENANCE}")

    metadata_path.write_text(
        "\n".join(rewritten_header) + "\n\n" + body.lstrip("\n"),
        encoding="utf-8",
    )


def repack_wheel(source_wheel: Path, version: str, output_dir: Path) -> Path:
    source_name = source_wheel.name
    match = re.match(r"ttnn-(.+)-cp\d+-cp\d+-.*\.whl$", source_name)
    if not match:
        raise RuntimeError(f"Unexpected source wheel name: {source_name}")
    if match.group(1) != version:
        raise RuntimeError(f"Version mismatch: source wheel has {match.group(1)}, expected {version}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ttnn-shutov-repack-") as tmp:
        tmp_path = Path(tmp)
        unpack_dir = tmp_path / "unpacked"
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", str(source_wheel), "-d", str(unpack_dir)],
            check=True,
        )

        source_root = unpack_dir / f"ttnn-{version}"
        target_root = unpack_dir / f"ttnn_shutov-{version}"
        source_dist_info = source_root / f"ttnn-{version}.dist-info"
        target_dist_info = source_root / f"ttnn_shutov-{version}.dist-info"

        if not source_root.is_dir():
            raise RuntimeError(f"Missing unpacked directory: {source_root}")

        rewrite_metadata(source_dist_info / "METADATA")
        source_dist_info.rename(target_dist_info)
        source_root.rename(target_root)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(target_root),
                "-d",
                str(output_dir),
            ],
            check=True,
        )

    produced = output_dir / f"ttnn_shutov-{version}-cp310-cp310-manylinux_2_34_x86_64.whl"
    if not produced.is_file():
        candidates = sorted(output_dir.glob("ttnn_shutov-*.whl"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one repacked wheel in {output_dir}, found {candidates}")
        produced = candidates[0]

    subprocess.run([sys.executable, "-m", "twine", "check", str(produced)], check=True)
    return produced


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="ttnn-shutov-src-") as tmp:
        tmp_path = Path(tmp)
        source_wheel = args.source_wheel or tmp_path / Path(args.download_url).name
        if args.source_wheel is None:
            download_wheel(args.download_url, source_wheel)

        actual_sha256 = sha256_file(source_wheel)
        if actual_sha256 != args.expected_sha256:
            raise RuntimeError(
                "SHA256 mismatch for source wheel: " f"expected {args.expected_sha256}, got {actual_sha256}"
            )

        produced = repack_wheel(source_wheel, args.version, args.output_dir)
        print(f"Repacked wheel: {produced}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
