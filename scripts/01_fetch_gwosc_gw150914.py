"""Fetch GW150914 strain data files from GWOSC.

This script discovers event files from the GWOSC event page and downloads
H1/L1 4kHz HDF5 files into the local raw data directory.

Usage examples:
    python scripts/01_fetch_gwosc_gw150914.py --dry-run
    python scripts/01_fetch_gwosc_gw150914.py
    python scripts/01_fetch_gwosc_gw150914.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_EVENT_PAGES = [
    "https://gwosc.org/eventapi/html/GWTC-1-confident/GW150914/v3/",
    "https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3/",
]

USER_AGENT = "rd-sbi-repro/0.1 (GW150914 fetch script)"
HREF_PATTERN = re.compile(r'href=["\']([^"\']+\.hdf5)["\']', re.IGNORECASE)


@dataclass(frozen=True)
class DownloadRecord:
    url: str
    local_path: str
    size_bytes: int
    sha256: str


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def resolve_event_page(candidate_urls: Iterable[str]) -> tuple[str, str]:
    errors: list[str] = []
    for url in candidate_urls:
        try:
            html = fetch_text(url)
            if "GW150914" in html:
                return url, html
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Unable to access GWOSC event page:\n" + "\n".join(errors))


def extract_hdf5_links(html: str, base_url: str) -> list[str]:
    links: list[str] = []
    for match in HREF_PATTERN.finditer(html):
        href = match.group(1)
        full = urljoin(base_url, href)
        if full.lower().endswith(".hdf5"):
            links.append(full)
    return sorted(set(links))


def select_strain_files(urls: Iterable[str]) -> list[str]:
    selected: list[str] = []
    for url in urls:
        name = Path(urlparse(url).path).name.upper()
        if "_4KHZ_" not in name:
            continue
        if "H-H1_" in name or "L-L1_" in name:
            selected.append(url)
    return sorted(set(selected))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_to(url: str, target_path: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=120) as response, target_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch GW150914 H1/L1 4kHz data from GWOSC")
    parser.add_argument(
        "--event-page",
        type=str,
        default="",
        help="Optional event page URL override. If omitted, known GWOSC URLs are tried.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/gwosc/GW150914"),
        help="Output directory for downloaded HDF5 files.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=32,
        choices=[32, 4096],
        help="Only keep files whose filename duration matches this value.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List selected files without downloading.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    candidates = [args.event_page] if args.event_page else DEFAULT_EVENT_PAGES
    page_url, html = resolve_event_page(candidates)
    all_hdf5 = extract_hdf5_links(html, page_url)
    selected = select_strain_files(all_hdf5)
    selected = [url for url in selected if f"-{args.duration}.hdf5" in url]

    if not selected:
        raise RuntimeError(
            f"No H1/L1 4kHz files found for duration={args.duration}s from page: {page_url}"
        )

    print(f"Resolved event page: {page_url}")
    print(f"Discovered HDF5 files: {len(all_hdf5)}")
    print("Selected files:")
    for url in selected:
        print(f"  - {url}")

    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[DownloadRecord] = []

    for url in selected:
        filename = Path(urlparse(url).path).name
        target = args.output_dir / filename

        if target.exists() and not args.force:
            print(f"Skip existing: {target}")
        else:
            print(f"Download: {url}")
            download_to(url, target)

        records.append(
            DownloadRecord(
                url=url,
                local_path=str(target),
                size_bytes=target.stat().st_size,
                sha256=sha256_file(target),
            )
        )

    manifest_path = args.output_dir / "manifest_gw150914.json"
    payload = {
        "event": "GW150914",
        "event_page": page_url,
        "duration_seconds": args.duration,
        "file_count": len(records),
        "files": [record.__dict__ for record in records],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
