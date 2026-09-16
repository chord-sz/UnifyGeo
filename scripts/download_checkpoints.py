#!/usr/bin/env python3
"""Download released UnifyGeo VIGOR checkpoints and verify their hashes."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path


REPOSITORY = "chord-sz/UnifyGeo"
RELEASE_TAG = "v1.0.0-vigor-eval"
ASSETS = {
    "same-area": (
        "unifygeo_vigor_same_area.pth",
        "6443775bd39a84eb4dca04ed5db206be57a8a8fbac6e36a6c045da00a60025ff",
    ),
    "cross-area": (
        "unifygeo_vigor_cross_area.pth",
        "71175956671df210c76a830e4962730188bf0148938e597ef2641945d75cd79d",
    ),
}


def request(url, token=None, accept=None):
    headers = {"User-Agent": "UnifyGeo-checkpoint-downloader"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    if accept:
        headers["Accept"] = accept
    return urllib.request.Request(url, headers=headers)


def private_asset_urls(token):
    api_url = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{RELEASE_TAG}"
    with urllib.request.urlopen(request(api_url, token, "application/vnd.github+json")) as response:
        release = json.load(response)
    return {asset["name"]: asset["url"] for asset in release["assets"]}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(protocol, output_dir, token, private_urls):
    filename, expected_hash = ASSETS[protocol]
    destination = output_dir / filename
    if destination.is_file() and sha256(destination) == expected_hash:
        print(f"Already verified: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    if token:
        url = private_urls.get(filename)
        if url is None:
            raise RuntimeError(f"Release asset is missing: {filename}")
        download_request = request(url, token, "application/octet-stream")
    else:
        url = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{filename}"
        download_request = request(url)
    print(f"Downloading {filename} ...")
    with urllib.request.urlopen(download_request) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    actual_hash = sha256(temporary)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"SHA-256 mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
        )
    temporary.replace(destination)
    print(f"Verified: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("same-area", "cross-area", "all"), default="all")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("GITHUB_TOKEN")
    private_urls = private_asset_urls(token) if token else {}
    protocols = ASSETS if args.protocol == "all" else (args.protocol,)
    try:
        for protocol in protocols:
            download(protocol, args.output_dir, token, private_urls)
    except Exception as exc:
        print(f"Checkpoint download failed: {exc}", file=sys.stderr)
        if not token:
            print(
                "If the repository is private, authenticate with GitHub and set GITHUB_TOKEN.",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
