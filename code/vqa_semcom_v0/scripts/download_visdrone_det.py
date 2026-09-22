#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.cookiejar
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vqa_semcom.config import load_config, resolve_path


def _download_google_drive(file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(out_path), quiet=False, fuzzy=True)
        if out_path.exists() and out_path.stat().st_size >= 1024 * 1024:
            return
    except Exception as exc:
        print(f"gdown fallback unavailable or failed: {exc}")

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = opener.open(url)
    data = response.read()
    text = data[:4096].decode("utf-8", errors="ignore")
    match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
    if match:
        response = opener.open(f"{url}&confirm={match.group(1)}")
        data = response.read()
    out_path.write_bytes(data)
    if out_path.stat().st_size < 1024 * 1024:
        raise RuntimeError(
            f"Downloaded file is too small ({out_path.stat().st_size} bytes). "
            "Google Drive may require manual download."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v0.yaml")
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    zip_path = resolve_path(cfg["download"]["det_val_zip"])
    target_dir = resolve_path(cfg["paths"]["visdrone_val"])
    annotations = target_dir / "annotations"
    if annotations.exists() and not args.force:
        print(f"VisDrone valset already appears extracted at {target_dir}")
        return 0
    if not zip_path.exists() or args.force:
        print(f"Downloading VisDrone DET valset to {zip_path}")
        _download_google_drive(cfg["download"]["det_val_file_id"], zip_path)
    print(f"Extracting {zip_path} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir.parent)
    nested = target_dir.parent / "VisDrone2019-DET-val"
    if nested.exists() and nested != target_dir:
        if target_dir.exists() and not any(target_dir.iterdir()):
            target_dir.rmdir()
        if not target_dir.exists():
            nested.rename(target_dir)
    if not annotations.exists():
        raise RuntimeError(f"Expected annotations directory not found after extraction: {annotations}")
    print("Download/extract complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
