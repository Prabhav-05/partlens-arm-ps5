"""Fetch verified build assets or the metal-nut training category."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(".local")
HUB = "https://huggingface.co"
DATA_REPO = "MSherbinii/mvtec-ad-metal-nut"
DATA_REVISION = "8db7371c35a94709cbb3e69ac69f80d2716e2f8a"
LLAMA_REPO = "software-mansion/react-native-executorch-llama-3.2"
LLAMA_REVISION = "14abd0ac2a4c559114cb4f2944c51249d6656ed5"
RUNTIME_URL = "https://ossci-android.s3.amazonaws.com/executorch/release/1.4.1-xnnpack/executorch.aar"
ARTIFACTS = {
    "runtime": [(RUNTIME_URL, ROOT / "runtime/executorch.aar",
                 "8956c3e8eab9a55d8312002fab2205b401174694a478f4bc477dddca90d239f6")],
    "llama": [
        (f"{HUB}/{LLAMA_REPO}/resolve/{LLAMA_REVISION}/tokenizer.json", ROOT / "models/tokenizer.json",
         "0b9897f5668a5d202662c4bab2be785eb987daf194b557315a690f3d2dff1ce0"),
        (f"{HUB}/{LLAMA_REPO}/resolve/{LLAMA_REVISION}/1b/xnnpack/llama_3_2_1b_xnnpack_spinquant.pte",
         ROOT / "models/llama.pte", "cd9b0bcc5a1d71072a818b9237dba45fc84fdf7b244fb455f63806bfc7111ad2"),
    ],
}


def verify(path, digest, size):
    if not path.is_file() or size is not None and path.stat().st_size != size:
        return False
    if digest is not None:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest() == digest
    return True


def download(url, target, digest=None, size=None):
    if verify(target, digest, size):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".partial")
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response, pending.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if not verify(pending, digest, size):
                raise OSError(f"File check failed for {target.name}")
            pending.replace(target)
            return
        except OSError:
            pending.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def fetch_dataset():
    tree = f"{HUB}/api/datasets/{DATA_REPO}/tree/{DATA_REVISION}/metal_nut?recursive=true&limit=1000"
    with urlopen(tree, timeout=30) as response:
        entries = json.load(response)
    files = [entry for entry in entries if entry["type"] == "file" and Path(entry["path"]).suffix in {".png", ".txt"}]
    if len(files) != 430:
        raise RuntimeError(f"Expected 430 files, got {len(files)}")

    def fetch(entry):
        path = entry["path"]
        digest = entry.get("lfs", {}).get("oid")
        download(f"{HUB}/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/{path}",
                 ROOT / "data" / path, digest, entry["size"])

    with ThreadPoolExecutor(max_workers=12) as pool:
        for count, _ in enumerate(pool.map(fetch, files), 1):
            if count % 50 == 0 or count == len(files):
                print(f"Dataset files: {count}/{len(files)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", nargs="+", choices=("runtime", "dataset", "llama"))
    for name in parser.parse_args().asset:
        if name == "dataset":
            fetch_dataset()
            continue
        for url, target, digest in ARTIFACTS[name]:
            print(f"Verifying or downloading {target}", flush=True)
            download(url, target, digest)


if __name__ == "__main__":
    main()
