from __future__ import annotations

import dataclasses
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

TRANSFORMER_NFN_DATASETS = {
    "mnist": "https://huggingface.co/datasets/anonymized-acamedia/Small-Transformer-Zoo/resolve/main/MNIST-Transformers.zip",
    "ag_news": "https://huggingface.co/datasets/anonymized-acamedia/Small-Transformer-Zoo/resolve/main/AG-News-Transformers.zip",
}


@dataclasses.dataclass(frozen=True, slots=True)
class DownloadResult:
    archive: Path
    extracted_directory: Path
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_transformer_nfn_zoo(
    name: str,
    destination: str | Path,
    *,
    expected_sha256: str | None = None,
    overwrite: bool = False,
) -> DownloadResult:
    """Download the public Transformer-NFN model zoo with checksum support.

    Supply an expected checksum in production. The upstream project currently publishes direct
    archive URLs but not a checksum in its README, so the first reviewed download should be pinned
    into your experiment lockfile before distributed jobs begin.
    """

    if name not in TRANSFORMER_NFN_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}; choose from {sorted(TRANSFORMER_NFN_DATASETS)}"
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"transformer-nfn-{name}.zip"
    extracted = destination / f"transformer-nfn-{name}"
    if overwrite:
        archive.unlink(missing_ok=True)
        shutil.rmtree(extracted, ignore_errors=True)
    if not archive.exists():
        with (
            urllib.request.urlopen(TRANSFORMER_NFN_DATASETS[name]) as response,
            archive.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
    digest = _sha256(archive)
    if expected_sha256 and digest != expected_sha256:
        archive.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {archive}: expected {expected_sha256}, got {digest}"
        )
    if not extracted.exists():
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                resolved = (extracted / member.filename).resolve()
                if extracted.resolve() not in resolved.parents and resolved != extracted.resolve():
                    raise ValueError(f"Unsafe path in archive: {member.filename}")
            handle.extractall(extracted)
    return DownloadResult(archive=archive, extracted_directory=extracted, sha256=digest)
