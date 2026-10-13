# tanoa-sld -- 3i SlideBook reader for tanoa. GPL-2.0-or-later; see LICENSE.
"""The parsed index of one SlideBook file: everything `probe` and every plane
read need, computed once by the full `SlidebookFile` scan and then cached on
local disk.

The scan is the expensive part -- it reads up to the end of every pixel block,
which for a 388 GB file on a network share is the whole file. The resulting
index is a few KB to a few MB (one 8-byte offset per plane plus metadata), so
every later open -- probe, preview, each channel of a conversion, in this
process or the next -- loads it from the cache instead.

Cache key: absolute path + size + mtime_ns + `INDEX_VERSION`. A changed file,
or a new version of this reader, simply misses the cache. Entries live in a
local per-user directory, never beside the source (which may be on a
read-only or shared drive): `TANOA_SLD_CACHE_DIR` overrides it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["INDEX_VERSION", "SeriesIndex", "SldIndex", "cache_dir", "load_index"]

#: Bump whenever the parse (or the port it comes from) changes what it returns.
INDEX_VERSION = 1

CACHE_DIR_ENV_VAR = "TANOA_SLD_CACHE_DIR"

_log = logging.getLogger("tanoa_sld")


@dataclass
class SeriesIndex:
    size_x: int
    size_y: int
    size_z: int
    size_c: int
    size_t: int
    dimension_order: str
    little_endian: bool
    bits_per_pixel: int
    #: Byte offset of each plane, in the series' own plane order -- already
    #: including the spool-file metadata skips `openBytes` applies.
    plane_offsets: "list[int]"
    name: "str | None" = None
    description: str = ""
    physical_size_x_um: "float | None" = None
    physical_size_y_um: "float | None" = None
    physical_size_z_um: "float | None" = None
    channel_names: "dict[int, str]" = field(default_factory=dict)
    objective_model: "str | None" = None
    objective_magnification: "float | None" = None

    def plane_index(self, z: int, c: int, t: int) -> int:
        """FormatTools.getIndex for this series' dimension order."""
        sizes = {"Z": self.size_z, "C": self.size_c, "T": self.size_t}
        coord = {"Z": z, "C": c, "T": t}
        a, b, cc = self.dimension_order[2], self.dimension_order[3], self.dimension_order[4]
        return coord[a] + coord[b] * sizes[a] + coord[cc] * sizes[a] * sizes[b]


@dataclass
class SldIndex:
    path: str
    file_size: int
    series: "list[SeriesIndex]"
    scan_seconds: float = 0.0
    from_cache: bool = False

    def to_json(self) -> dict:
        return {"version": INDEX_VERSION, "path": self.path, "file_size": self.file_size,
                "scan_seconds": self.scan_seconds, "series": [asdict(s) for s in self.series]}

    @classmethod
    def from_json(cls, data: dict) -> "SldIndex":
        series = []
        for raw in data["series"]:
            raw = dict(raw)
            raw["channel_names"] = {int(k): v for k, v in raw.get("channel_names", {}).items()}
            series.append(SeriesIndex(**raw))
        return cls(path=data["path"], file_size=data["file_size"], series=series,
                   scan_seconds=data.get("scan_seconds", 0.0), from_cache=True)


def cache_dir() -> Path:
    override = os.environ.get(CACHE_DIR_ENV_VAR)
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "tanoa-sld" / "index"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "tanoa-sld" / "index"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tanoa-sld" / "index"


def _cache_path(path: Path) -> "tuple[Path, int]":
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|v{INDEX_VERSION}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:32]
    return cache_dir() / f"{digest}.json", stat.st_size


def build_index(path: Path, *, progress: Any = None, cancel: Any = None) -> SldIndex:
    """The full scan. Slow on a huge file -- `load_index` caches its result."""
    from ._slidebook import SlidebookFile

    started = time.perf_counter()
    f = SlidebookFile(path, progress=progress, cancel=cancel)
    try:
        series = []
        for s, ms in enumerate(f.core):
            meta = f.series_metadata[s] if s < len(f.series_metadata) else None
            # Resolve every plane's final offset now, in plane order: for a
            # spool (.spl) file `plane_offset` depends on which planes were
            # read before (Java's `metadataInPlanes`), and sequential order is
            # what a sequential reader (bfconvert) produces.
            offsets = [f.plane_offset(s, no) for no in range(ms.imageCount)]
            series.append(SeriesIndex(
                size_x=ms.sizeX, size_y=ms.sizeY, size_z=ms.sizeZ, size_c=ms.sizeC, size_t=ms.sizeT,
                dimension_order=ms.dimensionOrder, little_endian=ms.littleEndian,
                bits_per_pixel=ms.bitsPerPixel, plane_offsets=offsets,
                name=meta.name if meta else None,
                description=meta.description if meta else "",
                physical_size_x_um=meta.physical_size_x_um if meta else None,
                physical_size_y_um=meta.physical_size_y_um if meta else None,
                physical_size_z_um=meta.physical_size_z_um if meta else None,
                channel_names=dict(meta.channel_names) if meta else {},
                objective_model=meta.objective_model if meta else None,
                objective_magnification=meta.objective_magnification if meta else None,
            ))
        size = os.path.getsize(path)
    finally:
        f.close()
    return SldIndex(path=str(path), file_size=size, series=series, scan_seconds=time.perf_counter() - started)


def load_index(
    path: "str | os.PathLike", *, use_cache: bool = True, progress: Any = None, cancel: Any = None,
) -> SldIndex:
    path = Path(path)
    if use_cache:
        try:
            cached, size = _cache_path(path)
            data = json.loads(cached.read_text())
            if data.get("version") == INDEX_VERSION and data.get("file_size") == size:
                return SldIndex.from_json(data)
        except (OSError, ValueError, KeyError, TypeError):
            pass  # a missing or unreadable entry is just a cache miss
    _log.info("Indexing %s (first open; SlideBook files have no index of their own)...", path.name)
    index = build_index(path, progress=progress, cancel=cancel)
    _log.info("Indexed %s in %.1fs: %d series", path.name, index.scan_seconds, len(index.series))
    if use_cache:
        _store(path, index)
    return index


def _store(path: Path, index: SldIndex) -> None:
    try:
        cached, _size = _cache_path(path)
        cached.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: a crash mid-write must never leave a truncated entry that a
        # later open would trust.
        fd, tmp = tempfile.mkstemp(dir=cached.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(index.to_json(), fh)
        os.replace(tmp, cached)
    except OSError as exc:
        _log.warning("Could not cache the SlideBook index (%s); the next open will re-scan.", exc)


def describe(index: SldIndex) -> "list[dict[str, Any]]":
    return [{k: v for k, v in asdict(s).items() if k != "plane_offsets"} for s in index.series]
