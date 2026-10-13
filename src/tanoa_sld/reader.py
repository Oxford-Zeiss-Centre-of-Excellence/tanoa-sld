# tanoa-sld -- 3i SlideBook reader for tanoa. GPL-2.0-or-later; see LICENSE.
"""The tanoa `Reader` for 3i SlideBook `.sld` (and `.spl` spool) files.

Built for files far larger than RAM on slow storage (a 388 GB `.sld` on a
Windows network share is the motivating case):

- **Metadata** comes from a cached index (`_index.py`). The first open scans
  the file once -- unavoidable, SlideBook keeps no index of its own -- using
  large prefetched reads; through tanoa's GUI that runs as `prepare()`, with a
  real progress bar and Cancel. Every later open loads the index in
  milliseconds.
- **Pixels** are never loaded whole. `open_channel` returns a dask array
  chunked into row bands of one plane (about `CHUNK_BYTES` each); each chunk
  is a single positioned read of exactly its own bytes, safe to run from many
  threads at once (more requests in flight is what makes a network share
  fast). tanoa's OME-Zarr writer then streams those chunks block by block
  under its own `max_block_bytes` budget, so peak memory stays bounded by that
  budget, not by the file.
- **Preview** reads one plane strided: a whole small plane in one read, or
  just every k-th row of a huge one.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ._index import SldIndex, load_index
from ._slidebook import is_this_type

if TYPE_CHECKING:
    from tanoa.core.readers.base import SeriesInfo

__all__ = ["CHUNK_BYTES", "SlidebookReader"]

#: Target size of one lazily-read chunk (a band of rows of one plane).
CHUNK_BYTES = 16 * 1024 * 1024
#: A preview reads the whole plane in one request up to this size, and only
#: the rows it needs above it.
_WHOLE_PLANE_PREVIEW_BYTES = 64 * 1024 * 1024


class _PositionalFile:
    """Thread-safe `read_at(offset, n)`: `os.pread` where it exists (POSIX),
    one handle per thread elsewhere (Windows has no pread; a shared handle
    would serialise every read behind a seek)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._local = threading.local()
        self._handles: "list[Any]" = []
        self._lock = threading.Lock()
        self._fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0)) if hasattr(os, "pread") else None
        weakref.finalize(self, _PositionalFile._close_all, self._fd, self._handles)

    @staticmethod
    def _close_all(fd: "int | None", handles: "list[Any]") -> None:
        if fd is not None:
            os.close(fd)
        for fh in handles:
            fh.close()

    def read_at(self, offset: int, n: int) -> bytes:
        if self._fd is not None:
            chunks, done = [], 0
            while done < n:
                part = os.pread(self._fd, n - done, offset + done)
                if not part:
                    break
                chunks.append(part)
                done += len(part)
            data = b"".join(chunks)
        else:
            fh = getattr(self._local, "fh", None)
            if fh is None:
                fh = open(self.path, "rb", buffering=0)  # noqa: SIM115 -- closed by the finaliser
                self._local.fh = fh
                with self._lock:
                    self._handles.append(fh)
            fh.seek(offset)
            data = fh.read(n)
        if len(data) != n:
            raise OSError(f"{self.path}: wanted {n} bytes at offset {offset}, file ended after {len(data)}")
        return data


class _ChannelVolume:
    """Array-like (T, Z, Y, X) view of one channel for `dask.array.from_array`.
    Reads only the rows a slice asks for -- one positioned read per plane
    touched."""

    def __init__(self, source: _PositionalFile, series, channel: int) -> None:
        self._src = source
        self._s = series
        self._c = channel
        self.shape = (series.size_t, series.size_z, series.size_y, series.size_x)
        self.dtype = np.dtype("uint16")
        self.ndim = 4
        self._file_dtype = np.dtype("<u2" if series.little_endian else ">u2")

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        ts, zs, ys, xs = (_as_range(k, n) for k, n in zip(key, self.shape))
        out = np.empty((len(ts), len(zs), len(ys), len(xs)), dtype=self.dtype)
        if len(ys) and len(xs):
            for i, t in enumerate(ts):
                for j, z in enumerate(zs):
                    out[i, j] = self.read_rows(t, z, ys)[:, xs.start:xs.stop:xs.step]
        squeeze = tuple(axis for axis, k in enumerate(key) if isinstance(k, (int, np.integer)))
        return out.squeeze(axis=squeeze) if squeeze else out

    def read_rows(self, t: int, z: int, rows: range) -> np.ndarray:
        """Rows `rows` of plane (t, z) as a (len(rows), X) native-endian array:
        one contiguous read spanning first to last row, strided afterward."""
        s = self._s
        row_bytes = s.size_x * 2
        first, last = rows[0], rows[-1]
        offset = s.plane_offsets[s.plane_index(z, self._c, t)] + first * row_bytes
        raw = self._src.read_at(offset, (last - first + 1) * row_bytes)
        band = np.frombuffer(raw, dtype=self._file_dtype).reshape(last - first + 1, s.size_x)
        return band[:: rows.step if len(rows) > 1 else 1].astype(self.dtype, copy=False)


def _as_range(k, n: int) -> range:
    if isinstance(k, slice):
        return range(*k.indices(n))
    if isinstance(k, (int, np.integer)):
        k = int(k) + n if k < 0 else int(k)
        if not 0 <= k < n:
            raise IndexError(f"index {k} out of range for axis of size {n}")
        return range(k, k + 1)
    raise TypeError(f"unsupported index {k!r}")


class SlidebookReader:
    """3i SlideBook 4-6 `.sld` / `.spl`. Pixel type is always uint16 (as in
    Bio-Formats); byte order comes from the file."""

    name = "tanoa-sld"
    extensions = (".sld", ".spl")
    self_describing = True

    def __init__(self) -> None:
        self._indexes: "dict[tuple, SldIndex]" = {}
        self._files: "dict[str, _PositionalFile]" = {}
        self._lock = threading.Lock()

    # -- Reader protocol ---------------------------------------------------

    def can_read(self, path: Path) -> bool:
        path = Path(path)
        if path.suffix.lower() not in self.extensions:
            return False
        try:
            return is_this_type(path)
        except OSError:  # missing/unreadable: simply not ours to claim
            return False

    def prepare(self, path: Path, *, progress: "Callable[[float], None] | None" = None, cancel: Any = None) -> None:
        """tanoa's `PreparableReader`: the one-time index scan, with progress
        and cancel, run by the GUI before anything else touches the file."""
        self._index(Path(path), progress=progress, cancel=cancel)

    def probe(self, path: Path) -> "list[SeriesInfo]":
        from tanoa.core.readers.base import SeriesInfo

        index = self._index(Path(path))
        infos = []
        for i, s in enumerate(index.series):
            spacing = None
            if s.physical_size_x_um:
                spacing = (
                    s.physical_size_z_um or 1.0,
                    s.physical_size_y_um or s.physical_size_x_um,
                    s.physical_size_x_um,
                )
            infos.append(SeriesInfo(
                index=i,
                name=s.name or (Path(path).stem if len(index.series) == 1 else f"{Path(path).stem} #{i}"),
                shape_tczyx=(s.size_t, s.size_c, s.size_z, s.size_y, s.size_x),
                dtype="uint16",
                spacing_um_zyx=spacing,
                channel_names=[s.channel_names.get(c) or f"channel{c}" for c in range(s.size_c)],
                raw_metadata={
                    "significant_bits": s.bits_per_pixel or None,
                    "description": s.description,
                    "objective": s.objective_model,
                    "objective_magnification": s.objective_magnification,
                    "dimension_order": s.dimension_order,
                    "index_from_cache": index.from_cache,
                    "index_scan_seconds": index.scan_seconds,
                },
            ))
        return infos

    def open_channel(
        self, path: Path, series_index: int = 0, channel_index: int = 0,
        progress: "Callable[[float], None] | None" = None,
    ) -> "tuple[Any, dict[str, Any]]":
        import dask.array as da

        path = Path(path)
        s = self._series(path, series_index)
        if not 0 <= channel_index < s.size_c:
            raise IndexError(f"{path}: channel_index={channel_index} out of range for {s.size_c} channel(s).")
        volume = _ChannelVolume(self._file(path), s, channel_index)
        rows = max(1, min(s.size_y, CHUNK_BYTES // max(1, s.size_x * 2)))
        lazy = da.from_array(volume, chunks=(1, 1, rows, s.size_x), asarray=False, lock=False,
                             name=f"tanoa-sld-{abs(hash((str(path), series_index, channel_index)))}")
        if s.size_t == 1:
            lazy = lazy[0]
        if progress is not None:
            progress(1.0)
        return lazy, {"path": str(path), "series_index": series_index,
                      "channel_index": channel_index, "backend": self.name}

    # -- PreviewReader protocol --------------------------------------------

    def read_preview(
        self, path: Path, series_index: int, channel_index: int, *,
        t: int = 0, z: int = 0, max_px: int = 800,
    ) -> np.ndarray:
        path = Path(path)
        s = self._series(path, series_index)
        step = max(1, math.ceil(max(s.size_x, s.size_y) / max_px))
        volume = _ChannelVolume(self._file(path), s, channel_index)
        plane_bytes = s.size_x * s.size_y * 2
        if plane_bytes <= _WHOLE_PLANE_PREVIEW_BYTES or step == 1:
            plane = volume.read_rows(t, z, range(0, s.size_y))
            return np.ascontiguousarray(plane[::step, ::step])
        # A huge plane: one small read per kept row, never the whole plane.
        rows = range(0, s.size_y, step)
        out = np.empty((len(rows), len(range(0, s.size_x, step))), dtype=np.uint16)
        for k, y in enumerate(rows):
            out[k] = volume.read_rows(t, z, range(y, y + 1))[0, ::step]
        return out

    # -- internals -----------------------------------------------------------

    def _index(self, path: Path, *, progress: Any = None, cancel: Any = None) -> SldIndex:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        with self._lock:
            cached = self._indexes.get(key)
        if cached is not None:
            return cached
        index = load_index(path, progress=progress, cancel=cancel)
        with self._lock:
            self._indexes[key] = index
        return index

    def _series(self, path: Path, series_index: int):
        index = self._index(path)
        if not 0 <= series_index < len(index.series):
            raise IndexError(f"{path}: no series {series_index} (file has {len(index.series)}).")
        return index.series[series_index]

    def _file(self, path: Path) -> _PositionalFile:
        key = str(path.resolve())
        with self._lock:
            handle = self._files.get(key)
            if handle is None:
                handle = self._files[key] = _PositionalFile(key)
        return handle


if sys.version_info < (3, 10):  # pragma: no cover
    raise ImportError("tanoa-sld needs Python 3.10+")
