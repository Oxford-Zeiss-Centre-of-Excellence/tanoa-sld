# tanoa-sld -- 3i SlideBook reader for tanoa. GPL-2.0-or-later; see LICENSE.
"""Finding where a SlideBook pixel block ends -- the one step that has to read
(nearly) the whole file, since the format has no index.

`SlidebookReader.java` (lines 336-374) reads the block in 8 KB windows and
scans each byte-by-byte for the next block's marker. Over a network share
that is one round trip per 8 KB: hours for a multi-hundred-GB file. This
module gets the *identical* answer from large sequential reads (default
32 MiB), with the next chunk prefetched on a background thread while the
current one is searched, and the marker found by one C-level regex pass.

"Identical" includes the Java loop's quirks, derived from its arithmetic:

- Window 0 holds the 8192 bytes from `fp`; window k>=1 holds 20 bytes carried
  over from the previous window followed by 8172 fresh ones, so its data
  starts at P_k = fp + 8192 + (k-1)*8172 and it covers [P_k-20, P_k+8172).
- Only start positions i < n-6 are tested, so window 0 tests
  [fp, fp+8186) and window k tests [P_k-20, P_k+8146). That leaves a 6-byte
  blind spot [P_k+8146, P_k+8152) between consecutive windows: a marker
  starting there is never seen.
- A window is only searched while the file pointer, *after* reading it, is
  still before EOF -- so the window that reaches EOF is never searched.
- The pointer left behind is the marker's position minus 20 for window 0
  (`getFilePointer() - n + i - 20` with no carry-over), and exactly the
  marker's position for every later window.

`scan_reference` is the Java loop transcribed literally; tests run both on
the same bytes, including markers placed in the blind spots.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import BinaryIO, Callable

__all__ = ["ScanResult", "scan_fast", "scan_reference"]

_W0 = 8192  # first window
_WK = 8172  # fresh bytes per later window (8192 minus 20 carried over)
_CARRY = 20
_OVERLAP = 5  # a candidate spans 6 bytes

# Group 1: the 'h'/'i' family (Java 346-347); group 2: 'j'/'k'/'n'/"on" (356-359).
_CANDIDATE = re.compile(rb"(?=(?:([hi]\x00|\x00[hi])|([jkn]\x00|\x00[jkn]|on))..(?:II|MM))", re.S)


@dataclass(frozen=True)
class ScanResult:
    found: bool
    #: True for an 'h'/'i' marker, False for 'j'/'k'/'n'/"on" (only if found)
    hi_family: bool
    #: For the 'h'/'i' family: whether either marker byte is 'i' -- Java then
    #: drops the pixel block it just recorded (line 351-353).
    is_i: bool
    #: The file pointer the Java loop leaves behind.
    pointer: int


def _scanned(a: int, fp: int, length: int) -> bool:
    """Whether the Java loop would ever test a candidate starting at `a`."""
    if a < fp:
        return False
    if a < fp + _W0 - 6:
        return fp + _W0 < length  # window 0, searched only if it ends before EOF
    k = (a - (fp + _W0 - _CARRY)) // _WK + 1  # the only window k>=1 that could hold `a`
    if k < 1:
        return False
    p_k = fp + _W0 + (k - 1) * _WK
    if not (p_k - _CARRY <= a < p_k + _WK - 26):  # outside its tested range: a blind spot
        return False
    return p_k + _WK < length


def _pointer_for(a: int, fp: int) -> int:
    return a - 20 if a < fp + _W0 - 6 else a


class _Prefetcher:
    """Sequential large reads with one chunk always in flight ahead."""

    def __init__(self, fh: BinaryIO, start: int, chunk: int) -> None:
        self._fh, self._next, self._chunk = fh, start, chunk
        self._lock = threading.Lock()
        self._pending: "tuple[int, threading.Thread, list] | None" = None
        self._prefetch()

    def _read_at(self, offset: int, out: list) -> None:
        with self._lock:
            self._fh.seek(offset)
            out.append(self._fh.read(self._chunk + _OVERLAP))

    def _prefetch(self) -> None:
        out: list = []
        thread = threading.Thread(target=self._read_at, args=(self._next, out), daemon=True)
        thread.start()
        self._pending = (self._next, thread, out)
        self._next += self._chunk

    def next(self) -> "tuple[int, bytes]":
        offset, thread, out = self._pending
        thread.join()
        data = out[0] if out else b""
        if data:
            self._prefetch()
        return offset, data


def scan_fast(
    fh: BinaryIO, fp: int, length: int, *, chunk_bytes: int = 32 * 1024 * 1024,
    on_chunk: "Callable[[int], None] | None" = None,
) -> ScanResult:
    """Same answer as `scan_reference`, from large prefetched reads.
    `on_chunk(file_position)` runs after each chunk -- progress reporting, and
    the place a cancel request raises."""
    reader = _Prefetcher(fh, fp, chunk_bytes)
    while True:
        base, data = reader.next()
        if not data:
            break
        if on_chunk is not None:
            on_chunk(base + len(data))
        for m in _CANDIDATE.finditer(data):
            a = base + m.start()
            if a >= base + chunk_bytes:
                break  # belongs to the next chunk (the overlap); seen there
            if not _scanned(a, fp, length):
                continue
            hi = m.group(1) is not None
            pair = m.group(1) or m.group(2)
            return ScanResult(True, hi, hi and b"i" in pair, _pointer_for(a, fp))
        if len(data) <= chunk_bytes:
            break
    # Not found: Java reads on until a window reaches EOF and stops there.
    return ScanResult(False, False, False, length)


def scan_reference(fh: BinaryIO, fp: int, length: int) -> ScanResult:
    """SlidebookReader.java 336-374, transcribed literally (8 KB windows)."""
    def read_into(buf: bytearray, off: int, n: int, pos: int) -> "tuple[int, int]":
        real = min(n, length - pos)
        if real < 0:
            return -1, pos
        fh.seek(pos)
        data = fh.read(real)
        buf[off:off + len(data)] = data
        pos += len(data)
        rtn = len(data)
        if rtn == 0 and pos >= length - 1:
            rtn = -1
        return rtn, pos

    buf = bytearray(_W0)
    n, pos = read_into(buf, 0, _W0, fp)
    while pos < length:
        for i in range(0, n - 6):
            b0, b1, b4, b5 = buf[i], buf[i + 1], buf[i + 4], buf[i + 5]
            if not ((b4 == 0x49 and b5 == 0x49) or (b4 == 0x4D and b5 == 0x4D)):
                continue
            if (b0 in (0x68, 0x69) and b1 == 0) or (b0 == 0 and b1 in (0x68, 0x69)):
                return ScanResult(True, True, b0 == 0x69 or b1 == 0x69, pos - n + i - 20)
            if (b0 in (0x6A, 0x6B, 0x6E) and b1 == 0) or (b0 == 0 and b1 in (0x6A, 0x6B, 0x6E)) or (b0 == 0x6F and b1 == 0x6E):
                return ScanResult(True, False, False, pos - n + i - 20)
        tmp = buf
        buf = bytearray(_W0)
        buf[0:_CARRY] = tmp[len(tmp) - _CARRY:]
        n, pos = read_into(buf, _CARRY, _W0 - _CARRY, pos)
    return ScanResult(False, False, False, pos)


def open_for_scan(path: "str | os.PathLike") -> BinaryIO:
    # Unbuffered: every read here is already large, and Python's own 8 KB
    # buffer would only add a copy.
    return open(path, "rb", buffering=0)  # noqa: SIM115


def scan_predicted(
    fh: BinaryIO, fp: int, length: int, guesses: "list[int]", *, window: int = 64 * 1024,
) -> "ScanResult | None":
    """Look for the block's end marker only where it is expected, without
    reading the pixel data before it: a small window at each guessed block
    length (smallest first). Returns None when no guess is confirmed, and the
    caller then falls back to the exact `scan_fast`.

    Agrees with the Java scan whenever the pixel data itself contains no
    marker-like byte pattern before the real end -- the Java would stop at
    such a pattern, this does not look there. For data up to 14 bits that
    pattern cannot occur (it needs 0x49/0x4D in a high byte, i.e. values
    >= 0x4900); `TANOA_SLD_EXACT_INDEX=1` forces the exact scan regardless.
    """
    for guess in sorted({g for g in guesses if g >= 1024}):
        start = fp + guess - 64
        if start + 6 > length:
            continue
        fh.seek(start)
        data = fh.read(window)
        for m in _CANDIDATE.finditer(data):
            a = start + m.start()
            if not _scanned(a, fp, length):
                continue
            hi = m.group(1) is not None
            pair = m.group(1) or m.group(2)
            return ScanResult(True, hi, hi and b"i" in pair, _pointer_for(a, fp))
    return None
