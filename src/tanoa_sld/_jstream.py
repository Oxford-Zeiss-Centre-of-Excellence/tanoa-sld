# tanoa-sld -- 3i SlideBook reader for tanoa.
# Copyright (C) 2026 tanoa-sld contributors.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 2 of the License, or (at your option)
# any later version. See LICENSE.
"""Bug-compatible emulation of Bio-Formats' `loci.common.RandomAccessInputStream`
(ome-common-java, backed by `NIOFileHandle`) -- only the methods
`SlidebookReader` calls.

The port in `_slidebook.py` is line-by-line, and SlidebookReader's heuristics
depend on these exact semantics rather than on what the method names suggest.
Each one was checked against the Java source:

- `read()` returns a **signed** byte (`readByte()`), so `== -1` means the byte
  0xFF -- that is how the reader recognises a `0xFFFF` variable-length metadata
  block -- and raises EOFError past the end, never returns -1 there.
- `read(buf)` / `read(buf, off, n)` return how many bytes were read, which may
  be fewer than asked near the end; -1 only once nothing at all remains.
- `skipBytes(n)` moves forward by `min(n, length)`, and does nothing for n < 1.
- `readString(n)` clamps `n` to what is left, then decodes as UTF-8. Decoding
  binary data as UTF-8 can make the *character* count differ from the *byte*
  count; SlidebookReader then uses `String.indexOf` positions as byte offsets.
  That quirk is reproduced on purpose (errors="replace" follows the same
  maximal-subpart substitution Java's decoder uses).
- `findString(True, blockSize, "\\0")` returns the text up to and including the
  terminator and leaves the pointer just past it -- computed in characters,
  as Java's does.
- `readShort/readInt/readFloat` honour the current byte order (`order()`).
"""

from __future__ import annotations

import os
import struct

__all__ = ["JavaEOFError", "RandomAccessInputStream"]


class JavaEOFError(EOFError):
    """java.io.EOFException."""


class RandomAccessInputStream:
    def __init__(self, path: "str | os.PathLike") -> None:
        self._fh = open(path, "rb")  # noqa: SIM115 -- lifetime managed by close()
        self._length = os.fstat(self._fh.fileno()).st_size
        self._pos = 0
        self._little = False  # Java's default: big-endian

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        self._fh.close()

    # -- position -------------------------------------------------------
    def length(self) -> int:
        return self._length

    def getFilePointer(self) -> int:
        return self._pos

    def seek(self, pos: int) -> None:
        self._pos = int(pos)

    def order(self, little: bool) -> None:
        self._little = bool(little)

    def isLittleEndian(self) -> bool:
        return self._little

    def skipBytes(self, n: int) -> int:
        if n < 1:
            return 0
        old = self._pos
        self._pos = old + min(int(n), self._length)
        return self._pos - old

    # -- raw reads ------------------------------------------------------
    def _read_exact(self, n: int) -> bytes:
        if self._pos < 0 or self._pos + n > self._length:
            raise JavaEOFError("Attempting to read beyond end of file.")
        self._fh.seek(self._pos)
        data = self._fh.read(n)
        if len(data) != n:
            raise JavaEOFError("Attempting to read beyond end of file.")
        self._pos += n
        return data

    def read(self, buf: "bytearray | None" = None, off: int = 0, n: "int | None" = None) -> int:
        """`read()` with no buffer: one signed byte. With a buffer: Java's
        `read(byte[])` / `read(byte[], off, n)` -- fills in place, returns the
        count read (-1 when nothing remains)."""
        if buf is None:
            return struct.unpack("b", self._read_exact(1))[0]
        if n is None:
            n = len(buf) - off
        real = min(n, self._length - self._pos)
        if real < 0:
            return -1
        self._fh.seek(self._pos)
        data = self._fh.read(real)
        buf[off:off + len(data)] = data
        self._pos += len(data)
        rtn = len(data)
        # RandomAccessInputStream.read(byte[]): 0 bytes at the very end is -1.
        if rtn == 0 and self._pos >= self._length - 1:
            return -1
        return rtn

    def _fmt(self, code: str) -> str:
        return ("<" if self._little else ">") + code

    def readShort(self) -> int:
        return struct.unpack(self._fmt("h"), self._read_exact(2))[0]

    def readInt(self) -> int:
        return struct.unpack(self._fmt("i"), self._read_exact(4))[0]

    def readFloat(self) -> float:
        # Kept as a Python float of the exact float32 value; callers that
        # compare against Java doubles widen it the same way Java does.
        return struct.unpack(self._fmt("f"), self._read_exact(4))[0]

    def readBytes(self, n: int) -> bytes:
        """Not Java API -- used by the pixel reader for whole-plane reads."""
        return self._read_exact(n)

    # -- strings --------------------------------------------------------
    def available(self) -> int:
        return max(0, min(self._length - self._pos, 2**31 - 1))

    def readString(self, n: int) -> str:
        avail = self.available()
        if n > avail:
            n = avail
        if n <= 0:
            return ""
        return self._read_exact(n).decode("utf-8", errors="replace")

    def findString(self, save_string: bool, block_size: int, terminator: str) -> str:
        """`findString(true, blockSize, terminator)` for one terminator --
        the only form SlidebookReader uses (`readCString`)."""
        start = self._pos
        max_len = self._length - start
        out = ""
        consumed_bytes = 0
        while consumed_bytes < max_len and self._pos < self._length - 1:
            chunk = min(block_size, self._length - self._pos)
            self._fh.seek(self._pos)
            raw = self._fh.read(chunk)
            if not raw:
                raise OSError("Cannot read from stream: 0")
            self._pos += len(raw)
            consumed_bytes += len(raw)
            # InputStreamReader decodes the bytes it has pulled so far as one
            # stream; decoding the cumulative byte string reproduces that.
            out = self._fh_bytes_since(start).decode("utf-8", errors="replace")
            index = out.find(terminator)
            if index >= 0:
                # Java seeks by *character* index (startPos + min + tagLen).
                self._pos = start + index + len(terminator)
                return out[: index + len(terminator)]
        return out

    def _fh_bytes_since(self, start: int) -> bytes:
        self._fh.seek(start)
        return self._fh.read(self._pos - start)
