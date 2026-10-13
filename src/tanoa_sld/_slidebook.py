# tanoa-sld -- 3i SlideBook reader for tanoa.
#
# Line-by-line Python translation of Bio-Formats'
# components/formats-gpl/src/loci/formats/in/SlidebookReader.java
# (ome/bioformats, develop branch, 1374 lines), which is:
#
#   Copyright (C) 2005 - 2017 Open Microscopy Environment:
#     - Board of Regents of the University of Wisconsin-Madison
#     - Glencoe Software, Inc.
#     - University of Dundee
#
# Translation copyright (C) 2026 tanoa-sld contributors.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 2 of the License, or (at your option)
# any later version. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSE.
"""Faithful port of Bio-Formats' SlidebookReader.

Structure, control flow and every magic number follow the Java line for line
(Java line numbers are cited in comments so the two can be diffed by eye).
Where Java semantics differ from Python's they are reproduced explicitly:

- integer `/` and `%` truncate toward zero          -> `_div`, `_mod`
- `int` arithmetic wraps at 32 bits                  -> `_i32`, gated by
  `java_int_overflow` (see below)
- `float` arithmetic rounds to float32               -> `_f32`
- `String.trim()` strips every char <= U+0020        -> `_jtrim`
- byte reads are signed, strings are UTF-8 decoded   -> `_jstream`

**Two deliberate, behaviour-preserving speed-ups** -- the Python port would
otherwise be unusable on a large file, where Java gets away with tight loops:

1. The pixel-block end search (Java 336-374) -- the step that reads almost
   the whole file -- uses large prefetched reads and one regex pass, with
   Java's 8 KB window arithmetic (blind spots included) reproduced exactly;
   see `_scan.py`. The small 0xFFFF-block search (Java 256-277) keeps its
   8 KB windows, matched by `re.search` over the same buffer.
2. The 128-byte metadata-block walk (Java 490-493) skips straight past a
   pixel block instead of calling `withinPixels` once per 128 bytes of it
   (billions of calls for a multi-hundred-GB file). The same `q` values are
   processed either way.

**`java_int_overflow`.** Several Java expressions multiply two `int`s and
silently wrap at 2^31 before widening to `long`. The one that matters is
Java 1204, `offset + p * planeSize`: for any series larger than 2 GiB
(e.g. 2048x2048x16-bit planes from plane 256 on) the computed plane offset
wraps and Bio-Formats reads the wrong bytes. With `java_int_overflow=True`
this module reproduces that exactly (used to diff against real Bio-Formats);
the default, False, computes every offset in 64 bits.
"""

from __future__ import annotations

import re
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._jstream import JavaEOFError, RandomAccessInputStream
from ._scan import open_for_scan, scan_fast, scan_reference

__all__ = ["CoreMetadata", "ScanCancelled", "SlidebookFile", "is_this_type"]


class ScanCancelled(RuntimeError):
    """The scan was stopped by its `cancel` event."""

# -- Constants (Java 64-70) --------------------------------------------------

SLD_MAGIC_BYTES_1_0 = 0x006C
SLD_MAGIC_BYTES_1_1 = 0x0100
SLD_MAGIC_BYTES_1_2 = 0x0200
SLD_MAGIC_BYTES_2_0 = 0x01F5
SLD_MAGIC_BYTES_2_1 = 0x0102
SLD_MAGIC_BYTES_3 = 0xF6010101

_EPSILON = 0.000001  # loci.common.Constants.EPSILON
_INT_MAX = 2**31 - 1
_JAVA_WS = "".join(chr(c) for c in range(0x21))


# -- Java arithmetic --------------------------------------------------------


def _div(a: int, b: int) -> int:
    """Java integer division: truncates toward zero; ArithmeticException on /0."""
    if b == 0:
        raise ZeroDivisionError("/ by zero")
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _mod(a: int, b: int) -> int:
    """Java `%`: the result takes the sign of the dividend."""
    return a - _div(a, b) * b


def _wrap32(v: int) -> int:
    return ((v + 2**31) % 2**32) - 2**31


def _f32(v: float) -> float:
    return struct.unpack("f", struct.pack("f", v))[0]


def _jtrim(s: str) -> str:
    return s.strip(_JAVA_WS)


@dataclass
class CoreMetadata:
    """The subset of loci.formats.CoreMetadata this reader sets."""

    sizeX: int = 0
    sizeY: int = 0
    sizeZ: int = 0
    sizeC: int = 0
    sizeT: int = 0
    imageCount: int = 0
    littleEndian: bool = False
    bitsPerPixel: int = 0
    dimensionOrder: str = "XYZCT"
    pixelType: str = "uint16"


@dataclass
class SeriesMetadata:
    """What `MetadataTools.populatePixels` + the store calls (Java 1225-1344)
    would have put into OME, kept as plain values."""

    name: "str | None" = None
    description: str = ""
    physical_size_x_um: "float | None" = None
    physical_size_y_um: "float | None" = None
    physical_size_z_um: "float | None" = None
    channel_names: "dict[int, str]" = field(default_factory=dict)
    channel_nd_filters: "dict[int, float]" = field(default_factory=dict)
    plane_exposure_times: "dict[int, float]" = field(default_factory=dict)
    objective_model: "str | None" = None
    objective_magnification: "float | None" = None
    original: "dict[str, Any]" = field(default_factory=dict)


def _physical_size(v: "float | None") -> "float | None":
    # FormatTools.getPhysicalSize{X,Y,Z}: only a positive, finite value.
    if v is None or not (v - _EPSILON > 0) or v == float("inf"):
        return None
    return v


# -- isThisType (Java 98-113) ----------------------------------------------


def is_this_type(path: "str | Path") -> bool:
    stream = RandomAccessInputStream(path)
    try:
        block_len = 8
        stream.seek(4)
        little_endian = stream.readString(2) == "II"
        # FormatTools.validStream: seek(0), order(little), length >= blockLen
        stream.seek(0)
        stream.order(little_endian)
        if not stream.length() >= block_len:
            return False
        magic1 = stream.readShort()
        magic2 = stream.readShort()
        return ((magic2 & 0xFF00) == SLD_MAGIC_BYTES_1_1 or (magic2 & 0xFF00) == SLD_MAGIC_BYTES_1_2) and (
            magic1 == SLD_MAGIC_BYTES_1_0 or magic1 == SLD_MAGIC_BYTES_2_0
        )
    except JavaEOFError:
        return False
    finally:
        stream.close()


# The two marker searches, as regexes whose earliest match is the same index
# the Java `for (i ...)` loops stop at. Lookaheads so overlapping candidates
# are all considered.
_META_MARKER = re.compile(rb"(?=MM|II)")  # Java 260-261


class SlidebookFile:
    """One open SlideBook file: `__init__` is Java's `initFile`, `open_bytes`
    is `openBytes`. Not thread-safe by itself (the Java reader is not either);
    `open_bytes` takes a lock because the spool branch mutates state."""

    def __init__(
        self, path: "str | Path", *, java_int_overflow: bool = False, reference_scan: bool = False,
        progress: "Any | None" = None, cancel: "Any | None" = None,
    ) -> None:
        """`progress(fraction)` is called (throttled) during the scan;
        `cancel` is anything with `is_set()` (a threading.Event) -- once set,
        the scan stops at its next checkpoint with `ScanCancelled`."""
        self.path = str(path)
        self._progress = progress
        self._cancel = cancel
        self._last_report = 0.0
        self._overflow = java_int_overflow
        self._reference_scan = reference_scan
        self._lock = threading.Lock()
        self.series_metadata: "list[SeriesMetadata]" = []
        self._scan_fh = open_for_scan(self.path)
        try:
            self._init_file(self.path)
        finally:
            self._scan_fh.close()

    # -- progress / cancel (not in the Java) ---------------------------------

    def _checkpoint(self, position: int) -> None:
        if self._cancel is not None and self._cancel.is_set():
            raise ScanCancelled(f"indexing {Path(self.path).name} cancelled")
        if self._progress is not None:
            now = time.monotonic()
            if now - self._last_report >= 0.25:
                self._last_report = now
                length = self.input.length() or 1
                self._progress(min(1.0, position / length))

    # -- int helpers that honour the overflow switch ------------------------

    def _i32(self, v: int) -> int:
        return _wrap32(v) if self._overflow else v

    # -- FormatReader plumbing ----------------------------------------------

    def getSeriesCount(self) -> int:
        return len(self.core)

    def setSeries(self, i: int) -> None:
        self._series = i

    def getSeries(self) -> int:
        return self._series

    def _ms(self) -> CoreMetadata:
        return self.core[self._series]

    def getSizeX(self) -> int:
        return self._ms().sizeX

    def getSizeY(self) -> int:
        return self._ms().sizeY

    def getSizeZ(self) -> int:
        return self._ms().sizeZ

    def getSizeC(self) -> int:
        return self._ms().sizeC

    def getSizeT(self) -> int:
        return self._ms().sizeT

    def getImageCount(self) -> int:
        return self._ms().imageCount

    def isLittleEndian(self) -> bool:
        return self._ms().littleEndian

    def getZCTCoords(self, index: int) -> "list[int]":
        """FormatTools.getZCTCoords for the current series."""
        ms = self._ms()
        order = ms.dimensionOrder
        sizes = {"Z": ms.sizeZ, "C": ms.sizeC, "T": ms.sizeT}
        for axis, size in sizes.items():
            if size <= 0:
                raise ValueError(f"Invalid {axis} size: {size}")
        num = ms.imageCount
        if num <= 0:
            raise ValueError(f"Invalid plane count: {num}")
        if num != ms.sizeZ * ms.sizeC * ms.sizeT:
            raise ValueError("ZCT size vs image count mismatch")
        if index < 0 or index >= num:
            raise ValueError(f"Invalid image index: {index}/{num}")
        len0, len1 = sizes[order[2]], sizes[order[3]]
        v0 = index % len0
        v1 = (index // len0) % len1
        v2 = index // len0 // len1
        coord = {order[2]: v0, order[3]: v1, order[4]: v2}
        return [coord["Z"], coord["C"], coord["T"]]

    def getIndex(self, z: int, c: int, t: int) -> int:
        ms = self._ms()
        order = ms.dimensionOrder
        sizes = {"Z": ms.sizeZ, "C": ms.sizeC, "T": ms.sizeT}
        coord = {"Z": z, "C": c, "T": t}
        a, b, cc = order[2], order[3], order[4]
        return coord[a] + coord[b] * sizes[a] + coord[cc] * sizes[a] * sizes[b]

    def _meta(self, series: int) -> SeriesMetadata:
        while len(self.series_metadata) <= series:
            self.series_metadata.append(SeriesMetadata())
        return self.series_metadata[series]

    def _readCString(self) -> str:
        return self.input.findString(True, 256, "\0")

    def _withinPixels(self, offset: int) -> bool:  # Java 1349-1358
        for i in range(len(self.pixelOffsets)):
            pixel_offset = self.pixelOffsets[i]
            pixel_length = self.pixelLengths[i]
            if offset >= pixel_offset and offset < pixel_offset + pixel_length:
                return True
        return False

    def _pixel_block_end_containing(self, offset: int) -> "int | None":
        for pixel_offset, pixel_length in zip(self.pixelOffsets, self.pixelLengths):
            if pixel_offset <= offset < pixel_offset + pixel_length:
                return pixel_offset + pixel_length
        return None

    # -- openBytes (Java 115-173) -------------------------------------------

    def plane_offset(self, series: int, no: int) -> int:
        """Where plane `no` of `series` starts, after the spool-file metadata
        adjustment `openBytes` applies (Java 124-169)."""
        with self._lock:
            self.setSeries(series)
            inp = self.input
            offset = self.planeOffset[series][no]
            inp.seek(offset)
            if self.isSpool:
                length = self.pixelLengths[self.getSeries()]
                plane_size = self.getSizeX() * self.getSizeY() * 2  # FormatTools.getPlaneSize, UINT16
                diff = length - self.getImageCount() * plane_size
                inp.seek(inp.getFilePointer() - diff)
                for key in sorted(self.metadataInPlanes):
                    if key < no:
                        inp.skipBytes(256)
                beginning = inp.getFilePointer()
                for i in range(8):
                    if inp.getFilePointer() + 4 >= inp.length():
                        inp.seek(beginning)
                        break
                    inp.order(False)
                    magic = inp.readInt() & 0xFFFFFFFF
                    inp.order(self.isLittleEndian())
                    if magic == SLD_MAGIC_BYTES_3 and no not in self.metadataInPlanes.values():
                        self.metadataInPlanes[no] = 0
                        inp.skipBytes(252)
                        break
                    elif i == 7:
                        inp.seek(beginning)
                    else:
                        inp.skipBytes(252)
            return inp.getFilePointer()

    def open_bytes(self, series: int, no: int) -> bytes:
        """One full plane, raw (sizeX * sizeY * 2 bytes, file byte order)."""
        start = self.plane_offset(series, no)
        with self._lock:
            ms = self.core[series]
            self.input.seek(start)
            return self.input.readBytes(ms.sizeX * ms.sizeY * 2)

    def close(self) -> None:
        self.input.close()

    # -- initFile (Java 192-1345) -------------------------------------------

    def _init_file(self, id_: str) -> None:  # noqa: C901, PLR0912, PLR0915 -- a straight port
        self.core = [CoreMetadata()]
        self._series = 0
        self.input = inp = RandomAccessInputStream(id_)
        self.isSpool = id_.lower().endswith(".spl")
        self.metadataInPlanes: "dict[int, int]" = {}
        adjust = True

        inp.skipBytes(4)
        self.core[0].littleEndian = inp.read() == 0x49
        inp.order(self.isLittleEndian())

        self.metadataOffsets: "list[int]" = []
        self.pixelOffsets: "list[int]" = []
        self.pixelLengths: "list[int]" = []
        ndFilters: "list[float | None]" = []
        imageDescriptions: "dict[int, str]" = {}

        inp.seek(0)

        # gather offsets to metadata and pixel data blocks (Java 242-396)
        visited: "set[int]" = set()
        while inp.getFilePointer() < inp.length() - 8:
            self._checkpoint(inp.getFilePointer())
            # Not in the Java: each pass of this loop is decided by the file
            # pointer it starts from, so starting from the same one twice
            # means it would never terminate -- Bio-Formats hangs on such a
            # file; raising is the only useful behaviour.
            if inp.getFilePointer() in visited:
                raise ValueError(
                    f"SlideBook block structure not understood at byte {inp.getFilePointer()} "
                    "(the Bio-Formats reader loops forever on this file)"
                )
            visited.add(inp.getFilePointer())
            inp.skipBytes(4)
            checkOne = inp.read()
            checkTwo = inp.read()
            if (checkOne == ord("I") and checkTwo == ord("I")) or (checkOne == ord("M") and checkTwo == ord("M")):
                self.metadataOffsets.append(inp.getFilePointer() - 6)
                inp.skipBytes(inp.readShort() - 8)
            elif checkOne == -1 and checkTwo == -1:
                foundBlock = False
                block = bytearray(8192)
                inp.read(block)
                while not foundBlock:
                    m = _META_MARKER.search(block, 0, len(block))
                    if m is not None and m.start() < len(block) - 2:
                        i = m.start()
                        foundBlock = True
                        inp.seek(inp.getFilePointer() - len(block) + i - 2)
                        self.metadataOffsets.append(inp.getFilePointer() - 2)
                        inp.skipBytes(inp.readShort() - 5)
                    if not foundBlock:
                        block[0] = block[len(block) - 2]
                        block[1] = block[len(block) - 1]
                        if inp.read(block, 2, len(block) - 2) == -1:
                            # Java loops forever here (it re-scans the same
                            # stale buffer at EOF); stopping is the only
                            # behaviour a caller can use.
                            raise JavaEOFError("unterminated 0xFFFF metadata block")
            else:
                s = None
                fp = inp.getFilePointer() - 6
                inp.seek(fp)
                length_byte = inp.read()
                if 0 < length_byte <= 32:
                    s = inp.readString(length_byte)

                if s is not None and "Annotation" in s:
                    if s == "CTimelapseAnnotation":
                        inp.skipBytes(41)
                        if inp.read() == 0:
                            inp.skipBytes(10)
                        else:
                            inp.seek(inp.getFilePointer() - 1)
                    elif s == "CIntensityBarAnnotation":
                        inp.skipBytes(56)
                        n = inp.read()
                        while n == 0 or n < 6 or n > 0x80:
                            n = inp.read()
                        inp.seek(inp.getFilePointer() - 1)
                    elif s == "CCubeAnnotation":
                        inp.skipBytes(66)
                        n = inp.read()
                        if n != 0:
                            inp.seek(inp.getFilePointer() - 1)
                    elif s == "CScaleBarAnnotation":
                        inp.skipBytes(38)
                        extra = inp.read()
                        if extra <= 16:
                            inp.skipBytes(3 + extra)
                        else:
                            inp.skipBytes(2)
                elif s is not None and "Decon" in s:
                    inp.seek(fp)
                    while inp.read() != ord("]"):
                        pass
                else:
                    if _mod(fp, 2) == 1:
                        fp -= 2
                    inp.seek(fp)

                    # make sure there isn't another block nearby
                    checkString = inp.readString(64)
                    if "II" in checkString or "MM" in checkString:
                        index = checkString.find("II")
                        if index == -1:
                            index = checkString.find("MM")
                        inp.seek(fp + index - 4)
                        continue
                    inp.seek(fp)

                    self.pixelOffsets.append(fp)
                    try:
                        # Java 336-374: find where this pixel block ends. Same
                        # answer either way; `scan_fast` uses large prefetched
                        # reads instead of 8 KB ones (see _scan.py).
                        scan = scan_reference if self._reference_scan else scan_fast
                        if self._reference_scan:
                            res = scan(self._scan_fh, fp, inp.length())
                        else:
                            res = scan(self._scan_fh, fp, inp.length(), on_chunk=self._checkpoint)
                        inp.seek(res.pointer)
                        if res.found and (not res.hi_family or res.is_i):
                            self.pixelOffsets.pop()

                        if inp.getFilePointer() <= inp.length():
                            if len(self.pixelOffsets) > len(self.pixelLengths):
                                length = inp.getFilePointer() - fp
                                if _mod(_div(length, 2), 2) == 1:
                                    self.pixelOffsets[-1] = fp + 2
                                    length -= 2
                                if length >= 1024:
                                    self.pixelLengths.append(length)
                                else:
                                    self.pixelOffsets.pop()
                        else:
                            self.pixelOffsets.pop()
                    except JavaEOFError:
                        self.pixelOffsets.pop()

        # Java 398-420
        orderedSeries: "list[int]" = []
        uniqueSeries: "dict[int, list[int]]" = {}
        i = 0
        while i < len(self.pixelOffsets):
            length = self.pixelLengths[i]
            offset = self.pixelOffsets[i]
            padding = 0 if self.isSpool else 7
            if length + offset + padding > inp.length():
                self.pixelOffsets.pop(i)
                self.pixelLengths.pop(i)
                i -= 1
            else:
                v = uniqueSeries.setdefault(length, [])
                if not v:
                    orderedSeries.append(length)
                v.append(i)
            i += 1

        # Java 422-441
        if len(self.pixelOffsets) > 1:
            little = self.isLittleEndian()
            seriesCount = 0
            for key in orderedSeries:
                nBlocks = len(uniqueSeries.get(key, []))
                if nBlocks == 0:
                    nBlocks += 1
                seriesCount += nBlocks
            self.core = [CoreMetadata(littleEndian=little) for _ in range(seriesCount)]

        # Java 443-468
        pixelSize: "dict[int, float]" = {}
        objectives: "dict[int, str]" = {}
        magnifications: "dict[int, int]" = {}
        pixelSizeZ: "list[float | None]" = []
        exposureTimes: "list[int]" = []

        imageNames: "list[str | None]" = [None] * self.getSeriesCount()
        channelNames: "list[str]" = []
        nextName = 0

        n_px = len(self.pixelOffsets)
        sizeX = [0] * n_px
        sizeY = [0] * n_px
        sizeZ = [0] * n_px
        sizeC = [0] * n_px
        divValues = [0] * n_px

        # try to find the width and height (Java 470-756)
        iCount = 0
        hCount = 0
        uCount = 0
        prevSeries = -1
        prevSeriesU = -1
        for mi in range(len(self.metadataOffsets)):
            off = self.metadataOffsets[mi]
            if self.isSpool and off == 0:
                off = 276
            inp.seek(off)
            nxt = inp.length() if mi == len(self.metadataOffsets) - 1 else self.metadataOffsets[mi + 1]
            totalBlocks = self._i32(_div(nxt - off, 128))
            q = 0
            while q < totalBlocks:
                block_start = off + self._i32(q * 128)
                if self._withinPixels(block_start):
                    # Speed-up 2 (module docstring): jump to the first q past
                    # this pixel block instead of testing every 128 bytes.
                    if not self._overflow or q * 128 < 2**31:
                        end = self._pixel_block_end_containing(block_start)
                        q = max(q + 1, _div(end - off + 127, 128))
                    else:
                        q += 1
                    continue
                inp.seek(off + q * 128)
                n = inp.readShort() & 0xFFFF  # (char) readShort()
                while n == 0 and inp.getFilePointer() < off + (q + 1) * 128:
                    n = inp.readShort() & 0xFFFF
                if inp.getFilePointer() >= inp.length() - 2:
                    break
                ch = chr(n)
                if ch == "i":
                    iCount += 1
                    inp.skipBytes(70)
                    expTime = inp.readInt()
                    if expTime > 0:
                        exposureTimes.append(expTime)
                    inp.skipBytes(20)
                    size = float(inp.readFloat())
                    pixelSizeZ.append(size if size - _EPSILON > 0 else None)
                    inp.seek(inp.getFilePointer() - 20)

                    for j in range(n_px):
                        end = inp.length() if j == n_px - 1 else self.pixelOffsets[j + 1]
                        if inp.getFilePointer() < end:
                            if sizeX[j] == 0:
                                x = inp.readShort()
                                y = inp.readShort()
                                if x != 0 and y != 0:
                                    sizeX[j] = x
                                    sizeY[j] = y
                                    checkX = inp.readShort()
                                    checkY = inp.readShort()
                                    div = inp.readShort()
                                    if checkX == checkY:
                                        divValues[j] = div
                                        sizeX[j] = _div(sizeX[j], 1 if div == 0 else div)
                                        div = inp.readShort()
                                        sizeY[j] = _div(sizeY[j], 1 if div == 0 else div)
                                else:
                                    inp.skipBytes(8)
                            if prevSeries != j:
                                iCount = 1
                            prevSeries = j
                            sizeC[j] = iCount
                            break
                elif ch == "u":
                    uCount += 1
                    for j in range(self.getSeriesCount()):
                        end = inp.length() if j == self.getSeriesCount() - 1 else self.pixelOffsets[j + 1]
                        if inp.getFilePointer() < end:
                            if prevSeriesU != j:
                                uCount = 1
                            prevSeriesU = j
                            sizeZ[j] = uCount
                            break
                elif ch == "h":
                    hCount += 1
                elif ch == "j":
                    inp.skipBytes(2)
                    check = inp.readString(2)
                    if check in ("II", "MM"):
                        pointer = inp.getFilePointer()
                        # this block should contain an image name
                        inp.skipBytes(10)
                        if nextName < len(imageNames):
                            name = _jtrim(self._readCString())
                            if len(name) > 0:
                                imageNames[nextName] = name
                                nextName += 1

                        fp = inp.getFilePointer()
                        if _mod(inp.getFilePointer(), 2) == 1:
                            inp.skipBytes(1)
                        while inp.readShort() == 0:
                            pass
                        if inp.readShort() == 0:
                            inp.skipBytes(4)
                        else:
                            inp.skipBytes(16)
                        diff = inp.getFilePointer() - fp
                        if diff > 123 and _mod(fp, 2) == 0 and diff != 142 and diff != 143 and diff != 130:
                            inp.seek(fp + 123)

                        x = inp.readInt()
                        y = inp.readInt()

                        if x > 0x8000 or y > 0x8000:
                            inp.seek(inp.getFilePointer() - 7)
                            x = inp.readInt()
                            y = inp.readInt()
                        elif x == 0 or y == 0:
                            inp.seek(inp.getFilePointer() - 27)
                            x = inp.readInt()
                            y = inp.readInt()

                        div = inp.readShort()
                        x = _div(x, 1 if (div == 0 or div > 0x100) else div)
                        div = inp.readShort()
                        y = _div(y, 1 if (div == 0 or div > 0x100) else div)

                        if x > 0x10000 or y > 0x10000:
                            inp.seek(inp.getFilePointer() - 11)
                            x = inp.readInt()
                            y = inp.readInt()
                            div = inp.readShort()
                            x = _div(x, 1 if div == 0 else div)
                            div = inp.readShort()
                            y = _div(y, 1 if div == 0 else div)

                            if x > 0x10000 or y > 0x10000:
                                inp.skipBytes(2)
                                x = inp.readInt()
                                y = inp.readInt()
                                div = inp.readShort()
                                x = _div(x, 1 if div == 0 else div)
                                div = inp.readShort()
                                y = _div(y, 1 if div == 0 else div)

                        if (
                            nextName >= 1 and x > 16
                            and (x < sizeX[nextName - 1] or sizeX[nextName - 1] == 0)
                            and y > 16 and (y < sizeY[nextName - 1] or sizeY[nextName - 1] == 0)
                        ):
                            sizeX[nextName - 1] = x
                            sizeY[nextName - 1] = y
                            adjust = False

                        inp.seek(pointer + 214)
                        validBits = inp.readShort()
                        if nextName >= 1 and self.core[nextName - 1].bitsPerPixel == 0 and 0 < validBits <= 16:
                            self.core[nextName - 1].bitsPerPixel = validBits
                elif ch == "m":
                    # this block should contain a channel name
                    if inp.getFilePointer() > self.pixelOffsets[0] or self.isSpool:
                        inp.skipBytes(14)
                        name = _jtrim(self._readCString())
                        if len(name) > 1:
                            channelNames.append(name)
                elif ch == "d":
                    # objective info and pixel size X/Y
                    inp.skipBytes(6)
                    fp = inp.getFilePointer()
                    while inp.read() == 0:
                        pass
                    inp.seek(inp.getFilePointer() - 1)
                    nSkipped = inp.getFilePointer() - fp
                    if nSkipped < 8:
                        inp.skipBytes(8 - nSkipped)
                    objective = _jtrim(self._readCString())
                    inp.seek(fp + 144)
                    pixSize = inp.readFloat()
                    magnification = inp.readShort()

                    mult = 1
                    if len(pixelSize) < len(divValues):
                        mult = divValues[len(pixelSize)]
                    v = _f32(pixSize * _f32(float(mult)))
                    if v - _EPSILON > 0:
                        pixelSize[nextName - 1] = v
                        objectives[nextName - 1] = objective
                        magnifications[nextName - 1] = magnification
                elif ch == "e":
                    inp.skipBytes(174)
                    ndFilters.append(float(inp.readFloat()))
                    inp.skipBytes(40)
                    if 0 <= nextName < self.getSeriesCount():
                        self.setSeries(nextName)
                        self._meta(nextName).original.setdefault("channel intensification", []).append(inp.readShort())
                elif ch == "k":
                    inp.skipBytes(14)
                    if nextName > 0:
                        self.setSeries(nextName - 1)
                    self._meta(self.getSeries()).original["Mag. changer"] = self._readCString()
                elif ch == "n":
                    fp1 = inp.getFilePointer()
                    inp.seek(inp.getFilePointer() - 3)
                    while inp.read() != 0:
                        inp.seek(inp.getFilePointer() - 2)
                    fp2 = inp.getFilePointer()
                    length_n = inp.read() - 1

                    currentSeries = 0
                    for j in range(n_px):
                        end = inp.length() if j == n_px - 1 else self.pixelOffsets[j + 1]
                        if inp.getFilePointer() < end:
                            currentSeries = j
                            break

                    if length_n > 0 and fp1 - fp2 != 2:
                        if fp2 < fp1:
                            inp.seek(inp.getFilePointer() - 1)
                            descr = self._readCString()
                            if len(descr) < 2:  # Java: StringIndexOutOfBoundsException
                                raise ValueError("SlideBook 'n' block: description too short")
                            descr = descr[: len(descr) - 2]
                            if not descr.endswith("Annotatio"):
                                imageDescriptions[currentSeries] = _jtrim(descr)
                        else:
                            imageDescriptions[currentSeries] = _jtrim(inp.readString(length_n))
                elif self.isSpool:
                    # spool files don't necessarily have block identifiers
                    for j in range(n_px):
                        end = inp.length() if j == n_px - 1 else self.pixelOffsets[j + 1]
                        if inp.getFilePointer() < end:
                            inp.skipBytes(14)
                            check_s = inp.readShort()
                            x = inp.readShort()
                            y = inp.readShort()
                            if check_s == 0 and x > 16 and y > 16:
                                sizeX[j] = x
                                sizeY[j] = y
                            adjust = False
                            break
                q += 1

        # Java 758-806 -- series flattening by repeated image name
        currentName = imageNames[0]
        realCore: "list[CoreMetadata]" = []
        t = 1
        noFlattening = currentName is not None and currentName == "Untitled"
        for i in range(1, self.getSeriesCount()):
            if (
                imageNames[i] is None or imageNames[i] != currentName or noFlattening
                or (i == 1 and (sizeX[i - 1] != sizeX[i] or sizeY[i - 1] != sizeY[i]
                                or sizeC[i - 1] != sizeC[i] or sizeZ[i - 1] != sizeZ[i]))
            ):
                currentName = imageNames[i]
                nextCore = self.core[i - 1]
                nextCore.sizeT = t
                realCore.append(nextCore)
                if t == 1:
                    noFlattening = True
                t = 1
                if i == 1:
                    noFlattening = True
            else:
                t += 1
        self.core[self.getSeriesCount() - 1].sizeT = t
        realCore.append(self.core[self.getSeriesCount() - 1])
        flattened = False
        if len(self.core) != len(realCore) and not noFlattening:
            flattened = True
            self.core = realCore
            orderedSeries = []
            uniqueSeries = {}
            nextIndex = 0
            for i in range(len(self.core)):
                thisSeries = i
                orderedSeries.append(thisSeries)
                uniqueSeries.setdefault(thisSeries, []).append(nextIndex)
                length = self.pixelLengths[nextIndex]
                length *= self.core[i].sizeT
                self.pixelLengths[i] = length
                nextIndex += self.core[i].sizeT

        self.planeOffset: "list[list[int]]" = [[] for _ in range(self.getSeriesCount())]

        divByTwo = False
        divZByTwo = False

        nextPixelIndex = 0
        nextBlock = 0
        nextOffsetIndex = 0
        i32 = self._i32

        for i in range(self.getSeriesCount()):  # Java 817-1207
            self.setSeries(i)
            ms = self.core[i]

            pixelIndexes = uniqueSeries.get(orderedSeries[nextPixelIndex], [])
            nBlocks = len(pixelIndexes)
            if nextBlock >= nBlocks:
                nextPixelIndex += 1
                nextBlock = 0
                pixelIndexes = uniqueSeries.get(orderedSeries[nextPixelIndex], [])
                nBlocks = len(pixelIndexes)
            else:
                nextBlock += 1
            index = pixelIndexes[0] if len(pixelIndexes) == self.getSeriesCount() else i

            pixels = _div(self.pixelLengths[index], 2)
            x = True

            ms.sizeX = sizeX[index]
            ms.sizeY = sizeY[index]
            ms.sizeC = sizeC[index]
            ms.sizeZ = sizeZ[index]

            if self.getSizeC() > 64:
                # dimensions are probably incorrect
                ms.sizeC = 1
                ms.sizeZ = 1
                ms.sizeX = _div(ms.sizeX, 2)
                ms.sizeY = _div(ms.sizeY, 2)

            isMontage = False
            if i > 1 and ((imageNames[i] is not None and imageNames[i].startswith("Montage")) or self.getSizeC() >= 32):
                ms.sizeC = self.core[1].sizeC
                ms.sizeZ = self.core[1].sizeZ
                isMontage = True

            cGreater = ms.sizeC > ms.sizeZ

            if self.isSpool and ms.sizeC == 0:
                ms.sizeC = len(channelNames)

            if _mod(ms.sizeZ, nBlocks) == 0 and nBlocks != self.getSizeC():
                z = _div(ms.sizeZ, nBlocks)
                if z <= nBlocks:
                    ms.sizeZ = z

            if divByTwo:
                ms.sizeX = _div(ms.sizeX, 2)

            if divZByTwo and ms.sizeC > 1:
                ms.sizeZ = _div(_div(pixels, i32(ms.sizeX * ms.sizeY)), 2)
                ms.sizeC = 2

            if self.getSizeC() == 0:
                ms.sizeC = 1
            if self.getSizeZ() == 0:
                ms.sizeZ = 1

            plane = _div(pixels, i32(self.getSizeC() * self.getSizeZ()))
            if self.getSizeT() > 0:
                plane = _div(plane, self.getSizeT())

            X, Y = self.getSizeX, self.getSizeY
            if i32(X() * Y()) == pixels:
                if self.getSizeC() == 2 and _mod(X(), 2) == 0 and _mod(Y(), 2) == 0:
                    if self.getSizeC() != self.getSizeZ():
                        ms.sizeX = _div(ms.sizeX, 2)
                        divByTwo = True
                    else:
                        divZByTwo = True
                        ms.sizeC = 1
                else:
                    ms.sizeC = 1
                ms.sizeZ = 1
            elif i32(i32(X() * Y()) * self.getSizeZ()) == pixels:
                if (
                    self.getSizeC() == 2 and self.getSizeC() != self.getSizeZ()
                    and _mod(X(), 2) == 0 and _mod(Y(), 2) == 0
                    and (i == 0 or self.core[i - 1].sizeC > 1)
                ):
                    ms.sizeX = _div(ms.sizeX, 2)
                    divByTwo = True
                else:
                    ms.sizeC = 1
                    ms.sizeZ = _div(pixels, i32(X() * Y()))
            elif i32(i32(X() * Y()) * self.getSizeC()) == pixels:
                ms.sizeC = _div(pixels, i32(X() * Y()))
                ms.sizeZ = 1
            elif i32(i32(_div(X(), 2) * _div(Y(), 2)) * self.getSizeZ()) == pixels:
                ms.sizeX = _div(ms.sizeX, 2)
                ms.sizeY = _div(ms.sizeY, 2)
            elif i32(i32(i32(i32(_div(X(), 2) * _div(Y(), 2)) * self.getSizeC()) * self.getSizeZ()) * self.getSizeT()) == pixels:
                ms.sizeX = _div(ms.sizeX, 2)
                ms.sizeY = _div(ms.sizeY, 2)
            else:
                validSizes = abs(X() * Y()) <= _INT_MAX  # DataTools.safeMultiply32
                if X() == 0 or Y() == 0 or not validSizes:
                    ms.sizeX = _div(sizeX[index], 256)
                    ms.sizeY = _div(sizeY[index], 256)
                p = _div(pixels, i32(X() * Y()))
                if pixels == p * X() * Y():
                    if p != self.getSizeC() * self.getSizeZ():
                        if (
                            self.getSizeC() > 1 and self.core[i].sizeZ >= _div(p, self.getSizeC() - 1)
                            and p >= self.getSizeC() - 1 and p > 2
                        ):
                            self.core[i].sizeC -= 1
                            self.core[i].sizeZ = _div(p, self.getSizeC())
                        elif _mod(p, self.getSizeC()) != 0:
                            self.core[i].sizeC = 1
                            self.core[i].sizeZ = p
                        elif ms.sizeZ == p + 1:
                            ms.sizeC = 1
                            ms.sizeZ = 1
                            ms.sizeT = p
                        elif self.getSizeC() > 1 and ms.sizeZ == _div(p, self.getSizeC() - 1) + 1:
                            ms.sizeC -= 1
                            ms.sizeZ = 1
                            ms.sizeT = _div(p, self.getSizeC())
                        else:
                            if p > self.getSizeZ() and _div(p, self.getSizeZ()) < self.getSizeZ() - 1:
                                ms.sizeT = _div(p, self.getSizeC())
                                ms.sizeZ = 1
                            elif _mod(pixels, X()) == 0 and _mod(pixels, Y()) == 0:
                                while i32(X() * Y()) > plane:
                                    ms.sizeX = _div(ms.sizeX, 2)
                                    ms.sizeY = _div(ms.sizeY, 2)
                                originalX = X()
                                while i32(X() * Y()) < plane:
                                    ms.sizeX += originalX
                                    ms.sizeY = _div(plane, X())
                                newX = X() + originalX
                                if newX * _div(plane, newX) == plane and not flattened:
                                    ms.sizeX = newX
                                    ms.sizeY = _div(plane, newX)
                            elif not adjust:
                                ms.sizeZ = _div(p, self.getSizeC())
                            elif isMontage:
                                pixels = _div(pixels, self.getSizeC())
                                while pixels != i32(X() * Y()) or _div(Y(), X()) > 2:
                                    ms.sizeX += 16
                                    ms.sizeY = _div(pixels, X())
                elif self.isSpool:
                    ms.sizeZ = _div(p, self.getSizeC())
                elif p == 0:
                    adjust = True
                    if self.getSizeC() > 1:
                        ms.sizeC = 2 if self.getSizeC() == 3 else 1
                else:
                    if ms.sizeC > 1 and p <= ms.sizeC:
                        z = self.getSizeZ()
                        ms.sizeZ = 1
                        ms.sizeC = p
                        ms.sizeT = 1

                        if isMontage and pixels == X() * _div(pixels, X()):
                            pixels = _div(pixels, self.getSizeC())
                            while pixels != i32(X() * Y()):
                                ms.sizeX -= 16
                                ms.sizeY = _div(pixels, X())
                        elif not isMontage:
                            ms.sizeZ = z
                            adjust = True
                    elif isMontage:
                        pixels = _div(pixels, i32(self.getSizeC() * self.getSizeZ()))
                        originalX = X()
                        originalY = Y()
                        xGreater = X() > Y()
                        while i32(X() * Y()) != 0 and (
                            _mod(pixels, i32(X() * Y())) != 0 or (Y() / X() > 2)
                        ):
                            ms.sizeX += originalX
                            ms.sizeY = _div(pixels, X())
                            if not xGreater and X() >= Y():
                                break
                        if i32(X() * Y()) == 0:
                            if pixels != i32(X() * Y()):
                                pixels *= self.getSizeC() * self.getSizeZ()
                                ms.sizeX = originalX
                                ms.sizeY = originalY
                                isMontage = False
                        if _mod(pixels, originalX - _div(originalX, 4)) == 0:
                            newX = originalX - _div(originalX, 4)
                            newY = _div(pixels, newX)
                            if newX * newY == pixels:
                                ms.sizeX = newX
                                ms.sizeY = newY
                                isMontage = True
                                adjust = False
                    elif p != i32(self.getSizeZ() * self.getSizeC()):
                        if _mod(pixels, X()) == 0 and _mod(pixels, Y()) == 0:
                            while i32(X() * Y()) > plane:
                                ms.sizeX = _div(ms.sizeX, 2)
                                ms.sizeY = _div(ms.sizeY, 2)
                        else:
                            ms.sizeZ = 1
                            ms.sizeC = 1
                            ms.sizeT = p

            if self.getSizeC() == 0:
                ms.sizeC = 1
            if self.getSizeZ() == 0:
                ms.sizeZ = 1

            div = i32(self.getSizeC() * self.getSizeZ())
            if self.getSizeT() > 0:
                div = i32(div * self.getSizeT())
            if div > 1:
                plane = _div(pixels, div)

            diff = 2 * (pixels - i32(i32(X() * Y()) * div))
            if _mod(self.pixelLengths[index], 2) == 1:
                diff += 1

            if abs(diff) > _div(plane, 2):
                diff = 0

            if adjust and diff == 0:
                ratio = X() / Y()
                widthGreater = X() > Y()
                while i32(X() * Y()) > plane:
                    if x:
                        ms.sizeX = _div(ms.sizeX, 2)
                    else:
                        ms.sizeY = _div(ms.sizeY, 2)
                    x = not x
                if i32(X() * Y()) != plane:
                    while ratio - (X() / Y()) >= 0.01:
                        first = True
                        while first or i32(X() * Y()) < plane or (X() < Y() and widthGreater):
                            if first:
                                first = False
                            ms.sizeX += 1
                            ms.sizeY = _div(plane, X())

            nPlanes = i32(self.getSizeZ() * self.getSizeC())
            ms.sizeT = _div(pixels, i32(i32(X() * Y()) * nPlanes))
            while i32(i32(i32(X() * Y()) * nPlanes) * self.getSizeT()) > pixels:
                ms.sizeT -= 1
            if self.getSizeT() == 0:
                ms.sizeT = 1

            if cGreater and self.getSizeC() == 1 and self.getSizeZ() > 1:
                ms.sizeC = self.getSizeZ()
                ms.sizeZ = 1

            ms.imageCount = i32(nPlanes * self.getSizeT())
            ms.pixelType = "uint16"  # FormatTools.UINT16
            ms.dimensionOrder = "XYZCT" if nBlocks > 1 else "XYZTC"

            self.planeOffset[i] = [0] * self.getImageCount()
            nextImage = 0
            offset = self.pixelOffsets[i]
            planeSize = i32(i32(X() * Y()) * 2)

            if diff < planeSize:
                offset += diff
            else:
                offset += _mod(diff, planeSize)

            length = self.pixelLengths[i]
            planes = _div(length, planeSize)
            if planes > ms.imageCount:
                planes = ms.imageCount

            for p in range(planes):
                zct = self.getZCTCoords(p)
                if flattened and zct[0] == 0 and zct[1] == 0:
                    offset = self.pixelOffsets[nextOffsetIndex]
                    nextOffsetIndex += 1

                    if (
                        zct[2] > 0 and _mod(self.planeOffset[i][nextImage - 1], 2) != _mod(offset, 2)
                        and (offset - self.planeOffset[i][nextImage - 1] > i32(i32(3 * X()) * Y()))
                        and diff == 0
                    ):
                        diff = 31
                    if diff < planeSize:
                        offset += diff
                    else:
                        offset += _mod(diff, planeSize)

                    self.planeOffset[i][nextImage] = offset
                elif flattened and zct[0] == 0:
                    idx = self.getIndex(0, 0, zct[2])
                    self.planeOffset[i][nextImage] = self.planeOffset[i][idx] + i32(zct[1] * planeSize)
                elif flattened:
                    self.planeOffset[i][nextImage] = self.planeOffset[i][nextImage - 1] + planeSize
                elif nextImage < len(self.planeOffset[i]):
                    # Java 1204: `offset + p * planeSize` -- the int product
                    # wraps past 2 GiB in Bio-Formats (see module docstring).
                    self.planeOffset[i][nextImage] = offset + i32(p * planeSize)
                nextImage += 1
        self.setSeries(0)

        # Java 1210-1223
        if len(pixelSizeZ) > 0:
            seriesIndex = 0
            for q in range(self.getSeriesCount()):
                msq = self.core[q]
                inc = msq.sizeC * msq.sizeT
                if seriesIndex + inc > len(pixelSizeZ):
                    z = msq.sizeT
                    msq.sizeT = msq.sizeZ
                    msq.sizeZ = z
                    inc = msq.sizeC * msq.sizeT
                seriesIndex += inc

        # Java 1225-1344 -- metadata store, kept as plain values
        for i in range(self.getSeriesCount()):
            meta = self._meta(i)
            if imageNames[i] is not None:
                meta.name = imageNames[i]
            meta.description = imageDescriptions.get(i, "")
            objective = objectives.get(i)
            if objective is not None:
                meta.objective_model = objective
                if magnifications.get(i, 0) > 0:
                    meta.objective_magnification = float(magnifications[i])

        exposureIndex = len(exposureTimes) - len(channelNames)
        if exposureIndex >= 1:
            exposureIndex += 1
        for i in range(self.getSeriesCount()):
            self.setSeries(i)
            meta = self._meta(i)
            if pixelSize.get(i) is not None:
                meta.physical_size_x_um = _physical_size(pixelSize[i])
                meta.physical_size_y_um = _physical_size(pixelSize[i])
            idx = 0
            for q in range(i):
                idx += self.core[q].sizeC * self.core[q].sizeT
            if idx < len(pixelSizeZ) and pixelSizeZ[idx] is not None:
                meta.physical_size_z_um = _physical_size(pixelSizeZ[idx])
            for plane_no in range(self.getImageCount()):
                c = self.getZCTCoords(plane_no)[1]
                if 0 <= exposureIndex + c < len(exposureTimes):
                    meta.plane_exposure_times[plane_no] = float(exposureTimes[exposureIndex + c])
            exposureIndex += self.getSizeC()
        self.setSeries(0)

        index = 0
        for i in range(self.getSeriesCount()):
            self.setSeries(i)
            meta = self._meta(i)
            for c in range(self.getSizeC()):
                if index < len(channelNames) and channelNames[index] is not None:
                    meta.channel_names[c] = channelNames[index]
                if index < len(ndFilters) and ndFilters[index] is not None:
                    meta.channel_nd_filters[c] = ndFilters[index]
                index += 1
        self.setSeries(0)
