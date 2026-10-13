# tanoa-sld -- 3i SlideBook reader for tanoa. GPL-2.0-or-later; see LICENSE.
"""Synthetic SlideBook (.sld) files, built from the block structure
SlidebookReader.java's own comments describe (128-byte typed metadata blocks
with an 'II'/'MM' marker at +4 and a length at +6; variable-length 0xFFFF
blocks; annotation-string blocks; pixel blocks of uint16 terminated by the
next 'h' block).

These files are NOT claimed to be what SlideBook writes. Their purpose is a
differential test: Bio-Formats' own reader and this port are both run on the
same bytes and must agree on every dimension and every pixel -- including
wherever Bio-Formats' heuristics produce a surprising answer. Real files are
still the final word (see tests/test_against_bioformats.py's real-file mode).
"""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Series:
    width: int
    height: int
    n_channels: int = 1
    n_z: int = 1
    name: "str | None" = None
    channel_names: "list[str]" = field(default_factory=list)
    pixel_size_um: "float | None" = None
    z_step_um: "float | None" = None
    objective: "str | None" = None


class Builder:
    def __init__(self, little: bool = True, seed: int = 0) -> None:
        self.little = little
        self.e = "<" if little else ">"
        self.buf = bytearray()
        self.rng = np.random.default_rng(seed)
        self.planes: "list[np.ndarray]" = []

    # -- blocks -----------------------------------------------------------
    def _block(self, type_code: int, length: int = 128) -> bytearray:
        b = bytearray(length)
        struct.pack_into(self.e + "H", b, 0, type_code)
        b[4:6] = b"II" if self.little else b"MM"
        struct.pack_into(self.e + "h", b, 6, length)
        return b

    def header(self) -> None:
        b = self._block(0x006C)
        struct.pack_into(self.e + "H", b, 2, 0x0100)
        self.buf += b

    def i_block(self, s: Series) -> None:
        b = self._block(ord("i"))
        struct.pack_into(self.e + "i", b, 72, 100)  # exposure
        struct.pack_into(self.e + "f", b, 96, s.z_step_um or 0.0)
        struct.pack_into(self.e + "hhhhhh", b, 80, s.width, s.height, 0, 0, 1, 1)
        self.buf += b

    def u_block(self) -> None:
        self.buf += self._block(ord("u"))

    def h_block(self) -> None:
        self.buf += self._block(ord("h"))

    def j_block(self, name: str) -> None:
        b = self._block(ord("j"), 256)
        raw = name.encode() + b"\0"
        b[16:16 + len(raw)] = raw
        self.buf += b

    def m_block(self, name: str) -> None:
        b = self._block(ord("m"))
        raw = name.encode() + b"\0"
        b[16:16 + len(raw)] = raw
        self.buf += b

    def d_block(self, objective: str, pixel_size_um: float, magnification: int = 60) -> None:
        b = self._block(ord("d"), 256)
        raw = objective.encode() + b"\0"
        b[16:16 + len(raw)] = raw
        struct.pack_into(self.e + "f", b, 8 + 144, pixel_size_um)
        struct.pack_into(self.e + "h", b, 8 + 148, magnification)
        self.buf += b

    def ffff_block(self, payload: int = 300) -> None:
        """A variable-length 0xFFFF block, closed by the next typed block."""
        b = bytearray(8 + payload)
        b[4:6] = b"\xff\xff"
        self.buf += b
        while len(self.buf) % 2:
            self.buf += b"\0"

    def annotation_block(self, kind: str = "CScaleBarAnnotation") -> None:
        raw = kind.encode()
        b = bytearray(128)
        b[0] = len(raw)
        b[1:1 + len(raw)] = raw
        self.buf += b

    def pixels(self, s: Series, n_planes: int) -> None:
        # Both bytes of every uint16 in 0x80..0xFE: never 'I'/'M'/'h'/'i'/0
        # or 0xFF, so no marker the scanner looks for can occur inside.
        hi = self.rng.integers(0x80, 0xFF, size=(n_planes, s.height, s.width), dtype=np.uint16)
        lo = self.rng.integers(0x80, 0xFF, size=(n_planes, s.height, s.width), dtype=np.uint16)
        data = (hi << 8) | lo
        self.planes.append(data)
        self.buf += data.astype(self.e + "u2").tobytes()

    def pad(self, n: int = 512) -> None:
        self.buf += bytes(n)

    def write(self, path: Path) -> Path:
        path.write_bytes(bytes(self.buf))
        return path


def simple(path: Path, s: Series, *, little: bool = True, seed: int = 0, extras: bool = False) -> Path:
    """Header, per-channel 'i' blocks, per-Z 'u' blocks, optional name and
    objective, one pixel block, then the closing 'h' block."""
    b = Builder(little=little, seed=seed)
    b.header()
    if s.name:
        b.j_block(s.name)
    if s.objective and s.pixel_size_um:
        b.d_block(s.objective, s.pixel_size_um)
    for _ in range(s.n_channels):
        b.i_block(s)
    for _ in range(s.n_z):
        b.u_block()
    if extras:
        b.annotation_block()
    b.pixels(s, s.n_channels * s.n_z)
    b.h_block()
    for name in s.channel_names:
        b.m_block(name)
    b.pad()
    return b.write(path)


def multi(path: Path, series: "list[Series]", *, little: bool = True, seed: int = 0) -> Path:
    b = Builder(little=little, seed=seed)
    b.header()
    for s in series:
        if s.name:
            b.j_block(s.name)
        for _ in range(s.n_channels):
            b.i_block(s)
        for _ in range(s.n_z):
            b.u_block()
        b.pixels(s, s.n_channels * s.n_z)
        b.h_block()
    b.pad()
    return b.write(path)


def random_file(path: Path, seed: int) -> Path:
    rnd = random.Random(seed)
    n_series = rnd.choice([1, 1, 2, 3])
    shape = (rnd.choice([32, 48, 64, 96, 128]), rnd.choice([32, 40, 64, 100, 128]))
    series = []
    for k in range(n_series):
        w, h = shape if rnd.random() < 0.6 else (rnd.choice([32, 64, 80]), rnd.choice([32, 64, 72]))
        series.append(Series(
            width=w, height=h, n_channels=rnd.choice([1, 1, 2, 3]), n_z=rnd.choice([1, 2, 3, 5]),
            name=rnd.choice([None, "Capture 1", f"Capture {k + 1}"]),
            z_step_um=rnd.choice([0.0, 0.5, 1.0]),
        ))
    if n_series == 1 and rnd.random() < 0.5:
        s = series[0]
        s.objective, s.pixel_size_um = "Plan-Apo 60x", rnd.choice([0.108, 0.215])
        s.channel_names = [f"Ch{c}" for c in range(s.n_channels)]
        return simple(path, s, little=rnd.random() < 0.7, seed=seed, extras=rnd.random() < 0.5)
    return multi(path, series, little=rnd.random() < 0.7, seed=seed)
