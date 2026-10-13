# tanoa-sld -- GPL-2.0-or-later; see LICENSE.
"""Minimal, fast checks. The one that matters most: `scan_fast` (large
prefetched reads) must give exactly what the literal transcription of the
Java loop gives -- including markers in the loop's 6-byte blind spots."""
import io
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import synthetic as syn  # noqa: E402

from tanoa_sld._scan import scan_fast, scan_reference  # noqa: E402

FP = 1000
MARKERS = [b"h\x00\x00\x00II", b"\x00i\x00\x00MM", b"j\x00\x00\x00II", b"on\x00\x00MM"]


def _blind_spot(k: int) -> int:
    return FP + 8192 + (k - 1) * 8172 + 8146  # first unscanned start after window k


@pytest.mark.parametrize("case", range(60))
def test_fast_scan_equals_the_java_loop(case):
    rnd = random.Random(case)
    size = rnd.choice([9_000, 40_000, 120_000])
    data = bytearray(rnd.randrange(0x80, 0xFF) for _ in range(size))
    for _ in range(rnd.choice([0, 1, 3])):
        k = rnd.randrange(1, 8)
        where = rnd.choice([rnd.randrange(FP - 50, size - 10), _blind_spot(k) + rnd.randrange(-2, 8)])
        where = max(0, min(where, size - 6))
        data[where:where + 6] = rnd.choice(MARKERS)
    fh = io.BytesIO(bytes(data))
    expected = scan_reference(fh, FP, size)
    for chunk in (4096, 8192, 65_536):
        assert scan_fast(fh, FP, size, chunk_bytes=chunk) == expected


def test_reader_round_trip_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TANOA_SLD_CACHE_DIR", str(tmp_path / "cache"))
    pytest.importorskip("tanoa")
    from tanoa.testing.conformance import check_reader

    from tanoa_sld._slidebook import SlidebookFile
    from tanoa_sld.reader import SlidebookReader

    path = syn.random_file(tmp_path / "a.sld", 0)  # 2 series, accepted by the port
    reader = SlidebookReader()
    infos = reader.probe(path)
    assert [i.shape_tczyx for i in infos] == [(1, 3, 5, 32, 96), (1, 1, 4, 32, 40)]
    port = SlidebookFile(path)
    for s, info in enumerate(infos):
        t, c, z, y, x = info.shape_tczyx
        for ch in range(c):
            got = np.asarray(reader.open_channel(path, s, ch)[0]).reshape(t, z, y, x)
            port.setSeries(s)
            for zz in range(z):
                raw = port.open_bytes(s, port.getIndex(zz, ch, 0))
                np.testing.assert_array_equal(got[0, zz], np.frombuffer(raw, "<u2").reshape(y, x))
    assert SlidebookReader()._index(path).from_cache  # a fresh reader: loaded, not re-scanned
    assert check_reader(SlidebookReader(), [path]).ok
