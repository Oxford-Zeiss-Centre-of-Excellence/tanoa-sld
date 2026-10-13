# tanoa-sld -- GPL-2.0-or-later; see LICENSE.
"""tanoa backends run tanoa_sld.inspect FILE.sld [--dump | --exact]

Default: trace the index build live -- every metadata block type, and for
each pixel block whether its end was found by prediction (metadata only,
fast) or needed a full scan. Ctrl-C any time; the printed lines are the
useful part.

--dump: never scans pixel data. Prints the raw metadata blocks before the
first pixel block and the typed blocks in the last 8 MB of the file, as hex,
so the layout can be worked out. Metadata only -- safe to share.
"""
import argparse
import os
import struct
import sys
import time

from ._slidebook import SlidebookFile

_TAIL = 8 * 1024 * 1024


def _hex(data: bytes, base: int, rows: int = 8) -> str:
    out = []
    for r in range(0, min(len(data), rows * 16), 16):
        chunk = data[r:r + 16]
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        out.append(f"      {base + r:>14,}  {chunk.hex(' ')}  {text}")
    return "\n".join(out)


def _type_of(block: bytes, fmt: str) -> "tuple[int, int]":
    for p in range(0, min(len(block), 128) - 1, 2):
        n = struct.unpack_from(fmt + "H", block, p)[0]
        if n:
            return n, p
    return 0, 0


def dump(path: str) -> None:
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(8)
        little = head[4:6] == b"II"
        fmt = "<" if little else ">"
        print(f"file size {size:,} bytes; byte order {'little' if little else 'big'}; first 8 bytes {head.hex(' ')}")
        print("== blocks from the start, up to the first pixel data:")
        pos, shown = 0, 0
        while pos < size - 8 and shown < 64:
            fh.seek(pos)
            block = fh.read(512)
            if block[4:6] in (b"II", b"MM"):
                length = struct.unpack_from(fmt + "h", block, 6)[0]
                n, p = _type_of(block, fmt)
                print(f"  META   @{pos:>14,}  length={length}  type={chr(n) if 0 < n < 128 else n!r} (at +{p})")
                print(_hex(block[:max(128, min(length, 512))], pos, rows=max(8, min(length, 512) // 16)))
                shown += 1
                pos += length if length >= 8 else 128
            elif block[4:6] == b"\xff\xff":
                print(f"  0xFFFF variable-length block @{pos:,} -- stopping the forward walk here")
                print(_hex(block, pos, rows=8))
                break
            else:
                print(f"  PIXELS (or an annotation) start @{pos:,} -- first bytes:")
                print(_hex(block[:64], pos, rows=4))
                break
        # The header may hold 8-byte file offsets (a real SlideBook header
        # did: three values pointing into the file's last ~660 MB). Follow
        # each one and dump what is there -- still metadata only.
        fh.seek(0)
        header = fh.read(4096)
        seen = set()
        print("== 8-byte offsets found in the header, and what they point at:")
        for k in range(8, len(header) - 7):
            value = struct.unpack_from(fmt + "q", header, k)[0]
            if 1_000_000 <= value < size and value not in seen and len(seen) < 12:
                seen.add(value)
                fh.seek(value)
                target = fh.read(1024)
                print(f"  header+{k}: offset {value:,} ({size - value:,} bytes before EOF)")
                print(_hex(target, value, rows=32))
        print(f"== typed blocks in the last {_TAIL // (1024 * 1024)} MB (pixel blocks end where these begin):")
        start = max(0, size - _TAIL)
        fh.seek(start)
        tail = fh.read(_TAIL)
        found = 0
        i = tail.find(b"II" if little else b"MM", 4)
        while i != -1 and found < 40:
            off = i - 4
            if off + 8 <= len(tail):
                length = struct.unpack_from(fmt + "h", tail, off + 6)[0]
                n, p = _type_of(tail[off:off + 128], fmt)
                if length in (128, 256, 512, 1024, 2048) and 0x61 <= n <= 0x7A:
                    print(f"  META   @{start + off:>14,}  length={length}  type={chr(n)!r}")
                    if chr(n) in "ijud" and found < 12:
                        print(_hex(tail[off:off + min(length, 256)], start + off, rows=min(length, 256) // 16))
                    found += 1
            i = tail.find(b"II" if little else b"MM", i + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--exact", action="store_true", help="the exact full scan (slow) instead of prediction")
    ap.add_argument("--dump", action="store_true", help="hex-dump metadata layout; never scans pixel data")
    args = ap.parse_args(argv)
    if args.dump:
        dump(args.path)
        return 0
    t0 = time.perf_counter()
    trace = lambda m: print(f"[{time.perf_counter() - t0:8.2f}s] {m}", flush=True)  # noqa: E731
    try:
        f = SlidebookFile(args.path, predict=not args.exact, trace=trace)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    trace(f"done: {len(f.core)} series")
    for i, ms in enumerate(f.core):
        meta = f.series_metadata[i] if i < len(f.series_metadata) else None
        trace(f"  series {i}: X={ms.sizeX} Y={ms.sizeY} Z={ms.sizeZ} C={ms.sizeC} T={ms.sizeT} "
              f"order={ms.dimensionOrder} name={meta.name if meta else None!r} "
              f"pixel={meta.physical_size_x_um if meta else None} um")
    predicted = sum(1 for _, how in f.scan_log if how == "predicted")
    trace(f"pixel blocks located by prediction: {predicted}/{len(f.scan_log)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
