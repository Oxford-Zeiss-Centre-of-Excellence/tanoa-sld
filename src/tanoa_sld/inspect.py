# tanoa-sld -- GPL-2.0-or-later; see LICENSE.
"""python -m tanoa_sld.inspect FILE.sld [--exact]

Prints the file's block layout as the index is built -- every metadata block
type, and for each pixel block whether its end was found by prediction
(metadata only, fast) or needed a full scan (reads the whole block). Ctrl-C
at any point; the lines already printed are the useful part. Prints no pixel
data -- safe to share."""
import argparse
import sys
import time

from ._slidebook import SlidebookFile


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--exact", action="store_true", help="the exact full scan (slow) instead of prediction")
    args = ap.parse_args(argv)
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
