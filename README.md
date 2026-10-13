# tanoa-sld

3i SlideBook (`.sld`, `.spl`) reader for [tanoa](https://github.com/Oxford-Zeiss-Centre-of-Excellence/Tanoa),
built for files far larger than RAM on slow storage (the motivating case: a
388 GB `.sld` on a Windows network share).

- **A line-by-line Python port of Bio-Formats' `SlidebookReader.java`.** It
  reaches the same layout decisions, with one deliberate fix: Bio-Formats
  computes plane offsets with 32-bit `int` arithmetic, which wraps for any
  series over 2 GiB and silently reads the wrong plane. This port uses 64-bit
  offsets (`java_int_overflow=True` restores the Java behaviour for testing).
- **Metadata-only open.** SlideBook keeps no index. Bio-Formats finds each
  pixel block's end by reading the whole block (hours for a 388 GB file);
  this reader predicts the end from the block geometry and checks one small
  window there. It falls back to an exact scan (32 MiB prefetched reads) only
  where the prediction does not confirm. `TANOA_SLD_EXACT_INDEX=1` forces
  the exact scan. `python -m tanoa_sld.inspect FILE.sld` shows, block by
  block, which path was taken.
- **Instant later opens.** The parsed index is cached locally, outside the
  data folder (override with `TANOA_SLD_CACHE_DIR`).
- **Memory-bounded streaming to OME-Zarr.** Channels are lazy dask arrays of
  ~16 MiB row bands. Each band is one positioned read, so reads can run in
  parallel over the network. tanoa's writer streams them under its
  `max_block_bytes` budget. The preview reads only a strided plane.
- In tanoa's GUI the first scan runs as `prepare()`, with a progress bar and
  Cancel.

## Install

From the app's Backends panel ("SlideBook (tanoa-sld)"), or:

```bash
pixi run tanoa backends install tanoa-sld
```

## Status

Tested against synthetic SlideBook files: the fast scanner matches the literal
Java loop, chunked reads match whole-plane reads, it passes tanoa's
conformance kit, and it converts to OME-Zarr. Not yet tested against real
SlideBook acquisitions. Please compare a file against Fiji/Bio-Formats and
report any difference: `tests/bioformats_oracle.py` runs both readers on the
same file.

## Licence

GPL-2.0-or-later. This is a derivative work of Bio-Formats' GPL-licensed
SlidebookReader (see `src/tanoa_sld/_slidebook.py`'s header). It is a
separate, optional package from tanoa (Apache-2.0).
