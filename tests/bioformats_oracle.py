# tanoa-sld -- 3i SlideBook reader for tanoa. GPL-2.0-or-later; see LICENSE.
"""Ground truth from real Bio-Formats: run its `loci.formats.in.SlidebookReader`
class directly (not `ImageReader`, which might pick another reader) through
the JVM that bioio-bioformats/bffile manage, and report what it saw in the
same shape `describe_port` reports the port.

Needs `bffile` (tanoa's `bioformats-dev` pixi environment, or
`tanoa backends install bioio-bioformats`). Import is lazy so the rest of the
test suite runs without a JVM.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def _plane_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:16]


def describe_bioformats(path: Path, *, planes: bool = True) -> "dict[str, Any]":
    import bffile._java_stuff  # noqa: F401 -- configures the Bio-Formats Maven endpoint
    import scyjava

    scyjava.start_jvm()
    scyjava.jimport("loci.common.DebugTools").setRootLevel("ERROR")
    SlidebookReader = scyjava.jimport("loci.formats.in.SlidebookReader")
    MetadataTools = scyjava.jimport("loci.formats.MetadataTools")
    reader = SlidebookReader()
    store = MetadataTools.createOMEXMLMetadata()
    reader.setMetadataStore(store)
    try:
        reader.setId(str(path))
    except Exception as exc:  # noqa: BLE001 -- a refusal is a result to compare
        return {"error": True}  # the exception type differs by language; refusing is what is compared
    try:
        out = {"series": []}
        for s in range(reader.getSeriesCount()):
            reader.setSeries(s)
            entry = {
                "x": int(reader.getSizeX()), "y": int(reader.getSizeY()),
                "z": int(reader.getSizeZ()), "c": int(reader.getSizeC()), "t": int(reader.getSizeT()),
                "order": str(reader.getDimensionOrder()), "count": int(reader.getImageCount()),
                "little": bool(reader.isLittleEndian()),
                "name": _opt_str(store.getImageName(s)),
            }
            px = store.getPixelsPhysicalSizeX(s)
            entry["px_um"] = None if px is None else round(float(px.value()), 6)
            pz = store.getPixelsPhysicalSizeZ(s)
            entry["pz_um"] = None if pz is None else round(float(pz.value()), 6)
            entry["channels"] = [
                _opt_str(store.getChannelName(s, c)) for c in range(int(store.getChannelCount(s)))
            ]
            if planes:
                entry["planes"] = [
                    _plane_digest(bytes(np.asarray(reader.openBytes(no), dtype=np.int8).tobytes()))
                    for no in range(reader.getImageCount())
                ]
            out["series"].append(entry)
        return out
    finally:
        reader.close()


def _opt_str(v: Any) -> "str | None":
    return None if v is None else str(v)


def describe_port(path: Path, *, planes: bool = True, java_int_overflow: bool = True) -> "dict[str, Any]":
    from tanoa_sld._slidebook import SlidebookFile

    try:
        f = SlidebookFile(path, java_int_overflow=java_int_overflow)
    except Exception:  # noqa: BLE001
        return {"error": True}
    try:
        out = {"series": []}
        for s, ms in enumerate(f.core):
            meta = f.series_metadata[s] if s < len(f.series_metadata) else None
            entry = {
                "x": ms.sizeX, "y": ms.sizeY, "z": ms.sizeZ, "c": ms.sizeC, "t": ms.sizeT,
                "order": ms.dimensionOrder, "count": ms.imageCount, "little": ms.littleEndian,
                "name": meta.name if meta else None,
                "px_um": None if not meta or meta.physical_size_x_um is None else round(meta.physical_size_x_um, 6),
                "pz_um": None if not meta or meta.physical_size_z_um is None else round(meta.physical_size_z_um, 6),
                "channels": [meta.channel_names.get(c) if meta else None for c in range(ms.sizeC)],
            }
            if planes:
                entry["planes"] = [_plane_digest(f.open_bytes(s, no)) for no in range(ms.imageCount)]
            out["series"].append(entry)
        return out
    finally:
        f.close()
