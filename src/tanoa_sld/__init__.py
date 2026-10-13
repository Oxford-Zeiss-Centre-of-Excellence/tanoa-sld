# tanoa-sld -- 3i SlideBook reader for tanoa. GPL-2.0-or-later; see LICENSE.
"""3i SlideBook (.sld / .spl) for tanoa: a line-by-line Python port of
Bio-Formats' SlidebookReader with a fast, cached index and chunked,
memory-bounded pixel reads. Registered with tanoa through the
`tanoa.readers` entry point (see pyproject.toml)."""

from ._index import cache_dir, load_index
from .reader import SlidebookReader

__all__ = ["SlidebookReader", "cache_dir", "load_index"]
__version__ = "0.1.0"
