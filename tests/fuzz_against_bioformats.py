# tanoa-sld -- GPL-2.0-or-later; see LICENSE.
"""python tests/fuzz_against_bioformats.py OUT_DIR N_SEEDS -- differential fuzz:
every seed's synthetic file through real Bio-Formats and through the port;
prints per-seed agreement and exits non-zero on any disagreement."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import bioformats_oracle as bo  # noqa: E402
import synthetic as syn  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = int(sys.argv[2])
accepted = agree = 0
bad = []
for seed in range(n):
    p = syn.random_file(out / f"s{seed}.sld", seed)
    bf, port = bo.describe_bioformats(p), bo.describe_port(p)
    ok = bf == port
    agree += ok
    accepted += "error" not in bf
    if not ok:
        bad.append(seed)
        print(f"seed {seed}: DISAGREE\n  bf   {json.dumps(bf)[:400]}\n  port {json.dumps(port)[:400]}")
print(f"{n} files: {accepted} accepted by Bio-Formats, {agree}/{n} identical results; disagreements: {bad}")
sys.exit(1 if bad else 0)
