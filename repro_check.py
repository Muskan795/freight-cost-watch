"""Run the pipeline 3 times from scratch and prove the outputs are byte-identical."""
import hashlib
import subprocess
import sys
from pathlib import Path

import config as cfg

hashes = []
for i in (1, 2, 3):
    out = Path(f"repro_run{i}")
    subprocess.run([sys.executable, "run.py", "--out-dir", str(out)], check=True, capture_output=True)
    digest = hashlib.sha256((out / cfg.OUTPUT_FILE).read_bytes()).hexdigest()
    hashes.append(digest)
    print(f"run {i}: sha256 {digest}")

if len(set(hashes)) == 1:
    print("\nPASS: all 3 runs produced byte-identical output.csv")
else:
    print("\nFAIL: outputs differ")
    sys.exit(1)
