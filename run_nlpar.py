"""Headless NLPAR runner via pyebsdindex.

Runs opt_lambda then calcnlpar on a single UP1 file. Output is written next
to the input as <input>_NLPAR_l<lam>sr<sr>.up1 by pyebsdindex.

Edit the path below to point at your data.
"""
from pyebsdindex import nlpar
import time
try:
    import psutil
    # pyebsdindex's _calcchunks assumes 8 B/px (float32 in+out); its default of
    # sysram/4 asks for multi-GB tiles on large scans. Keep float32 tiles <= 2 GB.
    CHUNK_BYTES = int(min(4e9, 0.25 * psutil.virtual_memory().available))
except Exception:
    CHUNK_BYTES = int(2e9)

# ============ EDIT THIS ============
INPUT_PATH = r'path/to/your/original.up1'
SEARCH_RADIUS = 3
INITIAL_LAMBDA = 0.9
# ===================================

print("Initializing NLPAR...")
nlobj = nlpar.NLPAR(INPUT_PATH, lam=INITIAL_LAMBDA, searchradius=SEARCH_RADIUS)

print("Optimizing lambda...")
t0 = time.time()
nlobj.opt_lambda(chunksize=CHUNK_BYTES, automask=True, autoupdate=True, backsub=False)
print(f"Lambda optimization took {time.time() - t0:.1f}s")

print(f"\nUsing lambda = {nlobj.lam}")
print(f"Search radius = {nlobj.searchradius}")

print("\nRunning NLPAR...")
t0 = time.time()
nlobj.calcnlpar(chunksize=CHUNK_BYTES, saturation_protect=True, automask=True, backsub=False)
print(f"NLPAR took {time.time() - t0:.1f}s")

print("\nDone.")
