from pyebsdindex import nlpar
import time

file = r'D:\EBSD TKD Pattern Processing\FGHAZ 25\map20250109115929899.up1'

print("Initializing NLPAR...")
nlobj = nlpar.NLPAR(file, lam=0.9, searchradius=3)

print("Optimizing lambda...")
t0 = time.time()
nlobj.opt_lambda(chunksize=0, automask=True, autoupdate=True, backsub=False)
print(f"Lambda optimization took {time.time()-t0:.1f}s")

print(f"\nUsing lambda = {nlobj.lam}")
print(f"Search radius = {nlobj.searchradius}")

print("\nRunning NLPAR...")
t0 = time.time()
nlobj.calcnlpar(chunksize=0, saturation_protect=True, automask=True, backsub=False)
print(f"NLPAR took {time.time()-t0:.1f}s")

print("\nDone!")
