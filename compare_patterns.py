"""Side-by-side comparison of one original vs NLPAR-processed pattern.

Quick visual sanity check. Edit the paths and the [col, row] below to
point at your data.
"""
from pyebsdindex import ebsd_pattern
import matplotlib.pyplot as plt

# ============ EDIT THESE ============
ORIG_PATH  = r'path/to/your/original.up1'
NLPAR_PATH = r'path/to/your/_NLPAR_result.up1'
col = 0   # column position in the scan
row = 0   # row position in the scan
# ====================================

f_orig  = ebsd_pattern.get_pattern_file_obj(ORIG_PATH)
f_nlpar = ebsd_pattern.get_pattern_file_obj(NLPAR_PATH)

pat_orig,  _ = f_orig.read_data( returnArrayOnly=True, convertToFloat=True, patStartCount=[[col, row], [1, 1]])
pat_nlpar, _ = f_nlpar.read_data(returnArrayOnly=True, convertToFloat=True, patStartCount=[[col, row], [1, 1]])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(pat_orig[0],  cmap='gray'); ax1.set_title('Original')
ax2.imshow(pat_nlpar[0], cmap='gray'); ax2.set_title('NLPAR')
plt.tight_layout()
plt.show()
