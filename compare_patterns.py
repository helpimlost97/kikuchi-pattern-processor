from pyebsdindex import ebsd_pattern
import matplotlib.pyplot as plt

# === CHANGE THESE TO PICK A PATTERN ===
col = 50  # column position in the scan
row = 30  # row position in the scan

# File paths
f_orig = ebsd_pattern.get_pattern_file_obj(r'D:\EBSD TKD Pattern Processing\FGHAZ 25\map20250109115929899.up1')
f_nlpar = ebsd_pattern.get_pattern_file_obj(r'D:\EBSD TKD Pattern Processing\FGHAZ 25\map20250109115929899_NLPAR_l7.67sr3.up1')

# Read single pattern at [col, row]
pat_orig, _ = f_orig.read_data(returnArrayOnly=True, convertToFloat=True, patStartCount=[[col, row], [1, 1]])
pat_nlpar, _ = f_nlpar.read_data(returnArrayOnly=True, convertToFloat=True, patStartCount=[[col, row], [1, 1]])

# Plot side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(pat_orig[0], cmap='gray')
ax1.set_title('Original')
ax2.imshow(pat_nlpar[0], cmap='gray')
ax2.set_title('NLPAR')
plt.tight_layout()
plt.show()
