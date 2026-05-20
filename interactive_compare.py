from pyebsdindex import ebsd_pattern
import matplotlib.pyplot as plt
import numpy as np

# File paths
orig_path = r'D:\EBSD TKD Pattern Processing\FGHAZ 25\map20250109115929899.up1'
nlpar_path = r'D:\EBSD TKD Pattern Processing\FGHAZ 25\map20250109115929899_NLPAR_l7.67sr3.up1'

# Open files
print("Loading pattern files...")
f_orig = ebsd_pattern.get_pattern_file_obj(orig_path)
f_nlpar = ebsd_pattern.get_pattern_file_obj(nlpar_path)

nrows = f_orig.nRows
ncols = f_orig.nCols
print(f"Scan dimensions: {ncols} cols x {nrows} rows = {ncols*nrows} patterns")

# Read ALL original patterns to build IQ map
print("Reading all patterns for IQ map (this may take a moment)...")
all_pats, _ = f_orig.read_data(returnArrayOnly=True, convertToFloat=True)
iq = np.std(all_pats, axis=(1, 2)).reshape(nrows, ncols)
del all_pats  # free memory
print("IQ map ready. Click on the map to compare patterns.\n")

# Set up figure: IQ map on left, original pattern top-right, NLPAR bottom-right
fig = plt.figure(figsize=(14, 7))
ax_iq = fig.add_subplot(1, 2, 1)
ax_orig = fig.add_subplot(2, 2, 2)
ax_nlpar = fig.add_subplot(2, 2, 4)

ax_iq.imshow(iq, cmap='gray')
ax_iq.set_title('IQ Map (click a point)')
marker, = ax_iq.plot([], [], 'r+', markersize=15, markeredgewidth=2)

ax_orig.set_title('Original')
ax_nlpar.set_title('NLPAR')

def on_click(event):
    if event.inaxes != ax_iq:
        return
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    if col < 0 or col >= ncols or row < 0 or row >= nrows:
        return

    # Update marker
    marker.set_data([col], [row])

    # Read patterns at clicked position
    pat_o, _ = f_orig.read_data(returnArrayOnly=True, convertToFloat=True,
                                 patStartCount=[[col, row], [1, 1]])
    pat_n, _ = f_nlpar.read_data(returnArrayOnly=True, convertToFloat=True,
                                  patStartCount=[[col, row], [1, 1]])

    ax_orig.clear()
    ax_orig.imshow(pat_o[0], cmap='gray')
    ax_orig.set_title(f'Original [{col}, {row}]')

    ax_nlpar.clear()
    ax_nlpar.imshow(pat_n[0], cmap='gray')
    ax_nlpar.set_title(f'NLPAR [{col}, {row}]')

    fig.canvas.draw_idle()
    print(f"Showing pattern at col={col}, row={row}")

fig.canvas.mpl_connect('button_press_event', on_click)
plt.tight_layout()
plt.show()
