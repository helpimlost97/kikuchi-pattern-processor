# Draft upstream report for PyEBSDIndex (not posted; Max decides)

Target: https://github.com/USNavalResearchLaboratory/PyEBSDIndex/issues
Affects: `pyebsdindex/opencl/clnlpar.cl`, kernel `calcnlpar`, in 0.3.9.x and `main`
(checked 2026-10-13). The fix is one line; a PR body is included below the issue text.

---

## Issue title

NLPAR OpenCL kernel uses the centre pattern's sigma for every neighbour (`sigma[indx_xy]`
instead of `sigma[indx_ij]`), so GPU results differ from the CPU implementation

## Issue body

**Summary**

In `pyebsdindex/opencl/clnlpar.cl`, kernel `calcnlpar`, the "calculate the weights" loop
looks up the neighbour's sigma with the centre pattern's index:

```c
          for(i=x-sr; i<=x+sr; ++i){
              dd = d[count];
              if (dd > 1.e-3){
                indx_ij =  (i >= 0) ? (i): abs(i);
                indx_ij =  (indx_ij < ncol) ? (indx_ij): ncol - (indx_ij -ncol +1);
                indx_ij = (indx_ij + indx_j);

                sigma_ij = sigma[indx_xy];      // <-- centre index; indx_ij is computed but unused
                sigma_ij *= sigma_ij;
                nn = n[count];
                dd -= nn*(sigma_ij+sigma0);
                norm = (sigma0 + sigma_ij)*sqrt(2.0*nn);
```

`sigma0` is already `sigma[indx_xy]^2`, so the normalisation becomes `2*sigma_i^2`
instead of `sigma_i^2 + sigma_j^2` (Brewick, Wright, Rowenhorst 2019, Eq. 7). The
`normd` kernel in the same file and the numba kernel `nlpar_nb` in `nlpar_cpu.py` both
index the neighbour correctly (`sigma[indx_ij]` / `sigma[j_nn, i_nn]`).

The effect is invisible when the noise level is uniform across the scan (sigma_i ~
sigma_j), which is probably why it went unnoticed, and grows wherever sigma varies: across
grain or phase boundaries with different pattern quality, along contamination or
topography, and at the edge of the recorded area.

**Reproduction**

Synthetic 24 x 16 scan, 48 x 48 uint8 patterns, intensities < 200 (so saturation
protection never triggers in either engine), Gaussian noise with sigma = 6 in the left
half and sigma = 35 in the right half, `searchradius=3`, `lam=0.9`,
`saturation_protect=True`, `automask=False`. Compare `nlpar_cpu.NLPAR.calcnlpar` with
`opencl.nlpar_cl.NLPAR.calcnlpar`, ignoring the `sr`-wide border (the two engines treat
scan borders differently anyway: the numba kernel shifts the window inward, the OpenCL
kernel mirrors indices).

| comparison (interior patterns) | mean abs diff | max abs diff | columns at the noise boundary |
|---|---|---|---|
| CPU vs GPU as shipped | 2.8 | 119 | 14.1 |
| CPU vs GPU with the one-line fix below | 0.000 | 1 | 0.000 |

(grey levels out of 255; the max of 1 is integer rounding.)

**Fix**

```diff
-                sigma_ij = sigma[indx_xy];
+                sigma_ij = sigma[indx_ij];
```

**Environment**

pyebsdindex 0.3.9.1 (pip), pyopencl 2026.1.4, NVIDIA RTX 5070 Laptop (driver 616.64),
Windows 11, Python 3.14. The line is unchanged in `main` as of 2026-10-13.

**Two related observations, not bugs**

- Border handling differs between the two engines (window shifted inward on the CPU,
  mirrored indices on the GPU), so results within `searchradius` of the scan edge never
  agree exactly. Worth a note in the docs if it is intentional.
- The saturation threshold differs: `0.999 * max` in `nlpar_nb`, `0.9961 * max` in
  `calcnlpar_cl` / `calcsigma_cl`. For 8-bit data the CPU drops only 255 and the GPU drops
  254 and 255.

---

## PR title

Fix neighbour sigma lookup in the OpenCL NLPAR kernel

## PR body

`calcnlpar` in `clnlpar.cl` read `sigma[indx_xy]` (the centre pattern) for every
neighbour in the weight loop; `indx_ij` was computed on the line above but not used.
This made the weight normalisation `2*sigma_i^2` instead of `sigma_i^2 + sigma_j^2`, so
GPU results diverged from the numba implementation wherever the noise level varies
across the scan. One-line change to `sigma[indx_ij]`, matching `normd` in the same file
and `nlpar_nb` in `nlpar_cpu.py`.

Verified on a synthetic two-noise-level scan: interior GPU output now equals the CPU
output to within integer rounding (max diff 1 grey level, previously up to 119). Fixes
#<issue number>.
