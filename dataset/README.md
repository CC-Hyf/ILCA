# O1_28 UAV parallel-trajectory dataset

This directory contains 12,000 paired UAV/AUAV CSI samples. The published split
uses 10,800 training pairs and 1,200 test pairs with random seed 2026. Each
normalized CSI tensor has shape `(N, 2, 16, 32)` and dtype `float32`; the two
channels contain real and imaginary components.

## Files

- `train_uav*.npy`, `train_auav*.npy`: paired training CSI.
- `test_uav*.npy`, `test_auav*.npy`: paired test CSI.
- `deviating_node_channels*.npy`: five deviating-node channel sets.
- `deviating_node_subset_channels.npy`: selected deviating-node subset.
- `deviating_node_subset_indices.npy`: indices of the selected subset.
- `raw_complex_channels.npz`: unnormalized complex channel arrays.
- `norm_params.npz`: normalization extrema.
- `split_indices.npz`: split indices and deviating-node offsets.
- `gen_report.json`: geometry, split, normalization, and summary statistics.

The suffixes `_snr5db`, `_snr10db`, `_snr15db`, and `_snr20db` identify noisy
variants at the corresponding SNR. Files without an SNR suffix contain the base
normalized arrays.

## Loading example

```python
import numpy as np

uav = np.load("train_uav_snr20db.npy", mmap_mode="r")
auav = np.load("train_auav_snr20db.npy", mmap_mode="r")
print(uav.shape, auav.shape)  # (10800, 2, 16, 32)
```
