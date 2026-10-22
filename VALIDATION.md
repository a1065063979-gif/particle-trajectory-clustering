# Local validation

Validated on Windows on 2026-10-22 using Python 3.13.9 and CPU PyTorch in an isolated virtual environment with access to existing scientific packages.

| Suite | Passed |
| --- | --- |
| Spectrogram preprocessing | 3 |
| 1D CNN + DEC | 2 |
| PCA + K-means | 3 |

All eight existing tests passed without skips. Test commands are in README.md. Only temporary synthetic data were used; full training on the original research dataset was not rerun.

Validation package versions (not the original GPU training environment):

```text
numpy==2.3.5
scipy==1.16.3
scikit-learn==1.7.2
h5py==3.15.1
matplotlib==3.10.6
torch==2.14.1+cpu
threadpoolctl==3.5.0
```
