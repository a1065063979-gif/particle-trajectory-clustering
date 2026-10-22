# Multivariate particle trajectory clustering

Research code for unsupervised clustering of PIC particle trajectories using six channels: `x`, `px`, `py`, `Ex`, `Ey`, and `Bz` (`pz` is excluded).

## Methods

| Directory | Method | Entry point |
| --- | --- | --- |
| `code/spectrogram_dec` | Spectrogram CNN autoencoder + DEC; automatic K selection | `h5_spectrogram_cnn_dec_k_auto_without_pz_report_weight.py` |
| `code/cnn1d_dec` | Time-series 1D CNN autoencoder + DEC; K=5 | `run_1d_compare.py` |
| `code/pca_kmeans` | Flattened spectrogram + PCA (32 dimensions) + K-means; K=5 | `run_pca_kmeans.py` |

This repository contains the three methods, their existing synthetic-data tests, the report figures in `figures/`, and the saved main-method outputs in `results/spectrogram_dec/`. The combined report and presentation project, including both PDFs and LaTeX sources, is in [`report-and-presentation/`](report-and-presentation/). Raw HDF5 data and trained model checkpoints are not included.

The report figures retain the author-provided filenames and versions. The main results are copied unchanged from `k_auto_spec2/k_auto_spec_report_weight_5e4__350_results`, including CSV tables, NPZ archives, PNG/PDF plots, sampling metadata, and three plotting helper scripts. Preview plots retain their original `PREVIEW` names. Report figures may differ from the corresponding experiment plots.

## Installation

Use a Python environment with the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build appropriate for the computing environment. PCA does not train a neural network, but uses PyTorch CPU interpolation to match the spectrogram preprocessing. The dependency list is not a lockfile of the original training environment.

## Input and execution order

Place the original HDF5 file at `data/particle_data.h5`, or provide its path using the options below. Required datasets are `Times` (time vector), `unique_Ids` (integer particle IDs), and the six channels, each shaped `(time, particle)`. Time records must be increasing and equally spaced. Data are supplied separately.

Run commands from the repository root. First run the spectrogram method, then run the two comparisons against its saved results. The saved main-experiment NPZ files are already provided in `results/spectrogram_dec/`; the comparison scripts still require the original HDF5 data.

### Spectrogram CNN + DEC

```bash
python code/spectrogram_dec/h5_spectrogram_cnn_dec_k_auto_without_pz_report_weight.py
```

The defaults now match the author's specified run: `LAMBDA_CLUSTER=5e-4` and `MAX_DEC_EPOCHS=350`. AE pretraining remains at a maximum of 200 epochs with its original stopping rule. DEC runs 350 epochs and restores the model selected by the original silhouette rule; the selected model is not necessarily epoch 350.

The objective is channel-weighted reconstruction loss plus `5e-4 * KL_sum`, where KL is summed over the current minibatch and clusters. Batch size remains 64. Other model and optimization logic is preserved from the source script.

The main script accepts environment variables. For example, in PowerShell:

```powershell
$env:H5_PATH = 'D:/path/to/particle_data.h5'
$env:LAMBDA_CLUSTER = '5e-4'
$env:MAX_DEC_EPOCHS = '350'
python code/spectrogram_dec/h5_spectrogram_cnn_dec_k_auto_without_pz_report_weight.py
```

`OUTPUT_DIR` overrides the default `results/spectrogram_dec/`. Set `OUTPUT_DIR` to a fresh directory (for example, `results/spectrogram_dec_new_run/`) when retraining, to preserve the archived results: the main script may overwrite files in an existing directory. `PREPROCESS_ONLY=1` generates the preprocessing diagnostics without training; unset it or set it to `0` for a full run.

Fixed-stride sampling uses `TARGET_TIME_POINTS=2048` as an upper bound (6,010 original records become 2,004 records with stride 3). It does not add an anti-aliasing filter. Spectrograms use a 256-point window, hop 128, log magnitude, and resizing to `33 x 60` per channel. Frequency uses `fs=1`, not physical Hz. The latent dimension is 32 and automatic K selection searches 2 through 12.

### 1D CNN + DEC

```bash
python code/cnn1d_dec/run_1d_compare.py --h5 data/particle_data.h5 --results-dir results/spectrogram_dec
```

Requires `h5_spectrogram_cnn_dec_k_auto_without_pz_results.npz` from the main experiment. It reuses particle IDs, sampled times, normalization, and reconstruction channel weights. The reference must have K=5. Defaults include 32 latent dimensions, batch size 64, `--lambda-cluster 5e-4`, and `--dec-epochs 350`.

Outputs go to `code/cnn1d_dec/one_d_results/`; the time-series cache goes to `code/cnn1d_dec/timeseries_cache/`. Override with `--output-dir` and `--cache-dir`. DEC can resume with `--resume path/to/dec_last.pt`, using the original settings and output directory. See `--help` for all options.

### PCA + K-means

```bash
python code/pca_kmeans/run_pca_kmeans.py --h5 data/particle_data.h5 --results-dir results/spectrogram_dec
```

Requires both `h5_spectrogram_cnn_dec_k_auto_without_pz_results.npz` and `ae_initial_clustering_report_weight.npz` from the main experiment. It reproduces the saved spectrogram preprocessing, flattens the input, applies PCA to 32 dimensions, and clusters with K=5. The reference must have K=5.

Outputs go to `code/pca_kmeans/pca_kmeans_results/`; the spectrogram cache goes to `code/pca_kmeans/spectrogram_cache/`. Override with `--output-dir` and `--cache-dir`. See `--help` for all options. Both comparison scripts reject nonempty output directories except for supported 1D resume runs.

## Evaluation limits

The comparison scripts use the same 10,000-particle evaluation sample (seed 42) and report Euclidean silhouette in each method's own representation space. These are internal, method-specific measures, not classification accuracy or independent generalization estimates. Full-data preprocessing and clustering prevent interpreting the validation split as an independent test set. Cluster numbers are identifiers without assigned physical meanings.

## Tests

```bash
python -m unittest discover -s code/spectrogram_dec -p "test_*.py" -v
python -m unittest discover -s code/cnn1d_dec -p "test_*.py" -v
python -m unittest discover -s code/pca_kmeans -p "test_*.py" -v
```

Tests use temporary synthetic data. They cover preprocessing and irregular-time rejection, 1D network dimensions and loss reductions, short training and deterministic resume, particle-ID alignment, and PCA preprocessing/cache consistency. They do not reproduce the full research training run.
