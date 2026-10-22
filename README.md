# Unsupervised Deep Clustering of Multivariate Particle Trajectories in PIC Simulations

Internship research project by **Linshuo CHEN**, Sorbonne Université, supervised by **Andrea CIARDI**.

This project investigates unsupervised clustering of particle trajectories from particle-in-cell (PIC) simulations. It uses six channels—`x`, `px`, `py`, `Ex`, `Ey`, and `Bz`—and compares a spectrogram CNN autoencoder + deep embedded clustering (DEC) workflow with 1D CNN + DEC and PCA + K-means. The focus is on data representation, clustering methodology, and internal evaluation. `pz` is excluded.

## Report and presentation

- [Internship report (PDF)](report-and-presentation/Rapport%20de%20stage_LisnhuoCHEN.pdf)
- [Defense presentation (PDF)](report-and-presentation/defense-presentation.pdf)
- [Combined LaTeX project and compilation instructions](report-and-presentation/README.md)

The report and presentation share one source directory, preserving the original chapter files, bibliography, images, and relative paths. The entry points are `report.tex` and `defense-presentation.tex`. Supplied PDFs are included as exported; they were not rebuilt during repository preparation.

## Repository contents

| Location | Contents |
| --- | --- |
| [`code/spectrogram_dec/`](code/spectrogram_dec/) | Main spectrogram CNN + DEC method and preprocessing tests |
| [`code/cnn1d_dec/`](code/cnn1d_dec/) | 1D CNN + DEC comparison and tests |
| [`code/pca_kmeans/`](code/pca_kmeans/) | PCA + K-means comparison and tests |
| [`figures/`](figures/) | 13 figures selected for the report |
| [`results/spectrogram_dec/`](results/spectrogram_dec/) | 38 archived main-method output files and plotting helpers |
| [`report-and-presentation/`](report-and-presentation/) | Report and defense PDFs, LaTeX sources, and shared assets |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`VALIDATION.md`](VALIDATION.md) | Local test results and validation environment |

The archived results include CSV tables and training history, initial and final clustering NPZ files, PNG/PDF plots, and sampling metadata. The stored run records a clustering-loss weight of `5e-4`, 350 DEC epochs, and five selected clusters. Report figures retain their final supplied versions; they may differ from the experiment plots. Files marked `PREVIEW` are preserved as preview outputs.

Raw HDF5 input data, trained model checkpoints, and the comparison experiments' output archives are **not included**. Reading the report or inspecting the saved main-method results does not require running the code.

## Methods

| Directory | Method | Entry point |
| --- | --- | --- |
| `code/spectrogram_dec` | Spectrogram CNN autoencoder + DEC; automatic K selection | `h5_spectrogram_cnn_dec_k_auto_without_pz_report_weight.py` |
| `code/cnn1d_dec` | Time-series 1D CNN autoencoder + DEC; K=5 | `run_1d_compare.py` |
| `code/pca_kmeans` | Flattened spectrogram + PCA (32 dimensions) + K-means; K=5 | `run_pca_kmeans.py` |

## Installation

Use a Python environment with the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build appropriate for the computing environment. PCA does not train a neural network, but uses PyTorch CPU interpolation to match the spectrogram preprocessing. The dependency list is not a lockfile of the original training environment.

## Input and execution order

Place the original HDF5 file at `data/particle_data.h5`, or provide its path using the options below. Required datasets are `Times` (time vector), `unique_Ids` (integer particle IDs), and the six channels, each shaped `(time, particle)`. Time records must be increasing and equally spaced. Data are supplied separately.

Run commands from the repository root. The two comparisons can use the archived main-experiment NPZ files already provided in `results/spectrogram_dec/`, together with the original HDF5 data. To create an entirely new experiment, run the spectrogram method first and point both comparisons at its new output directory using `--results-dir`.

### Spectrogram CNN + DEC

The default configuration uses `LAMBDA_CLUSTER=5e-4` and `MAX_DEC_EPOCHS=350`. AE pretraining remains at a maximum of 200 epochs with its original stopping rule. DEC runs 350 epochs and restores the model selected by the original silhouette rule; the selected model is not necessarily epoch 350.

The objective is channel-weighted reconstruction loss plus `5e-4 * KL_sum`, where KL is summed over the current minibatch and clusters. Batch size remains 64. Other model and optimization logic is preserved from the source script.

The main script accepts environment variables. For example, in PowerShell:

```powershell
$env:H5_PATH = 'D:/path/to/particle_data.h5'
$env:OUTPUT_DIR = 'results/spectrogram_dec_new_run'
$env:LAMBDA_CLUSTER = '5e-4'
$env:MAX_DEC_EPOCHS = '350'
$env:PREPROCESS_ONLY = '0'
python code/spectrogram_dec/h5_spectrogram_cnn_dec_k_auto_without_pz_report_weight.py
```

Linux / Bash:

```bash
H5_PATH='/path/to/particle_data.h5' \
OUTPUT_DIR='results/spectrogram_dec_new_run' \
LAMBDA_CLUSTER=5e-4 MAX_DEC_EPOCHS=350 PREPROCESS_ONLY=0 \
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

All eight tests passed without skips during local validation on 2026-10-22. See [VALIDATION.md](VALIDATION.md) for the environment and scope. LaTeX file references were checked, but local document compilation was not performed because a LaTeX compiler was unavailable.
