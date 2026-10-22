"""Six-channel spectrogram -> PCA(32) -> K-means(5), with reusable input cache.

No network training or GPU. PyTorch CPU interpolation is used only to reproduce
the original preprocessing exactly. Existing training results are read-only.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import spectrogram
import sklearn
from sklearn import config_context
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from threadpoolctl import threadpool_limits

CHANNELS = ["x", "px", "py", "Ex", "Ey", "Bz"]
FINAL_NAME = "h5_spectrogram_cnn_dec_k_auto_without_pz_results.npz"
INITIAL_NAME = "ae_initial_clustering_report_weight.npz"
CACHE_VERSION = 1


def csv_out(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def align_ids(reference, other):
    for values in (reference, other):
        if values.ndim != 1 or values.dtype.kind not in "iu" or len(np.unique(values)) != len(values):
            raise ValueError("Particle IDs must be unique integers")
    lookup = {int(value): i for i, value in enumerate(other)}
    if len(reference) != len(other) or any(int(value) not in lookup for value in reference):
        raise ValueError("Particle ID sets differ between input files")
    return np.array([lookup[int(value)] for value in reference])


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def preprocessing_signature(saved):
    digest = hashlib.sha256(f"spectrogram-cache-v{CACHE_VERSION}".encode())
    for key in ("unique_ids", "channel_names", "source_time_indices", "sampled_times",
                "time_mean", "time_std", "spec_mean", "spec_std", "stft_nperseg",
                "stft_hop", "spec_freq_bins", "spec_time_bins"):
        value = np.ascontiguousarray(saved[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_saved(saved):
    if saved["channel_names"].tolist() != CHANNELS:
        raise ValueError(f"Expected channels in this order: {CHANNELS}")
    align_ids(saved["unique_ids"], saved["unique_ids"])
    for key in ("time_mean", "time_std", "spec_mean", "spec_std"):
        values = saved[key]
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key}")
        if key.endswith("std") and np.any(values <= 0):
            raise ValueError(f"Nonpositive {key}")
    idx = saved["source_time_indices"]
    if (idx.ndim != 1 or idx.dtype.kind not in "iu" or len(idx) < 2
            or idx[0] < 0 or np.any(np.diff(idx.astype(np.int64)) <= 0)):
        raise ValueError("Invalid source time indices")
    if not np.all(np.diff(idx) == idx[1] - idx[0]):
        raise ValueError("Expected fixed-stride time sampling")
    if saved["sampled_times"].shape != idx.shape:
        raise ValueError("Sampled times and indices have different lengths")
    nperseg, hop = int(saved["stft_nperseg"].item()), int(saved["stft_hop"].item())
    if not 1 <= hop <= nperseg <= len(idx):
        raise ValueError("Invalid spectrogram window/hop")
    if min(int(saved["spec_freq_bins"].item()), int(saved["spec_time_bins"].item())) < 1:
        raise ValueError("Invalid spectrogram target shape")
    if int(saved["selected_k"].item()) != 5:
        raise ValueError("Reference DEC run must have K=5 for this experiment")


def log_spectrogram(values, nperseg, hop, target_hw):
    # Keep the original implementation, including CPU torch bilinear interpolation.
    import torch
    import torch.nn.functional as functional
    _, _, magnitude = spectrogram(
        values, fs=1.0, window=("tukey", .25), nperseg=nperseg,
        noverlap=nperseg-hop, detrend="constant", return_onesided=True,
        scaling="density", mode="magnitude", axis=-1)
    logged = np.log1p(magnitude).astype(np.float32)
    return functional.interpolate(torch.from_numpy(logged).unsqueeze(1),
                                  size=target_hw, mode="bilinear",
                                  align_corners=False).squeeze(1).numpy()


def get_spectrogram_cache(h5_path, cache_dir, saved, chunk_size, threads):
    signature = preprocessing_signature(saved)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = cache_dir / "standardized_spectrograms.npy", cache_dir / "cache_metadata.json"
    shape = (len(saved["unique_ids"]), 6, int(saved["spec_freq_bins"].item()),
             int(saved["spec_time_bins"].item()))
    if meta_path.exists():
        info = json.loads(meta_path.read_text(encoding="utf-8"))
        if info.get("signature") != signature or not data_path.is_file():
            raise ValueError("Cache does not match this run or is incomplete; choose a new --cache-dir")
        cached = np.load(data_path, mmap_mode="r", allow_pickle=False)
        if cached.shape != shape or cached.dtype != np.float32:
            raise ValueError("Invalid cached tensor shape/dtype")
        print(f"Reusing cache: {data_path}\n  originally built from: {info['h5']}", flush=True)
        return cached
    if any(cache_dir.iterdir()):
        raise ValueError("Cache directory is nonempty without completed metadata; choose a new --cache-dir")
    if not h5_path.is_file():
        raise FileNotFoundError(f"Original H5 required to create cache: {h5_path}")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("First cache build needs PyTorch for the same CPU bilinear resize as training. Use your particle environment.") from exc
    torch.set_num_threads(threads)
    idx = saved["source_time_indices"]
    selector = slice(int(idx[0]), int(idx[-1]) + 1, int(idx[1] - idx[0]))
    nperseg, hop = int(saved["stft_nperseg"].item()), int(saved["stft_hop"].item())
    start_time = time.perf_counter()
    with h5py.File(h5_path, "r") as f:
        h5_ids = f["unique_Ids"][:]
        # Convert H5 column order into the reference final archive order.
        destinations = align_ids(h5_ids, saved["unique_ids"])
        times = f["Times"][:]
        if idx[-1] >= len(times) or not np.array_equal(times[idx], saved["sampled_times"]):
            raise ValueError("H5 sampled times differ from reference run")
        if not np.allclose(np.diff(times[idx]), np.diff(times[idx])[0]):
            raise ValueError("Time sampling is not uniform")
        for channel in CHANNELS:
            if f[channel].shape != (len(times), len(h5_ids)):
                raise ValueError(f"Unexpected H5 dimensions: {channel}")
        data = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float32, shape=shape)
        try:
            for c, channel in enumerate(CHANNELS):
                print(f"Preprocessing {channel} ({c+1}/6)", flush=True)
                for start in range(0, len(h5_ids), chunk_size):
                    stop = min(start + chunk_size, len(h5_ids))
                    raw = f[channel][selector, start:stop].astype(np.float32)
                    values = ((raw - saved["time_mean"][c]) / saved["time_std"][c]).T.astype(np.float32)
                    spec = log_spectrogram(values, nperseg, hop, shape[2:])
                    spec -= saved["spec_mean"][c]
                    spec /= saved["spec_std"][c]
                    if not np.isfinite(spec).all():
                        raise ValueError(f"Nonfinite preprocessed values in {channel}")
                    data[destinations[start:stop], c] = spec
                print(f"  {len(h5_ids)} particles completed", flush=True)
            data.flush()
        finally:
            data._mmap.close()
    np.savez(cache_dir / "preprocessing_parameters.npz", **{
        key: saved[key] for key in ("unique_ids", "channel_names", "source_time_indices", "sampled_times",
                                   "time_mean", "time_std", "spec_mean", "spec_std",
                                   "stft_nperseg", "stft_hop", "spec_freq_bins", "spec_time_bins")})
    info = {"signature": signature, "cache_version": CACHE_VERSION, "shape": shape,
            "h5": str(h5_path.resolve()), "h5_size_bytes": h5_path.stat().st_size,
            "h5_mtime_ns": h5_path.stat().st_mtime_ns,
            "layout": "particle, channel, frequency, time; final NPZ particle ID order",
            "normalization": "saved training time and spectrogram means/stds; no refitting",
            "window": "tukey(0.25)", "fs": 1.0, "detrend": "constant",
            "mode": "magnitude", "scaling": "density", "log": "log1p",
            "resize": "torch CPU bilinear, align_corners=False, antialias=False",
            "build_seconds": time.perf_counter()-start_time,
            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__}
    meta_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return np.load(data_path, mmap_mode="r", allow_pickle=False)


def fit_baseline(data, seed, n_init):
    flat = data.reshape(len(data), -1)
    if min(flat.shape) < 32:
        raise ValueError("PCA(32) requires at least 32 particles and 32 input features")
    print(f"PCA: {flat.shape} -> ({len(data)}, 32)", flush=True)
    start = time.perf_counter()
    pca = PCA(n_components=32, whiten=False, svd_solver="randomized", random_state=seed,
              iterated_power=7, n_oversamples=10, copy=True)
    vectors = pca.fit_transform(flat)
    pca_seconds = time.perf_counter()-start
    print(f"PCA retained variance: {pca.explained_variance_ratio_.sum():.6f}", flush=True)
    print(f"K-means: K=5, n_init={n_init}", flush=True)
    start = time.perf_counter()
    kmeans = KMeans(n_clusters=5, n_init=n_init, max_iter=1000, random_state=seed, algorithm="lloyd")
    labels = kmeans.fit_predict(vectors)
    if len(np.unique(labels)) != 5:
        raise ValueError("K-means returned fewer than 5 nonempty clusters")
    return pca, kmeans, vectors, labels, {"pca_seconds": pca_seconds, "kmeans_seconds": time.perf_counter()-start}


def evaluate_methods(methods, sample, working_memory):
    rows, scores = [], {}
    with config_context(working_memory=working_memory):
        for method, (space, vectors, labels) in methods.items():
            if vectors.ndim != 2 or vectors.shape[1] != 32 or len(vectors) != len(labels):
                raise ValueError(f"Invalid 32-dimensional representation for {method}")
            if not np.isfinite(vectors).all():
                raise ValueError(f"Nonfinite features for {method}")
            if len(np.unique(labels[sample])) != 5:
                raise ValueError(f"Evaluation sample omits clusters for {method}; increase --sample-size")
            print(f"Silhouette: {method}, {len(sample)} identical sampled particles", flush=True)
            values = silhouette_samples(vectors[sample], labels[sample], metric="euclidean")
            score = float(values.mean())
            scores[method] = values
            rows.append([method, space, 32, 5, len(sample), score])
            print(f"  {score:.8f}", flush=True)
    return rows, scores


def plots(output, pca, vectors, labels, rows):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, 33), np.cumsum(pca.explained_variance_ratio_), marker=".")
    ax.set(xlabel="Number of principal components", ylabel="Cumulative explained variance", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(output / "pca_explained_variance.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(5):
        keep = labels == k
        ax.scatter(vectors[keep, 0], vectors[keep, 1], s=4, alpha=.45, label=f"C{k} (n={keep.sum()})", rasterized=True)
    ax.set(xlabel="PC1", ylabel="PC2", title="PCA + K-means (2D display; clustering uses 32 PCs)")
    ax.legend(markerscale=3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"pca_clusters.{ext}", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [row[-1] for row in rows]
    bars = ax.bar([f"{row[0]}\n({row[1]})" for row in rows], values, color=["#638bb8", "#50a296", "#d78948"])
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set(ylabel="Silhouette score", ylim=(min(0, min(values)-.1), max(.1, max(values)+.15)),
           title="Same particle sample; each method's own 32D representation")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"silhouette_comparison.{ext}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--results-dir", type=Path, default=here.parents[1] / "results" / "spectrogram_dec")
    parser.add_argument("--h5", type=Path, default=here.parents[1] / "data" / "particle_data.h5")
    parser.add_argument("--output-dir", type=Path, default=here / "pca_kmeans_results")
    parser.add_argument("--cache-dir", type=Path, default=here / "spectrogram_cache")
    parser.add_argument("--sample-size", type=int, default=10000, help="0 means all particles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--working-memory", type=int, default=256, help="Pairwise distance block budget in MiB, not total RAM")
    args = parser.parse_args()
    if args.sample_size < 0 or min(args.n_init, args.chunk_size, args.threads, args.working_memory) < 1:
        raise ValueError("Invalid sample size/resource parameter")
    final = load_npz(args.results_dir / FINAL_NAME)
    initial = load_npz(args.results_dir / INITIAL_NAME)
    validate_saved(final)
    ids = final["unique_ids"]
    order = align_ids(ids, initial["unique_ids"])
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n = len(ids) if args.sample_size == 0 else min(args.sample_size, len(ids))
    if n <= 5:
        raise ValueError("Silhouette sample must have more than five particles")
    sample = np.random.RandomState(args.seed).permutation(len(ids))[:n]
    with threadpool_limits(limits=args.threads):
        data = get_spectrogram_cache(args.h5, args.cache_dir, final, args.chunk_size, args.threads)
        try:
            pca, km, vectors, labels, timing = fit_baseline(data, args.seed, args.n_init)
        finally:
            data._mmap.close()
        methods = {"PCA_KMeans": ("PCA", vectors, labels),
                   "AE_KMeans": ("AE latent", initial["latent_vectors"][order], initial["kmeans_labels"][order]),
                   "DEC": ("DEC latent", final["latent_vectors"], final["labels"])}
        rows, silhouettes = evaluate_methods(methods, sample, args.working_memory)
    out = args.output_dir
    csv_out(out / "silhouette_comparison.csv", ["method", "evaluation_space", "dimensions", "k", "sample_count", "silhouette"], rows)
    csv_out(out / "pca_cluster_counts.csv", ["cluster", "count"], enumerate(np.bincount(labels, minlength=5).tolist()))
    csv_out(out / "pca_explained_variance.csv", ["component", "explained_variance_ratio", "cumulative_ratio"],
            zip(range(1, 33), pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)))
    csv_out(out / "particle_labels.csv", ["particle_id", "pca_kmeans_label"], ((int(pid), int(y)) for pid, y in zip(ids, labels)))
    csv_out(out / "evaluation_sample.csv", ["sample_order", "particle_id", "final_archive_row"],
            ((i, int(ids[j]), int(j)) for i, j in enumerate(sample)))
    np.savez_compressed(out / "pca_kmeans_results.npz", unique_ids=ids, labels=labels, latent_vectors=vectors,
                        pca_components=pca.components_, pca_mean=pca.mean_,
                        explained_variance_ratio=pca.explained_variance_ratio_, cluster_centers=km.cluster_centers_,
                        selected_k=np.array([5]), seed=np.array([args.seed]), sample_indices=sample)
    np.savez_compressed(out / "sample_silhouettes.npz", unique_ids=ids[sample], **silhouettes)
    plots(out, pca, vectors, labels, rows)
    metadata = {"status": "complete", "results_dir": str(args.results_dir.resolve()),
                "cache_dir": str(args.cache_dir.resolve()), "preprocessing_signature": preprocessing_signature(final),
                "particle_count": len(ids), "sample_count": n, "seed": args.seed, "channels": CHANNELS,
                "pca": {"components": 32, "solver": "randomized", "iterated_power": 7, "n_oversamples": 10,
                        "whiten": False, "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())},
                "kmeans": {"k": 5, "n_init": args.n_init, "max_iter": 1000, "algorithm": "lloyd",
                           "inertia": float(km.inertia_), "iterations": int(km.n_iter_)},
                "threads": args.threads, "distance_chunk_memory_mib": args.working_memory, **timing,
                "silhouette_protocol": "Euclidean silhouette within identical RandomState(seed).permutation(N)[:n] sample; each method uses its own 32D space.",
                "scope": "Internal evaluation on data used to fit the representations; not held-out accuracy or common-space evaluation.",
                "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Status: complete\nResults: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
