"""Plot final-record momenta, aligning saved DEC labels by particle ID."""
import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=Path("D:/internship/data/particle_data.h5"))
    args = parser.parse_args()
    with np.load(folder / "h5_spectrogram_cnn_dec_k_auto_without_pz_results.npz") as result:
        ids = result["unique_ids"].copy()
        labels = result["labels"].copy()
    with h5py.File(args.h5, "r") as source:
        source_ids = source["unique_Ids"][:]
        times = source["Times"][:]
        if np.any(np.diff(times) <= 0):
            raise ValueError("Recording times must be strictly increasing.")
        time = times[-1]
        for key in ("px", "py"):
            if source[key].shape != (len(times), len(source_ids)):
                raise ValueError(f"Unexpected {key} dimensions.")
        if len(np.unique(ids)) != len(ids) or len(np.unique(source_ids)) != len(source_ids):
            raise ValueError("Particle IDs must be unique.")
        if not np.array_equal(np.sort(ids), np.sort(source_ids)):
            raise ValueError("Raw data and clustering must contain exactly the same particles.")
        lookup = {particle_id: index for index, particle_id in enumerate(source_ids.tolist())}
        indices = np.array([lookup[particle_id] for particle_id in ids.tolist()])
        px = source["px"][-1, :][indices]
        py = source["py"][-1, :][indices]
    if labels.shape != ids.shape or not (np.isfinite(px).all() and np.isfinite(py).all()):
        raise ValueError("Labels or momentum values are invalid.")

    clusters, counts = np.unique(labels, return_counts=True)
    # Match the existing report's tab10 colormap normalized over the cluster IDs.
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(clusters)))
    point_colors = colors[np.searchsorted(clusters, labels)]
    order = np.random.default_rng(42).permutation(len(ids))
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 16, "axes.labelsize": 16})
    fig, ax = plt.subplots(figsize=(8.6, 7.5), layout="constrained")
    ax.scatter(px[order], py[order], c=point_colors[order], s=9, alpha=0.65,
               linewidths=0, rasterized=True)
    ax.set_xlabel(r"$p_x$")
    ax.set_ylabel(r"$p_y$")
    ax.set_title(f"Particle momenta at the final recorded time\n" + rf"$t = {time:,.0f}$", pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                      markersize=6, label=f"C{int(cluster)} (n = {count:,})")
               for cluster, count, color in zip(clusters, counts, colors)]
    ax.legend(handles=handles, title="DEC cluster", loc="upper right", framealpha=0.95)
    stem = folder / "12_final_px_py_by_cluster_spectrogram_k_auto_without_pz"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["particle_id", "cluster", "time", "px", "py"])
        writer.writerows(zip(ids, labels, np.full(len(ids), time), px, py))
    print(f"Plotted all {len(ids)} particles at t={time}; aligned by unique ID.")
    print(f"Saved PNG, PDF and data CSV: {stem}")


if __name__ == "__main__":
    main()
