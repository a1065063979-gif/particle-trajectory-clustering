"""Six-channel 1D time-series AE + DEC reference experiment (K=5, z=32).

Uses the final spectrogram run's particle IDs, sampling and time normalization.
No spectrograms are used as input. The only cross-method metric is silhouette
in each method's own 32-dimensional representation on identical particle IDs.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import config_context
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

CHANNELS = ["x", "px", "py", "Ex", "Ey", "Bz"]
REFERENCE_NAME = "h5_spectrogram_cnn_dec_k_auto_without_pz_results.npz"


def csv_write(path, header, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def npz_read(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def id_order(reference, other):
    for ids in (reference, other):
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
            raise ValueError("Particle IDs must be unique integers")
    lookup = {int(value): i for i, value in enumerate(other)}
    if len(reference) != len(other) or any(int(value) not in lookup for value in reference):
        raise ValueError("Particle ID sets differ")
    return np.array([lookup[int(value)] for value in reference], dtype=np.int64)


def reference_signature(saved):
    digest = hashlib.sha256(b"1d-time-cache-v1")
    for key in ("unique_ids", "source_time_indices", "sampled_times", "channel_names", "time_mean", "time_std"):
        array = np.ascontiguousarray(saved[key])
        digest.update(key.encode() + str(array.dtype).encode() + str(array.shape).encode() + array.tobytes())
    return digest.hexdigest()


def validate_reference(saved):
    if saved["channel_names"].tolist() != CHANNELS:
        raise ValueError(f"Expected channels: {CHANNELS}")
    id_order(saved["unique_ids"], saved["unique_ids"])
    if int(saved["selected_k"].item()) != 5:
        raise ValueError("The reference run must have K=5")
    for key in ("time_mean", "time_std"):
        if saved[key].shape != (6,) or not np.isfinite(saved[key]).all():
            raise ValueError(f"Invalid {key}")
    if np.any(saved["time_std"] <= 0):
        raise ValueError("Standard deviations must be positive")
    indices = saved["source_time_indices"]
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < 16
            or indices[0] < 0 or np.any(np.diff(indices.astype(np.int64)) <= 0)
            or not np.all(np.diff(indices) == indices[1] - indices[0])):
        raise ValueError("Expected at least 16 time points with fixed increasing stride")
    if saved["sampled_times"].shape != indices.shape:
        raise ValueError("Saved times and indices do not match")
    z, y = saved["latent_vectors"], saved["labels"]
    if z.shape != (len(saved["unique_ids"]), 32) or y.shape != (len(z),) or not np.isfinite(z).all():
        raise ValueError("Invalid reference features/labels")


def time_cache(h5_path, cache_dir, saved, chunk=256):
    signature = reference_signature(saved)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path, metadata = cache_dir / "standardized_timeseries.npy", cache_dir / "cache_metadata.json"
    shape = (len(saved["unique_ids"]), 6, len(saved["source_time_indices"]))
    if metadata.exists():
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info.get("signature") != signature or not path.exists():
            raise ValueError("Cache mismatch or incomplete cache; choose a new --cache-dir")
        array = np.load(path, mmap_mode="c", allow_pickle=False)
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError("Invalid cached time-series shape/dtype")
        print(f"Reusing time-series cache: {path}", flush=True)
        return array
    if any(cache_dir.iterdir()):
        raise ValueError("Incomplete/nonempty cache directory; choose a new --cache-dir")
    indices = saved["source_time_indices"]
    selector = slice(int(indices[0]), int(indices[-1])+1, int(indices[1]-indices[0]))
    with h5py.File(h5_path, "r") as f:
        h5_ids, times = f["unique_Ids"][:], f["Times"][:]
        destinations = id_order(h5_ids, saved["unique_ids"])
        if indices[-1] >= len(times) or not np.array_equal(times[indices], saved["sampled_times"]):
            raise ValueError("H5 times differ from the reference run")
        if not np.allclose(np.diff(times[indices]), np.diff(times[indices])[0]):
            raise ValueError("Sampled time intervals are not uniform")
        for channel in CHANNELS:
            if f[channel].shape != (len(times), len(h5_ids)):
                raise ValueError(f"Invalid H5 dimensions: {channel}")
        data = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        try:
            for c, channel in enumerate(CHANNELS):
                for start in range(0, len(h5_ids), chunk):
                    stop = min(start+chunk, len(h5_ids))
                    raw = f[channel][selector, start:stop].astype(np.float32)
                    values = ((raw-saved["time_mean"][c])/saved["time_std"][c]).T
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite standardized values: {channel}")
                    data[destinations[start:stop], c] = values
                print(f"Loaded standardized channel {channel}: {shape[0]} particles", flush=True)
            data.flush()
        finally:
            data._mmap.close()
    np.savez(cache_dir / "preprocessing_parameters.npz", **{key: saved[key] for key in
             ("unique_ids", "channel_names", "source_time_indices", "sampled_times", "time_mean", "time_std")})
    metadata.write_text(json.dumps({"signature": signature, "shape": shape, "h5": str(h5_path.resolve()),
        "layout": "particle,channel,time; reference final NPZ particle order",
        "normalization": "saved full-training-data time_mean and time_std, no refitting"}, indent=2), encoding="utf-8")
    return np.load(path, mmap_mode="c", allow_pickle=False)


class TimeDataset(Dataset):
    def __init__(self, array):
        self.tensor = torch.from_numpy(array)

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return idx, self.tensor[idx]


class Conv1DAutoencoder(nn.Module):
    """Direct 1D counterpart of the current 8/16/32/64-channel 2D network."""
    def __init__(self, length, channels=6, latent_dim=32):
        super().__init__()
        self.length = length
        widths = [channels, 8, 16, 32, 64]
        layers = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers.extend([nn.Conv1d(a, b, 3, stride=2, padding=1), nn.ReLU()])
        self.encoder_conv = nn.Sequential(*layers)
        with torch.no_grad():
            self.encoded_shape = tuple(self.encoder_conv(torch.zeros(1, channels, length)).shape[1:])
        flat = int(np.prod(self.encoded_shape))
        self.fc_enc = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)
        layers = []
        widths = [64, 32, 16, 8, channels]
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.ConvTranspose1d(a, b, 3, stride=2, padding=1, output_padding=int(i == 3)))
            if i < 3:
                layers.append(nn.ReLU())
        self.decoder_conv = nn.Sequential(*layers)

    def encode(self, x):
        return self.fc_enc(self.encoder_conv(x).flatten(1))

    def decode(self, z):
        x = self.decoder_conv(self.fc_dec(z).view(len(z), *self.encoded_shape))
        return F.interpolate(x, size=self.length, mode="linear", align_corners=False) if x.shape[-1] != self.length else x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class WeightedMSE(nn.Module):
    def __init__(self, weights):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", weights.view(1, -1, 1))
        self.register_buffer("normalizer", weights.mean())

    def forward(self, recon, target):
        return ((recon-target).square()*self.weights).mean()/self.normalizer


def target_distribution(q):
    weight = q.square()/q.sum(dim=0, keepdim=True).clamp_min(1e-10)
    return weight/weight.sum(dim=1, keepdim=True).clamp_min(1e-10)


def kl_sum(p, q):
    p, q = p.clamp_min(1e-10), q.clamp_min(1e-10)
    return torch.sum(p*torch.log(p/q))


class DECModel(nn.Module):
    def __init__(self, ae, centers):
        super().__init__()
        self.autoencoder = ae
        self.cluster_centers = nn.Parameter(torch.as_tensor(centers, dtype=torch.float32).clone())

    def forward(self, x):
        recon, z = self.autoencoder(x)
        q = (1+(z[:, None, :]-self.cluster_centers[None, :, :]).square().sum(dim=2)).reciprocal()
        q = q/q.sum(dim=1, keepdim=True)
        return recon, z, q


def state_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def extract(model, loader, device, dec=False):
    model.eval()
    z_all = np.empty((len(loader.dataset), 32), dtype=np.float32)
    q_all = torch.empty((len(loader.dataset), 5)) if dec else None
    for indices, x in loader:
        if dec:
            _, z, q = model(x.to(device))
            q_all[indices] = q.cpu()
        else:
            z = model.encode(x.to(device))
        z_all[indices.numpy()] = z.cpu().numpy()
    if not np.isfinite(z_all).all() or (dec and not torch.isfinite(q_all).all()):
        raise ValueError("Nonfinite extracted features/assignments")
    return (z_all, q_all, q_all.argmax(1).numpy()) if dec else z_all


def sampled_score(z, labels, size, seed):
    sample = np.random.RandomState(seed).permutation(len(z))[:min(size, len(z))]
    n_clusters = len(np.unique(labels[sample]))
    if not 2 <= n_clusters < len(sample):
        return float("nan")
    return float(silhouette_score(z[sample], labels[sample], metric="euclidean"))


def ae_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for _, batch in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite AE loss")
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()*len(batch)
            count += len(batch)
    return total/count


def dec_epoch(model, loader, criterion, optimizer, p, device, weight):
    model.train()
    totals, count = np.zeros(3), 0
    for indices, batch in loader:
        batch = batch.to(device)
        recon, _, q = model(batch)
        rec, cluster = criterion(recon, batch), kl_sum(p[indices].to(device), q)
        loss = rec + weight*cluster
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite DEC loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals += np.array([loss.item(), rec.item(), cluster.item()])*len(batch)
        count += len(batch)
    return totals/count


def choose_device(requested):
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)
    try:
        probe = torch.ones(8, device=device).square().sum()
        if device.type == "cuda":
            torch.cuda.synchronize()
        probe.item()
    except Exception as exc:
        raise RuntimeError("Selected GPU/PyTorch combination cannot execute. Use a compatible node (as for the 2D run), or --device cpu.") from exc
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""), flush=True)
    return device


def diagnostic_plots(out, ae_history, history, model, dataset, device, ids, labels, z, comparison):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ae_history[:, 0], ae_history[:, 1], label="train")
    axes[0].plot(ae_history[:, 0], ae_history[:, 2], label="validation")
    axes[0].set(xlabel="AE epoch", ylabel="Weighted reconstruction MSE")
    axes[0].legend()
    axes[1].plot(history[:, 0], history[:, 5])
    axes[1].set(xlabel="DEC epoch", ylabel="1D latent silhouette (monitor sample)")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(6, 1, figsize=(10, 11), sharex=True)
    _, x = dataset[0]
    model.eval()
    with torch.no_grad():
        recon, _ = model(x.unsqueeze(0).to(device))
    for c, name in enumerate(CHANNELS):
        axes[c].plot(x[c].numpy(), label="input", linewidth=.8)
        axes[c].plot(recon[0, c].cpu().numpy(), label="reconstruction", linewidth=.8, alpha=.8)
        axes[c].set_ylabel(name)
    axes[0].legend()
    axes[0].set_title(f"Selected final 1D model reconstruction; particle ID {int(ids[0])}")
    axes[-1].set_xlabel("Uniform sample index")
    fig.tight_layout()
    fig.savefig(out / "final_reconstruction.png", dpi=180)
    plt.close(fig)
    coords = PCA(n_components=2, random_state=42).fit_transform(z)
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in range(5):
        keep = labels == k
        ax.scatter(coords[keep, 0], coords[keep, 1], s=3, alpha=.5, label=f"C{k} n={keep.sum()}", rasterized=True)
    ax.set(xlabel="Latent PC1", ylabel="Latent PC2", title="1D DEC: 2D display of the 32D representation")
    ax.legend(markerscale=3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"final_latent_pca.{ext}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([r[0] for r in comparison], [r[-1] for r in comparison])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set(ylabel="Silhouette in each method's own 32D space", ylim=(-.1, 1.05))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"silhouette_comparison.{ext}", dpi=180)
    plt.close(fig)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=here.parents[1] / "results" / "spectrogram_dec")
    parser.add_argument("--h5", type=Path, default=here.parents[1] / "data" / "particle_data.h5")
    parser.add_argument("--output-dir", type=Path, default=here / "one_d_results")
    parser.add_argument("--cache-dir", type=Path, default=here / "timeseries_cache")
    parser.add_argument("--resume", type=Path, help="Resume DEC from this experiment's dec_last.pt; use the same output directory/settings")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ae-epochs", type=int, default=200)
    parser.add_argument("--ae-patience", type=int, default=15)
    parser.add_argument("--dec-epochs", type=int, default=350)
    parser.add_argument("--lambda-cluster", type=float, default=5e-4)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--dec-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--n-init", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--monitor-sample-size", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--working-memory", type=int, default=256)
    return parser.parse_args()


def run(args):
    start_time = time.perf_counter()
    if min(args.batch_size, args.ae_epochs, args.ae_patience, args.dec_epochs, args.n_init, args.threads, args.working_memory) < 1:
        raise ValueError("Epoch counts, batch size and resource parameters must be positive")
    if min(args.sample_size, args.monitor_sample_size) <= 5:
        raise ValueError("Silhouette sample sizes must exceed five")
    if not all(np.isfinite(x) for x in [args.lambda_cluster, args.ae_lr, args.dec_lr, args.weight_decay]) or min(args.ae_lr, args.dec_lr) <= 0 or min(args.lambda_cluster, args.weight_decay) < 0:
        raise ValueError("Invalid loss/optimizer parameters")
    saved = npz_read(args.results_dir / REFERENCE_NAME)
    validate_reference(saved)
    out = args.output_dir.resolve()
    if args.resume:
        if args.resume.resolve().parent != out or not args.resume.is_file():
            raise ValueError("--resume must point to a checkpoint in --output-dir")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new directory or resume dec_last.pt")
    device = choose_device(args.device)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = time_cache(args.h5, args.cache_dir, saved)
    dataset = TimeDataset(data)
    n, length = len(dataset), data.shape[-1]
    if n < 10:
        raise ValueError("At least 10 particles required")
    split_gen = torch.Generator().manual_seed(args.seed)
    train, val = random_split(dataset, [int(.8*n), n-int(.8*n)], generator=split_gen)
    dec_train, dec_other = random_split(dataset, [int(.5*n), n-int(.5*n)], generator=torch.Generator().manual_seed(args.seed+81))
    ae_gen = torch.Generator().manual_seed(args.seed)
    dec_gen = torch.Generator().manual_seed(args.seed+81)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=ae_gen)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    full_loader = DataLoader(dataset, batch_size=args.batch_size)
    dec_loader = DataLoader(dec_train, batch_size=args.batch_size, shuffle=True, generator=dec_gen)
    weights = saved.get("reconstruction_channel_weights", np.array([1, 1, 1, .2, .2, 1], dtype=np.float32))
    if weights.shape != (6,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid channel reconstruction weights")
    model = Conv1DAutoencoder(length).to(device)
    criterion = WeightedMSE(weights).to(device)
    contract = {"signature": reference_signature(saved), "architecture": "conv1d-8-16-32-64-k3-s2-v1",
                "latent_dim": 32, "k": 5, "weights": weights.tolist(),
                **{k: getattr(args, k) for k in ("seed", "batch_size", "ae_epochs", "ae_patience", "lambda_cluster",
                                                "ae_lr", "dec_lr", "weight_decay", "n_init", "monitor_sample_size")}}
    print(f"Input: {tuple(data.shape)}, latent_dim=32, K=5\nAE train/validation: {len(train)}/{len(val)}; DEC optimization/other: {len(dec_train)}/{len(dec_other)}\nLoss: weighted reconstruction + {args.lambda_cluster:g} * minibatch KL sum", flush=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["contract"] != contract:
            raise ValueError("Checkpoint training/preprocessing settings differ from current arguments")
        initial = npz_read(out / "ae_initial_clustering_1d.npz")
        if not np.array_equal(initial["unique_ids"], saved["unique_ids"]):
            raise ValueError("Saved initial features have different particle IDs")
        ae_history = checkpoint["ae_history"]
        ae_best_epoch = checkpoint["ae_best_epoch"]
    else:
        np.savez(out / "data_splits.npz", unique_ids=saved["unique_ids"], ae_train_indices=train.indices,
                 ae_validation_indices=val.indices, dec_train_indices=dec_train.indices, dec_other_indices=dec_other.indices)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
        ae_history, best_val, patience, ae_best_epoch = [], float("inf"), 0, 0
        for epoch in range(1, args.ae_epochs+1):
            loss = ae_epoch(model, train_loader, criterion, device, optimizer)
            val_loss = ae_epoch(model, val_loader, criterion, device)
            ae_history.append([epoch, loss, val_loss])
            csv_write(out / "ae_training_history.csv", ["epoch", "train_loss", "validation_loss"], ae_history)
            print(f"AE Epoch {epoch:03d} | train={loss:.8f} | validation={val_loss:.8f}", flush=True)
            if val_loss < best_val:
                best_val, patience, ae_best_epoch = val_loss, 0, epoch
                save_checkpoint(out / "ae_best.pt", {"model": state_cpu(model), "epoch": epoch, "validation_loss": val_loss, "contract": contract})
            else:
                patience += 1
            if patience >= args.ae_patience:
                break
        model.load_state_dict(torch.load(out / "ae_best.pt", map_location="cpu", weights_only=True)["model"])
        z_ae = extract(model, full_loader, device)
        print("Initializing K-means: K=5", flush=True)
        km = KMeans(n_clusters=5, n_init=args.n_init, max_iter=1000, random_state=args.seed)
        y_ae = km.fit_predict(z_ae)
        if len(np.unique(y_ae)) != 5:
            raise ValueError("AE initialization has fewer than five nonempty clusters")
        initial = {"unique_ids": saved["unique_ids"], "latent_vectors": z_ae, "kmeans_labels": y_ae,
                   "cluster_centers": km.cluster_centers_, "selected_k": np.array([5])}
        np.savez_compressed(out / "ae_initial_clustering_1d.npz", **initial)
    dec = DECModel(model, initial["cluster_centers"]).to(device)
    optimizer = torch.optim.Adam(dec.parameters(), lr=args.dec_lr, weight_decay=args.weight_decay)
    if args.resume:
        dec.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best_score = checkpoint["history"], checkpoint["best_score"]
        best_state, best_epoch = checkpoint["best_state"], checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"]+1
        if args.dec_epochs < checkpoint["epoch"]:
            raise ValueError("--dec-epochs cannot be smaller than resumed epoch")
        dec_gen.set_state(checkpoint["dec_generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        print(f"Resuming DEC at epoch {start_epoch}", flush=True)
    else:
        start_epoch, history, best_epoch = 1, [], 0
        best_score = sampled_score(initial["latent_vectors"], initial["kmeans_labels"], args.monitor_sample_size, args.seed)
        if not np.isfinite(best_score):
            raise ValueError("Initial silhouette is undefined; check sample/cluster sizes")
        best_state = state_cpu(dec)
    _, q, previous = extract(dec, full_loader, device, dec=True)
    p = target_distribution(q)
    header = ["epoch", "total_loss", "reconstruction_loss", "mean_batch_kl_sum", "weighted_kl_contribution", "latent_silhouette", "delta_alignment", "nonempty_clusters"]
    for epoch in range(start_epoch, args.dec_epochs+1):
        total, rec, cluster = dec_epoch(dec, dec_loader, criterion, optimizer, p, device, args.lambda_cluster)
        z, q, labels = extract(dec, full_loader, device, dec=True)
        p = target_distribution(q)
        delta = float(np.mean(labels != previous))
        score = sampled_score(z, labels, args.monitor_sample_size, args.seed)
        nonempty = len(np.unique(labels))
        history.append([epoch, float(total), float(rec), float(cluster), float(args.lambda_cluster*cluster), score, delta, nonempty])
        print(f"DEC Epoch {epoch:03d} | recon={rec:.8f} | weighted KL={args.lambda_cluster*cluster:.8f} | delta={delta:.6f} | latent silhouette={score:.6f} | clusters={nonempty}", flush=True)
        # Keep the requested five-cluster solution; report collapsed epochs in history.
        if nonempty == 5 and np.isfinite(score) and score > best_score:
            best_score, best_epoch, best_state = score, epoch, state_cpu(dec)
        previous = labels
        checkpoint = {"contract": contract, "epoch": epoch, "model": state_cpu(dec), "optimizer": optimizer.state_dict(),
                      "best_state": best_state, "best_score": best_score, "best_epoch": best_epoch,
                      "history": history, "ae_history": ae_history, "ae_best_epoch": ae_best_epoch,
                      "dec_generator_state": dec_gen.get_state(), "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_checkpoint(out / "dec_last.pt", checkpoint)
        csv_write(out / "dec_training_history.csv", header, history)
    dec.load_state_dict(best_state)
    save_checkpoint(out / "dec_best.pt", {"model": best_state, "epoch": best_epoch, "monitor_silhouette": best_score, "contract": contract})
    z, q, labels = extract(dec, full_loader, device, dec=True)
    ids = saved["unique_ids"]
    score = sampled_score(z, labels, args.sample_size, args.seed)
    np.savez_compressed(out / "h5_1d_cnn_dec_results.npz", unique_ids=ids, labels=labels, latent_vectors=z,
        selected_k=np.array([5]), selected_dec_epoch=np.array([best_epoch]), final_sampled_silhouette=np.array([score]),
        lambda_cluster=np.array([args.lambda_cluster]), batch_size=np.array([args.batch_size]),
        reconstruction_channel_weights=weights, **{k: saved[k] for k in
        ("channel_names", "source_time_indices", "sampled_times", "time_mean", "time_std")})
    comparison = []
    for name, space, features, y in [("1D AE + K-means", "1D AE latent", initial["latent_vectors"], initial["kmeans_labels"]),
                                    ("1D CNN + DEC", "1D DEC latent", z, labels),
                                    ("Spectrogram CNN + DEC", "2D DEC latent", saved["latent_vectors"], saved["labels"])]:
        comparison.append([name, space, 32, len(np.unique(y)), min(args.sample_size, n), sampled_score(features, y, args.sample_size, args.seed)])
    csv_write(out / "silhouette_comparison.csv", ["method", "evaluation_space", "dimensions", "nonempty_clusters", "sample_count", "silhouette"], comparison)
    csv_write(out / "cluster_counts.csv", ["cluster", "count"], enumerate(np.bincount(labels, minlength=5).tolist()))
    csv_write(out / "particle_labels.csv", ["particle_id", "initial_label", "final_label"],
              ((int(pid), int(a), int(b)) for pid, a, b in zip(ids, initial["kmeans_labels"], labels)))
    sample = np.random.RandomState(args.seed).permutation(n)[:min(args.sample_size, n)]
    csv_write(out / "evaluation_sample.csv", ["sample_order", "particle_id", "final_archive_row"], ((i, int(ids[j]), int(j)) for i, j in enumerate(sample)))
    diagnostic_plots(out, np.array(ae_history), np.array(history), dec.autoencoder, dataset, device, ids, labels, z, comparison)
    metadata = {"status": "complete", **contract, "reference_results": str((args.results_dir/REFERENCE_NAME).resolve()),
                "cache_dir": str(args.cache_dir.resolve()), "input_shape": list(data.shape),
                "parameter_count_autoencoder": sum(v.numel() for v in model.parameters()),
                "parameter_count_with_centers": sum(v.numel() for v in dec.parameters()),
                "ae_epochs_executed": len(ae_history), "ae_best_epoch": ae_best_epoch,
                "dec_epochs_executed": len(history), "selected_dec_epoch": best_epoch,
                "selected_monitor_silhouette": best_score, "final_silhouette": score,
                "final_sample_count": len(sample), "device": str(device), "threads": args.threads,
                "elapsed_seconds_this_invocation": time.perf_counter()-start_time,
                "selection": "Best 5000-sample silhouette by default, including epoch 0; only five nonempty clusters eligible; no DEC early stopping",
                "loss": "weighted_channel_MSE / mean(channel_weights) + lambda_cluster * KL_sum_over_minibatch_and_clusters",
                "splits": "AE 80/20 seed; DEC optimization 50/50 seed+81; target updates use all particles. Not an independent test set.",
                "comparison": "Same particle sample; each method's own 32D space; architecture parameter counts are not matched",
                "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": str(torch.__version__)}
    (out / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected DEC epoch: {best_epoch}; final silhouette: {score:.8f}\nStatus: complete\nResults: {out}", flush=True)


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    with threadpool_limits(limits=max(1, args.threads)), config_context(working_memory=max(1, args.working_memory)):
        run(args)


if __name__ == "__main__":
    main()
