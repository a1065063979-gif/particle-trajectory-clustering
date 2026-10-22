#!/usr/bin/env python
# coding: utf-8

import os
import random
import json

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 8))

import h5py
import matplotlib
import numpy as np
from scipy.signal import spectrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===== Config =====

H5_PATH = os.environ.get(
    "H5_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "particle_data.h5"),
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(SCRIPT_DIR, "..", "..", "results", "spectrogram_dec"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = int(os.environ.get("SEED", "42"))
K_MIN = int(os.environ.get("K_MIN", "2"))
K_MAX = int(os.environ.get("K_MAX", "12"))
KMEANS_N_INIT = int(os.environ.get("KMEANS_N_INIT", "100"))
K_SELECTION_SAMPLE_SIZE = int(os.environ.get("K_SELECTION_SAMPLE_SIZE", "10000"))
TARGET_TIME_POINTS = int(os.environ.get("TARGET_TIME_POINTS", "2048"))
# TARGET_TIME_POINTS is an upper bound, not an exact output length. A fixed
# integer stride preserves the original measurements and uniform time spacing.
TIME_STRIDE = int(os.environ.get("TIME_STRIDE", "0"))  # 0: choose automatically
FIGURE2_PARTICLE_INDICES = os.environ.get("FIGURE2_PARTICLE_INDICES", "")
PREPROCESS_ONLY = os.environ.get("PREPROCESS_ONLY", "0") == "1"
STFT_NPERSEG = int(os.environ.get("STFT_NPERSEG", "256"))
STFT_HOP = int(os.environ.get("STFT_HOP", "128"))
SPEC_FREQ_BINS = int(os.environ.get("SPEC_FREQ_BINS", "33"))
SPEC_TIME_BINS = int(os.environ.get("SPEC_TIME_BINS", "60"))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "32"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
MAX_AE_EPOCHS = int(os.environ.get("MAX_AE_EPOCHS", "200"))
AE_PATIENCE = int(os.environ.get("AE_PATIENCE", "15"))
MAX_DEC_EPOCHS = int(os.environ.get("MAX_DEC_EPOCHS", "350"))
MIN_DEC_EPOCHS = int(os.environ.get("MIN_DEC_EPOCHS", "10"))
DEC_PATIENCE = int(os.environ.get("DEC_PATIENCE", "25"))
DELTA_TOL = float(os.environ.get("DELTA_TOL", "0.005"))
# Report-inspired comparison: L = reconstruction + Lambda * clustering.
# Lambda multiplies the SUM of KL over the current minibatch and clusters.
# This retains our channel-weighted MSE and preprocessing; it is not an exact
# reproduction of the report's normalization or full-data KL definition.
LAMBDA_CLUSTER = float(os.environ.get("LAMBDA_CLUSTER", "5e-4"))
SPECTROGRAM_CHUNK_SIZE = int(os.environ.get("SPECTROGRAM_CHUNK_SIZE", "512"))
AE_LR = float(os.environ.get("AE_LR", "1e-3"))
DEC_LR = float(os.environ.get("DEC_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))

if K_MIN < 2:
    raise ValueError("K_MIN must be >= 2 for silhouette-based k selection.")
if K_MAX < K_MIN:
    raise ValueError("K_MAX must be >= K_MIN.")
if TARGET_TIME_POINTS < 2 or TIME_STRIDE < 0:
    raise ValueError("TARGET_TIME_POINTS must be >= 2 and TIME_STRIDE must be >= 0.")
if STFT_NPERSEG < 2 or not 1 <= STFT_HOP <= STFT_NPERSEG:
    raise ValueError("Require STFT_NPERSEG >= 2 and 1 <= STFT_HOP <= STFT_NPERSEG.")
if not np.isfinite(LAMBDA_CLUSTER) or LAMBDA_CLUSTER < 0:
    raise ValueError("LAMBDA_CLUSTER must be finite and nonnegative.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_current_fig(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", path, flush=True)


def uniform_time_step(values, label):
    """Validate actual record times, allowing only floating-point roundoff."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite 1D array with at least two times.")
    float_times = values.astype(np.float64)
    differences = np.diff(float_times)
    step = float(differences[0])
    if np.any(differences <= 0):
        raise ValueError(f"{label} must be strictly increasing.")
    tolerance = 8 * np.finfo(np.float64).eps * max(1.0, np.max(np.abs(float_times)))
    if not np.allclose(differences, step, rtol=1e-9, atol=tolerance):
        raise ValueError(
            f"{label} are not equally spaced (min dt={differences.min():.12g}, "
            f"max dt={differences.max():.12g}). Fixed-stride selection cannot "
            "guarantee uniform sampling. Check the Times dataset before computing spectra."
        )
    return step


def select_uniform_time_indices(times, target_count, stride_override=0):
    source_dt = uniform_time_step(times, "Original times")
    # ceil((N - 1) / (target - 1)), using integer arithmetic.
    stride = stride_override or max(1, (len(times) + target_count - 3) // (target_count - 1))
    indices = np.arange(0, len(times), stride, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("TIME_STRIDE is too large: fewer than two samples remain.")
    sampled_dt = uniform_time_step(times[indices], "Sampled times")
    # Never append the last record: that can introduce a shorter final interval.
    return indices, stride, source_dt, sampled_dt


def figure2_indices(n_particles):
    if FIGURE2_PARTICLE_INDICES.strip():
        indices = np.array([int(s.strip()) for s in FIGURE2_PARTICLE_INDICES.split(",")])
    else:
        indices = np.arange(min(5, n_particles))
    if not 1 <= len(indices) <= 10:
        raise ValueError("Choose between 1 and 10 FIGURE2_PARTICLE_INDICES (5 recommended).")
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n_particles):
        raise ValueError("FIGURE2_PARTICLE_INDICES must be distinct valid zero-based particle indices.")
    return indices


def save_figure2(raw, channels, means, stds, indices, particle_ids, record_times, source_indices):
    """Plot actual globally standardized input traces; export reproducible data."""
    sample_index = np.arange(len(record_times))
    traces = np.stack([
        ((raw[name][:, indices] - means[name]) / stds[name]).T
        for name in channels
    ], axis=1)  # (selected particles, channels, sampled times)
    colors = plt.get_cmap("tab10")(np.arange(len(indices)))
    titles = [r"$x$", r"$p_x$", r"$p_y$", r"$E_x$", r"$E_y$", r"$B_z$"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for channel, ax in enumerate(axes.ravel()):
        for j, particle_index in enumerate(indices):
            ax.plot(sample_index, traces[j, channel], color=colors[j], linewidth=0.9,
                    alpha=0.85, label=f"Particle index {particle_index}")
        ax.set_title(titles[channel])
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Standardized value")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, len(sample_index) - 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(indices)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    stem = "figure2_standardized_six_channel_timeseries"
    for extension in ("png", "pdf"):
        path = os.path.join(OUTPUT_DIR, f"{stem}.{extension}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("Saved Figure 2:", path, flush=True)
    plt.close(fig)
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, f"{stem}_data.npz"),
        particle_indices=indices, unique_ids=particle_ids[indices],
        channel_names=np.array(channels), sampled_times=record_times,
        source_time_indices=source_indices, standardized_traces=traces,
        time_mean=np.array([means[name] for name in channels]),
        time_std=np.array([stds[name] for name in channels]),
    )


def cluster_mean_and_std(data_time_particle, labels, n_clusters):
    means = []
    stds = []
    counts = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        counts.append(int(mask.sum()))
        cluster_data = data_time_particle[:, mask]
        if cluster_data.shape[1] == 0:
            means.append(np.full(data_time_particle.shape[0], np.nan, dtype=np.float32))
            stds.append(np.full(data_time_particle.shape[0], np.nan, dtype=np.float32))
        else:
            means.append(cluster_data.mean(axis=1))
            stds.append(cluster_data.std(axis=1))

    return np.stack(means), np.stack(stds), np.array(counts)


def compute_log_spectrograms(data_particle_time, nperseg, hop, chunk_size, target_hw):
    noverlap = nperseg - hop
    n_particles = data_particle_time.shape[0]
    out = np.empty((n_particles, target_hw[0], target_hw[1]), dtype=np.float32)

    for start in range(0, n_particles, chunk_size):
        end = min(start + chunk_size, n_particles)
        _, _, sxx = spectrogram(
            data_particle_time[start:end],
            fs=1.0,  # frequency in cycles per uniformly sampled point, not Hz
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="magnitude",
            axis=-1,
        )
        sxx = np.log1p(sxx).astype(np.float32)
        sxx_torch = torch.from_numpy(sxx).unsqueeze(1)
        resized = F.interpolate(sxx_torch, size=target_hw, mode="bilinear", align_corners=False)
        out[start:end] = resized.squeeze(1).numpy().astype(np.float32)
        print(f"  spectrogram particles {start}:{end}", flush=True)

    return out


# ===== Load H5 and build six-channel spectrogram input =====

model_channels = ["x", "px", "py", "Ex", "Ey", "Bz"]
print("H5 path:", H5_PATH, flush=True)
print("Output dir:", OUTPUT_DIR, flush=True)
print("Channels:", model_channels, flush=True)
print("Auto k candidate range:", K_MIN, "to", K_MAX, flush=True)
print("KMeans n_init:", KMEANS_N_INIT, flush=True)
print("Target maximum time points (automatic stride):", TARGET_TIME_POINTS, flush=True)
print("STFT nperseg/hop:", STFT_NPERSEG, STFT_HOP, flush=True)
print("Spectrogram target size:", (SPEC_FREQ_BINS, SPEC_TIME_BINS), flush=True)
print("DEC objective: weighted reconstruction MSE + LAMBDA_CLUSTER * minibatch KL sum", flush=True)
print("LAMBDA_CLUSTER:", LAMBDA_CLUSTER, "Batch size:", BATCH_SIZE, flush=True)

with h5py.File(H5_PATH, "r") as f:
    times = f["Times"][:]
    n_time, n_particles = f["x"].shape
    if times.shape != (n_time,) or n_particles < 1:
        raise ValueError("Times must match the time axis of x; at least one particle is required.")
    time_indices, time_stride, original_dt, sampled_dt = select_uniform_time_indices(
        times, TARGET_TIME_POINTS, TIME_STRIDE,
    )
    if len(time_indices) < STFT_NPERSEG:
        raise ValueError(
            f"Fixed-stride sampling leaves {len(time_indices)} points, fewer than "
            f"STFT_NPERSEG={STFT_NPERSEG}. Increase TARGET_TIME_POINTS or reduce TIME_STRIDE."
        )
    selected_figure2_indices = figure2_indices(n_particles)

    print("Available datasets:", list(f.keys()), flush=True)
    print("Original time steps:", n_time, flush=True)
    print("Particles:", n_particles, flush=True)
    print("Sampled time steps:", len(time_indices), flush=True)
    print(f"Fixed stride: {time_stride}; original dt: {original_dt}; sampled dt: {sampled_dt}", flush=True)

    raw = {}
    for name in model_channels:
        if f[name].shape != (n_time, n_particles):
            raise ValueError(f"{name} does not have the same time/particle shape as x.")
        arr = f[name][time_indices, :].astype(np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite sampled values; cannot standardize.")
        raw[name] = arr
        print(f"{name}: sampled shape={arr.shape}, nan={np.isnan(arr).any()}, inf={np.isinf(arr).any()}", flush=True)

    sampled_times = times[time_indices]
    unique_ids = f["unique_Ids"][:]
    if unique_ids.shape != (n_particles,):
        raise ValueError("unique_Ids must contain one identifier per particle.")

time_mean = {}
time_std = {}

for name in model_channels:
    arr = raw[name]
    mean = arr.mean(dtype=np.float64).astype(np.float32)
    std = arr.std(dtype=np.float64).astype(np.float32)
    if std < 1e-8:
        std = np.float32(1.0)
    time_mean[name] = mean
    time_std[name] = std

save_figure2(
    raw, model_channels, time_mean, time_std, selected_figure2_indices,
    unique_ids, sampled_times, time_indices,
)
sampling_metadata = {
    "sampling_method": "fixed_stride_no_interpolation",
    "original_time_points": n_time,
    "sampled_time_points": len(time_indices),
    "target_maximum_time_points": TARGET_TIME_POINTS,
    "time_stride": time_stride,
    "original_time_step": original_dt,
    "sampled_time_step": sampled_dt,
    "original_first_time": float(times[0]),
    "original_last_time": float(times[-1]),
    "sampled_first_time": float(sampled_times[0]),
    "sampled_last_time": float(sampled_times[-1]),
    "trailing_records_omitted": int(n_time - 1 - time_indices[-1]),
    "particle_count": n_particles,
    "figure2_particle_indices": selected_figure2_indices.tolist(),
    "figure2_unique_ids": [int(unique_ids[i]) for i in selected_figure2_indices],
    "normalization_scope": "all_particles_and_selected_times_per_channel",
    "spectrogram_fs": 1.0,
    "frequency_units": "cycles_per_sample",
    "anti_alias_filter_applied": False,
}
with open(os.path.join(OUTPUT_DIR, "sampling_metadata.json"), "w", encoding="utf-8") as handle:
    json.dump(sampling_metadata, handle, indent=2, ensure_ascii=False)
if PREPROCESS_ONLY:
    print("PREPROCESS_ONLY=1: Figure 2 and sampling metadata saved; training skipped.", flush=True)
    raise SystemExit(0)

# Defer PyTorch imports so the paper figure can be generated without PyTorch.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

set_seed(SEED)
spectrograms = []
for name in model_channels:
    arr = raw[name]
    mean = time_mean[name]
    std = time_std[name]
    scaled_particle_time = ((arr - mean) / std).T.astype(np.float32)

    print(f"Computing log-spectrogram for {name}", flush=True)
    spec = compute_log_spectrograms(
        scaled_particle_time,
        STFT_NPERSEG,
        STFT_HOP,
        SPECTROGRAM_CHUNK_SIZE,
        (SPEC_FREQ_BINS, SPEC_TIME_BINS),
    )
    spectrograms.append(spec)

data_spec = np.stack(spectrograms, axis=1).astype(np.float32)
del spectrograms

spec_mean = data_spec.mean(axis=(0, 2, 3), keepdims=True)
spec_std = data_spec.std(axis=(0, 2, 3), keepdims=True)
spec_std = np.where(spec_std < 1e-8, 1.0, spec_std)
data_spec -= spec_mean
data_spec /= spec_std
data_scaled = data_spec

_, n_channels, n_freq, n_frames = data_scaled.shape
print("Spectrogram tensor after scaling:", data_scaled.shape, flush=True)
print("Frequency bins:", n_freq, "Time frames:", n_frames, flush=True)


# ===== Quick input diagnostics =====

plt.figure(figsize=(10, 4))
for i in range(min(5, n_particles)):
    plt.plot(sampled_times, raw["x"][:, i], label=f"particle {i}")
plt.xlabel("time")
plt.ylabel("x")
plt.title("x(t), first 5 particles")
plt.legend()
plt.grid(True)
save_current_fig("01_h5_x_timeseries_first5_spectrogram_k_auto_without_pz.png")

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.ravel()
particle_to_show = 0
for ax, name, channel_idx in zip(axes, model_channels, range(len(model_channels))):
    im = ax.imshow(data_spec[particle_to_show, channel_idx], origin="lower", aspect="auto", cmap="magma")
    ax.set_title(name)
    ax.set_xlabel("Resized time-bin index")
    ax.set_ylabel("Resized frequency-bin index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle(f"Input log-spectrograms, particle {particle_to_show}", fontsize=14)
fig.tight_layout()
save_current_fig("02_input_log_spectrogram_particle0_without_pz.png")


# ===== Torch dataset =====

class ParticleSpectrogramDataset(Dataset):
    def __init__(self, array):
        self.tensor = torch.from_numpy(array)

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        return idx, self.tensor[idx]


dataset = ParticleSpectrogramDataset(data_scaled)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, flush=True)
print("Dataset size:", len(dataset), flush=True)
print("Train/val:", len(train_dataset), len(val_dataset), flush=True)
print("Batch size:", BATCH_SIZE, flush=True)


# ===== 2D convolutional autoencoder =====


class Conv2DSpectrogramAutoencoder(nn.Module):
    def __init__(self, n_channels, input_hw, latent_dim):
        super().__init__()
        self.input_hw = input_hw
        self.enc_conv1 = nn.Conv2d(n_channels, 8, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, input_hw[0], input_hw[1])
            encoded = self._encode_conv(dummy)

        self.encoded_shape = tuple(encoded.shape[1:])
        self.flatten_dim = int(np.prod(self.encoded_shape))
        self.fc_enc = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        self.dec_deconv1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=0
        )
        self.dec_deconv2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=0
        )
        self.dec_deconv3 = nn.ConvTranspose2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=0
        )
        self.dec_deconv4 = nn.ConvTranspose2d(
            in_channels=8, out_channels=n_channels, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def _encode_conv(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        return x

    def encode(self, x):
        h = self._encode_conv(x)
        h = self.flatten(h)
        return self.fc_enc(h)

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), *self.encoded_shape)
        h = F.relu(self.dec_deconv1(h))
        h = F.relu(self.dec_deconv2(h))
        h = F.relu(self.dec_deconv3(h))
        x_recon = self.dec_deconv4(h)
        if x_recon.shape[-2:] != self.input_hw:
            x_recon = F.interpolate(x_recon, size=self.input_hw, mode="bilinear", align_corners=False)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


model = Conv2DSpectrogramAutoencoder(
    n_channels=n_channels,
    input_hw=(n_freq, n_frames),
    latent_dim=LATENT_DIM,
).to(device)

channel_recon_weights_np = np.array(
    [0.2 if name in ("Ex", "Ey") else 1.0 for name in model_channels],
    dtype=np.float32,
)
channel_recon_weights = torch.tensor(channel_recon_weights_np, dtype=torch.float32, device=device)
print("Reconstruction channel weights:", dict(zip(model_channels, channel_recon_weights_np)), flush=True)


class WeightedChannelMSELoss(nn.Module):
    def __init__(self, channel_weights):
        super().__init__()
        self.register_buffer("channel_weights", channel_weights.view(1, -1, 1, 1))
        self.normalizer = torch.mean(channel_weights)

    def forward(self, recon, target):
        squared_error = (recon - target) ** 2
        weighted_error = squared_error * self.channel_weights
        return weighted_error.mean() / self.normalizer


criterion = WeightedChannelMSELoss(channel_recon_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=WEIGHT_DECAY)
print("AE optimizer lr:", AE_LR, "weight_decay:", WEIGHT_DECAY, flush=True)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for _, batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
        total_samples += batch.size(0)
    return total_loss / total_samples


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for _, batch in dataloader:
        batch = batch.to(device)
        recon, _ = model(batch)
        loss = criterion(recon, batch)
        total_loss += loss.item() * batch.size(0)
        total_samples += batch.size(0)
    return total_loss / total_samples


@torch.no_grad()
def save_autoencoder_reconstruction_figure(model, dataset, device, filename, particle_idx=0):
    model.eval()
    _, sample = dataset[particle_idx]
    sample_batch = sample.unsqueeze(0).to(device)
    recon, _ = model(sample_batch)
    sample_np = sample.cpu().numpy()
    recon_np = recon.squeeze(0).cpu().numpy()

    fig, axes = plt.subplots(len(model_channels), 2, figsize=(9, 2.2 * len(model_channels)))
    for channel_idx, name in enumerate(model_channels):
        vmin = np.percentile(sample_np[channel_idx], 2)
        vmax = np.percentile(sample_np[channel_idx], 98)
        axes[channel_idx, 0].imshow(sample_np[channel_idx], origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        axes[channel_idx, 0].set_title(f"{name} input")
        axes[channel_idx, 1].imshow(recon_np[channel_idx], origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
        axes[channel_idx, 1].set_title(f"{name} recon")
        for ax in axes[channel_idx]:
            ax.set_xlabel("STFT frame")
            ax.set_ylabel("Frequency bin")
    fig.suptitle(f"Autoencoder reconstruction, particle {particle_idx}", fontsize=14)
    fig.tight_layout()
    save_current_fig(filename)


@torch.no_grad()
def compute_channel_reconstruction_mse(model, dataloader, device):
    model.eval()
    loss_sum = torch.zeros(n_channels, dtype=torch.float64)
    total_elements_per_channel = 0

    for _, batch in dataloader:
        batch = batch.to(device)
        recon, _ = model(batch)
        squared_error = (recon - batch) ** 2
        loss_sum += squared_error.sum(dim=(0, 2, 3)).cpu().double()
        total_elements_per_channel += batch.size(0) * batch.size(2) * batch.size(3)

    return (loss_sum / total_elements_per_channel).numpy()


def save_channel_reconstruction_mse(channel_mse, filename):
    weighted_channel_mse = channel_mse * channel_recon_weights_np
    path = os.path.join(OUTPUT_DIR, filename.replace(".png", ".csv"))
    np.savetxt(
        path,
        np.column_stack(
            [
                np.array(model_channels),
                channel_mse.astype(str),
                channel_recon_weights_np.astype(str),
                weighted_channel_mse.astype(str),
            ]
        ),
        delimiter=",",
        header="channel,mse,weight,weighted_mse",
        comments="",
        fmt="%s",
    )
    print("Saved channel reconstruction MSE:", path, flush=True)

    plt.figure(figsize=(7, 4))
    plt.bar(model_channels, channel_mse)
    plt.xlabel("Channel")
    plt.ylabel("MSE")
    plt.title("AE reconstruction MSE by channel")
    plt.grid(axis="y", alpha=0.3)
    save_current_fig(filename)


train_losses = []
val_losses = []
best_val_loss = float("inf")
best_state_dict = None
epochs_without_improvement = 0

for epoch in range(MAX_AE_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_one_epoch(model, val_loader, criterion, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"AE Epoch {epoch + 1:03d} | train loss = {train_loss:.8f} | val loss = {val_loss:.8f}", flush=True)

    if epoch == 0:
        save_autoencoder_reconstruction_figure(
            model,
            dataset,
            device,
            "03_ae_reconstruction_after_epoch001_spectrogram_k_auto_without_pz.png",
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= AE_PATIENCE:
        print(f"AE early stopping at epoch {epoch + 1}", flush=True)
        break

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    model.to(device)

save_autoencoder_reconstruction_figure(
    model,
    dataset,
    device,
    "04_ae_reconstruction_best_spectrogram_k_auto_without_pz.png",
)

channel_recon_mse = compute_channel_reconstruction_mse(model, full_loader, device)
print("AE reconstruction MSE by channel:", dict(zip(model_channels, channel_recon_mse)), flush=True)
save_channel_reconstruction_mse(
    channel_recon_mse,
    "05_ae_reconstruction_mse_by_channel_spectrogram_k_auto_without_pz.png",
)

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="val loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Spectrogram 2D CNN autoencoder pretraining loss")
plt.legend()
plt.grid(True)
save_current_fig("06_ae_pretraining_loss_spectrogram_k_auto_without_pz.png")


@torch.no_grad()
def extract_latent_vectors(model, dataloader, device):
    model.eval()
    index_list = []
    latent_list = []
    for indices, batch in dataloader:
        batch = batch.to(device)
        z = model.encode(batch)
        index_list.append(indices.cpu())
        latent_list.append(z.cpu())
    indices = torch.cat(index_list, dim=0)
    latents = torch.cat(latent_list, dim=0)
    order = torch.argsort(indices)
    return latents[order].numpy()


latent_vectors = extract_latent_vectors(model, full_loader, device)
print("AE latent shape:", latent_vectors.shape, flush=True)

max_valid_k = min(K_MAX, len(dataset) - 1)
candidate_ks = list(range(K_MIN, max_valid_k + 1))
if not candidate_ks:
    raise ValueError("No valid k candidates. Need at least K_MIN + 1 samples.")

k_selection_rows = []
kmeans_by_k = {}
labels_by_k = {}

print("Selecting best k by KMeans inertia elbow...", flush=True)
for candidate_k in candidate_ks:
    candidate_kmeans = KMeans(
        n_clusters=candidate_k,
        n_init=KMEANS_N_INIT,
        max_iter=1000,
        random_state=SEED,
    )
    candidate_labels = candidate_kmeans.fit_predict(latent_vectors)
    candidate_counts = np.bincount(candidate_labels, minlength=candidate_k)
    candidate_sil = silhouette_score(
        latent_vectors,
        candidate_labels,
        sample_size=min(K_SELECTION_SAMPLE_SIZE, len(dataset)),
        random_state=SEED,
    )
    kmeans_by_k[candidate_k] = candidate_kmeans
    labels_by_k[candidate_k] = candidate_labels
    k_selection_rows.append(
        [
            candidate_k,
            candidate_kmeans.inertia_,
            np.log(candidate_kmeans.inertia_ + 1e-12),
            0.0,
            candidate_sil,
            candidate_counts.min(),
            candidate_counts.max(),
        ]
    )
    print(
        f"  k={candidate_k:02d} | inertia = {candidate_kmeans.inertia_:.6f} | "
        f"sampled silhouette = {candidate_sil:.6f} | "
        f"min/max count = {candidate_counts.min()}/{candidate_counts.max()}",
        flush=True,
    )

k_selection_array = np.array(k_selection_rows, dtype=float)

if len(candidate_ks) >= 3:
    # Elbow score: distance from each normalized log-inertia point to the line
    # connecting the first and last candidate. The largest distance is the knee.
    k_values = k_selection_array[:, 0]
    log_inertia_values = k_selection_array[:, 2]
    x_span = k_values.max() - k_values.min()
    y_span = log_inertia_values.max() - log_inertia_values.min()
    x_norm = (k_values - k_values.min()) / x_span
    if y_span < 1e-12:
        y_norm = np.zeros_like(log_inertia_values)
    else:
        y_norm = (log_inertia_values - log_inertia_values.min()) / y_span

    points = np.column_stack([x_norm, y_norm])
    line_start = points[0]
    line_end = points[-1]
    line_vec = line_end - line_start
    line_norm = np.linalg.norm(line_vec)
    if line_norm < 1e-12:
        elbow_scores = np.zeros(len(candidate_ks), dtype=np.float64)
    else:
        relative_points = points - line_start
        elbow_scores = np.abs(
            line_vec[0] * relative_points[:, 1] - line_vec[1] * relative_points[:, 0]
        ) / line_norm
    elbow_scores[0] = 0.0
    elbow_scores[-1] = 0.0
    k_selection_array[:, 3] = elbow_scores
    best_k = int(k_selection_array[np.argmax(elbow_scores), 0])
else:
    best_k = int(candidate_ks[-1])
    k_selection_array[k_selection_array[:, 0] == best_k, 3] = 1.0

N_CLUSTERS = best_k
kmeans = kmeans_by_k[N_CLUSTERS]
kmeans_labels = labels_by_k[N_CLUSTERS]
kmeans_sil = float(k_selection_array[k_selection_array[:, 0] == N_CLUSTERS, 4][0])
kmeans_inertia = float(k_selection_array[k_selection_array[:, 0] == N_CLUSTERS, 1][0])
kmeans_elbow_score = float(k_selection_array[k_selection_array[:, 0] == N_CLUSTERS, 3][0])
k_selection_path = os.path.join(OUTPUT_DIR, "k_selection_elbow_spectrogram_k_auto_without_pz.csv")
np.savetxt(
    k_selection_path,
    k_selection_array,
    delimiter=",",
    header="k,inertia,log_inertia,elbow_score,sampled_silhouette,min_cluster_count,max_cluster_count",
    comments="",
)
print("Saved k selection scores:", k_selection_path, flush=True)
print("KMeans cluster counts:", np.bincount(kmeans_labels, minlength=N_CLUSTERS), flush=True)
print(f"Selected K: {N_CLUSTERS}", flush=True)
print(f"Selected KMeans inertia: {kmeans_inertia:.6f}", flush=True)
print(f"Selected elbow score: {kmeans_elbow_score:.6f}", flush=True)
print(f"Selected KMeans sampled silhouette: {kmeans_sil:.6f}", flush=True)

plt.figure(figsize=(7, 5))
plt.plot(k_selection_array[:, 0], k_selection_array[:, 2], marker="o", label="log inertia")
plt.axvline(N_CLUSTERS, color="red", linestyle="--", label=f"selected k={N_CLUSTERS}")
plt.xlabel("K")
plt.ylabel("Log inertia")
plt.title("Automatic k selection by elbow")
plt.legend()
plt.grid(True)
save_current_fig("06_ae_latent_k_selection_elbow_spectrogram_k_auto_without_pz.png")

plt.figure(figsize=(7, 5))
plt.plot(k_selection_array[:, 0], k_selection_array[:, 4], marker="o")
plt.axvline(N_CLUSTERS, color="red", linestyle="--", label=f"selected k={N_CLUSTERS}")
plt.xlabel("K")
plt.ylabel("Sampled silhouette")
plt.title("Silhouette reference for selected k")
plt.legend()
plt.grid(True)
save_current_fig("06_ae_latent_k_selection_silhouette_reference_spectrogram_k_auto_without_pz.png")

pca_plot = PCA(n_components=2, random_state=SEED).fit_transform(latent_vectors)
plt.figure(figsize=(7, 5))
plt.scatter(pca_plot[:, 0], pca_plot[:, 1], c=kmeans_labels, s=8, cmap="tab10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"AE spectrogram latent PCA + KMeans, K={N_CLUSTERS}")
plt.colorbar(label="cluster")
save_current_fig("06_ae_latent_pca_kmeans_spectrogram_k_auto_without_pz.png")


# ===== DEC =====

def student_t_distribution(z, centers, alpha=1.0):
    dist_sq = torch.sum((z.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
    q = (1.0 + dist_sq / alpha) ** (-(alpha + 1.0) / 2.0)
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q


def target_distribution(q):
    weight = (q ** 2) / torch.sum(q, dim=0, keepdim=True)
    return weight / torch.sum(weight, dim=1, keepdim=True)


def kl_divergence_loss(p, q, eps=1e-10):
    q = torch.clamp(q, min=eps)
    p = torch.clamp(p, min=eps)
    return torch.sum(p * torch.log(p / q))


class DECModel(nn.Module):
    def __init__(self, autoencoder, cluster_centers):
        super().__init__()
        self.autoencoder = autoencoder
        self.cluster_centers = nn.Parameter(cluster_centers)

    def forward(self, x):
        x_recon, z = self.autoencoder(x)
        q = student_t_distribution(z, self.cluster_centers)
        return x_recon, z, q


cluster_centers_init = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
dec_model = DECModel(model, cluster_centers_init).to(device)
dec_optimizer = torch.optim.Adam(dec_model.parameters(), lr=DEC_LR, weight_decay=WEIGHT_DECAY)
print("DEC optimizer lr:", DEC_LR, "weight_decay:", WEIGHT_DECAY, flush=True)

dec_train_size = int(0.5 * len(dataset))
dec_holdout_size = len(dataset) - dec_train_size
dec_train_dataset, dec_holdout_dataset = random_split(
    dataset,
    [dec_train_size, dec_holdout_size],
    generator=torch.Generator().manual_seed(SEED + 81),
)
dec_train_loader = DataLoader(dec_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("DEC train/holdout:", len(dec_train_dataset), len(dec_holdout_dataset), flush=True)


@torch.no_grad()
def extract_q_and_labels(dec_model, dataloader, device):
    dec_model.eval()
    index_list = []
    q_list = []
    for indices, batch in dataloader:
        batch = batch.to(device)
        _, _, q = dec_model(batch)
        index_list.append(indices.cpu())
        q_list.append(q.cpu())
    indices = torch.cat(index_list, dim=0)
    q_all = torch.cat(q_list, dim=0)
    order = torch.argsort(indices)
    q_all = q_all[order]
    labels_all = torch.argmax(q_all, dim=1)
    return q_all, labels_all


@torch.no_grad()
def compute_full_dec_state(dec_model, dataloader, device):
    q_all, labels_all = extract_q_and_labels(dec_model, dataloader, device)
    p_all = target_distribution(q_all)
    return q_all, p_all, labels_all


def train_one_dec_epoch(dec_model, dataloader, optimizer, p_full, device, lambda_cluster):
    dec_model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    total_samples = 0

    for batch_indices, batch in dataloader:
        batch = batch.to(device)
        p_batch = p_full[batch_indices].to(device)

        optimizer.zero_grad()
        x_recon, _, q = dec_model(batch)
        recon_loss = criterion(x_recon, batch)
        kl_loss = kl_divergence_loss(p_batch, q)
        total_loss = recon_loss + lambda_cluster * kl_loss
        total_loss.backward()
        optimizer.step()

        bsz = batch.size(0)
        total_loss_sum += total_loss.item() * bsz
        recon_loss_sum += recon_loss.item() * bsz
        # Sample-weighted mean of the minibatch KL sums used in the objective.
        # This makes logged total == recon + lambda_cluster * logged KL.
        kl_loss_sum += kl_loss.item() * bsz
        total_samples += bsz

    return (
        total_loss_sum / total_samples,
        recon_loss_sum / total_samples,
        kl_loss_sum / total_samples,
    )


q_init, p_full, labels_prev = compute_full_dec_state(dec_model, full_loader, device)
print("Initial DEC cluster counts:", torch.bincount(labels_prev, minlength=N_CLUSTERS).cpu().numpy(), flush=True)
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "ae_initial_clustering_report_weight.npz"),
    latent_vectors=latent_vectors,
    kmeans_labels=kmeans_labels,
    initial_dec_labels=labels_prev.cpu().numpy(),
    unique_ids=unique_ids,
    selected_k=np.array([N_CLUSTERS]),
)

dec_total_losses = []
dec_recon_losses = []
dec_kl_losses = []
delta_alignments = []
dec_silhouettes = []
best_dec_silhouette = kmeans_sil
best_dec_state_dict = {k: v.cpu().clone() for k, v in dec_model.state_dict().items()}
epochs_without_improvement = 0

for epoch in range(MAX_DEC_EPOCHS):
    total_loss, recon_loss, kl_loss = train_one_dec_epoch(
        dec_model, dec_train_loader, dec_optimizer, p_full, device, LAMBDA_CLUSTER
    )
    dec_total_losses.append(total_loss)
    dec_recon_losses.append(recon_loss)
    dec_kl_losses.append(kl_loss)

    q_all, p_full, labels_now = compute_full_dec_state(dec_model, full_loader, device)
    delta_alignment = (labels_now != labels_prev).float().mean().item()
    delta_alignments.append(delta_alignment)

    current_latents = extract_latent_vectors(dec_model.autoencoder, full_loader, device)
    current_labels = labels_now.cpu().numpy()
    current_silhouette = silhouette_score(
        current_latents,
        current_labels,
        sample_size=min(5000, len(dataset)),
        random_state=SEED,
    )
    dec_silhouettes.append(current_silhouette)

    print(
        f"DEC Epoch {epoch + 1:03d} | total loss = {total_loss:.8f} | "
        f"recon loss = {recon_loss:.8f} | batch KL sum = {kl_loss:.8f} | "
        f"weighted KL = {LAMBDA_CLUSTER * kl_loss:.8f} | "
        f"delta alignment = {delta_alignment:.6f} | "
        f"latent silhouette = {current_silhouette:.6f}",
        flush=True,
    )

    if np.isfinite(current_silhouette) and current_silhouette > best_dec_silhouette:
        best_dec_silhouette = current_silhouette
        best_dec_state_dict = {k: v.cpu().clone() for k, v in dec_model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    labels_prev = labels_now

if best_dec_state_dict is not None:
    dec_model.load_state_dict(best_dec_state_dict)
    dec_model.to(device)

plt.figure(figsize=(9, 5))
plt.plot(dec_total_losses, label="total loss")
plt.plot(dec_recon_losses, label="recon loss")
plt.plot(LAMBDA_CLUSTER * np.array(dec_kl_losses), label="weighted KL contribution")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Spectrogram DEC training losses")
plt.legend()
plt.grid(True)
save_current_fig("07_dec_training_losses_spectrogram_k_auto_without_pz.png")
np.savetxt(
    os.path.join(OUTPUT_DIR, "dec_training_history_report_weight.csv"),
    np.column_stack([
        np.arange(1, len(dec_total_losses) + 1), dec_total_losses,
        dec_recon_losses, dec_kl_losses,
        LAMBDA_CLUSTER * np.array(dec_kl_losses),
        dec_silhouettes, delta_alignments,
    ]),
    delimiter=",",
    header="epoch,total_loss,reconstruction_loss,mean_batch_kl_sum,weighted_kl_contribution,latent_silhouette,delta_alignment",
    comments="",
)

plt.figure(figsize=(9, 5))
plt.plot(delta_alignments, label="delta alignment")
plt.axhline(DELTA_TOL, color="red", linestyle="--", label=f"delta_tol={DELTA_TOL}")
plt.plot(dec_silhouettes, label="latent silhouette")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Spectrogram DEC monitoring metrics")
plt.legend()
plt.grid(True)
save_current_fig("08_dec_delta_silhouette_spectrogram_k_auto_without_pz.png")

latent_vectors_dec = extract_latent_vectors(dec_model.autoencoder, full_loader, device)
q_final, labels_final_torch = extract_q_and_labels(dec_model, full_loader, device)
labels_dec = labels_final_torch.cpu().numpy()
final_silhouette = silhouette_score(
    latent_vectors_dec,
    labels_dec,
    sample_size=min(10000, len(dataset)),
    random_state=SEED,
)

print("Final DEC cluster counts:", np.bincount(labels_dec, minlength=N_CLUSTERS), flush=True)
print(f"Final DEC sampled silhouette: {final_silhouette:.6f}", flush=True)

results_path = os.path.join(OUTPUT_DIR, "h5_spectrogram_cnn_dec_k_auto_without_pz_results.npz")
np.savez(
    results_path,
    labels=labels_dec,
    unique_ids=unique_ids,
    latent_vectors=latent_vectors_dec,
    channel_names=np.array(model_channels),
    time_mean=np.array([time_mean[name] for name in model_channels]),
    time_std=np.array([time_std[name] for name in model_channels]),
    spec_mean=spec_mean.reshape(-1),
    spec_std=spec_std.reshape(-1),
    ae_channel_reconstruction_mse=channel_recon_mse,
    reconstruction_channel_weights=channel_recon_weights_np,
    loss_formula=np.array(["reconstruction + lambda_cluster * minibatch_kl_sum"]),
    lambda_cluster=np.array([LAMBDA_CLUSTER]),
    kl_reduction=np.array(["sum_over_minibatch_and_clusters"]),
    batch_size=np.array([BATCH_SIZE]),
    dec_total_losses=np.array(dec_total_losses),
    dec_reconstruction_losses=np.array(dec_recon_losses),
    dec_batch_kl_sums=np.array(dec_kl_losses),
    final_sampled_silhouette=np.array([final_silhouette]),
    stft_nperseg=np.array([STFT_NPERSEG]),
    stft_hop=np.array([STFT_HOP]),
    spec_freq_bins=np.array([SPEC_FREQ_BINS]),
    spec_time_bins=np.array([SPEC_TIME_BINS]),
    sampled_times=sampled_times,
    source_time_indices=time_indices,
    original_time_points=np.array([n_time]),
    time_stride=np.array([time_stride]),
    sampled_time_step=np.array([sampled_dt]),
    sampling_method=np.array(["fixed_stride_no_interpolation"]),
    selected_k=np.array([N_CLUSTERS]),
    k_selection_scores=k_selection_array,
    k_selection_method=np.array(["log_inertia_elbow"]),
    dec_latent_silhouettes=np.array(dec_silhouettes),
    dec_delta_alignments=np.array(delta_alignments),
)
print("Saved results:", results_path, flush=True)

pca_dec = PCA(n_components=2, random_state=SEED).fit_transform(latent_vectors_dec)
plt.figure(figsize=(7, 5))
plt.scatter(pca_dec[:, 0], pca_dec[:, 1], c=labels_dec, s=8, cmap="tab10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"Spectrogram DEC latent PCA, K={N_CLUSTERS}")
plt.colorbar(label="cluster")
save_current_fig("09_dec_latent_pca_spectrogram_k_auto_without_pz.png")

tsne_sample_size = min(6000, len(dataset))
rng = np.random.default_rng(SEED)
tsne_indices = rng.choice(len(dataset), size=tsne_sample_size, replace=False)
tsne_features = latent_vectors_dec[tsne_indices]
tsne_labels = labels_dec[tsne_indices]
tsne_perplexity = min(120, max(5, tsne_sample_size // 50))

tsne_kwargs = dict(
    n_components=2,
    perplexity=tsne_perplexity,
    learning_rate="auto",
    init="pca",
    random_state=SEED,
)
import inspect

if "max_iter" in inspect.signature(TSNE).parameters:
    tsne_kwargs["max_iter"] = 1500
else:
    tsne_kwargs["n_iter"] = 1500

print("t-SNE sample size:", tsne_sample_size, flush=True)
print("t-SNE perplexity:", tsne_perplexity, flush=True)
tsne_vectors = TSNE(**tsne_kwargs).fit_transform(tsne_features)

plt.figure(figsize=(7, 5))
plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c=tsne_labels, s=6, cmap="tab10")
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
plt.title(f"Spectrogram DEC latent t-SNE, K={N_CLUSTERS}")
plt.colorbar(label="cluster")
save_current_fig("10_dec_latent_tsne_spectrogram_k_auto_without_pz.png")


# ===== Physical interpretation by cluster =====

n_clusters_final = N_CLUSTERS
cluster_counts = np.bincount(labels_dec, minlength=n_clusters_final)

np.savetxt(
    os.path.join(OUTPUT_DIR, "cluster_counts_spectrogram_k_auto_without_pz.csv"),
    np.column_stack([np.arange(n_clusters_final), cluster_counts]),
    delimiter=",",
    header="cluster,count",
    comments="",
    fmt="%d",
)
print("Saved cluster counts:", os.path.join(OUTPUT_DIR, "cluster_counts_spectrogram_k_auto_without_pz.csv"), flush=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.ravel()

for ax, name in zip(axes, model_channels):
    means, _, _ = cluster_mean_and_std(raw[name], labels_dec, n_clusters_final)
    for cluster_id in range(n_clusters_final):
        ax.plot(sampled_times, means[cluster_id], label=f"C{cluster_id} n={cluster_counts[cluster_id]}")
    ax.set_title(name)
    ax.grid(True)

axes[0].legend(fontsize=8)
fig.suptitle("Cluster mean trajectories: six input variables", fontsize=16)
fig.supxlabel("time")
fig.tight_layout()
save_current_fig("11_cluster_mean_variables_spectrogram_k_auto_without_pz.png")

for name in model_channels:
    means, stds, _ = cluster_mean_and_std(raw[name], labels_dec, n_clusters_final)

    plt.figure(figsize=(10, 5))
    for cluster_id in range(n_clusters_final):
        mean = means[cluster_id]
        std = stds[cluster_id]
        line = plt.plot(sampled_times, mean, label=f"C{cluster_id} n={cluster_counts[cluster_id]}")[0]
        plt.fill_between(
            sampled_times,
            mean - std,
            mean + std,
            color=line.get_color(),
            alpha=0.12,
            linewidth=0,
        )
    plt.xlabel("time")
    plt.ylabel(name)
    plt.title(f"{name}: cluster mean +/- 1 std")
    plt.legend(fontsize=8)
    plt.grid(True)
    save_current_fig(f"cluster_mean_std_{name}_spectrogram_k_auto_without_pz.png")

summary_rows = []
summary_header = ["cluster", "count"]
for name in model_channels:
    summary_header.extend([f"{name}_initial_mean", f"{name}_final_mean", f"{name}_delta_mean", f"{name}_time_mean"])

for cluster_id in range(n_clusters_final):
    mask = labels_dec == cluster_id
    row = [cluster_id, int(mask.sum())]
    for name in model_channels:
        data = raw[name][:, mask]
        if data.shape[1] == 0:
            row.extend([np.nan, np.nan, np.nan, np.nan])
        else:
            initial_mean = data[0].mean()
            final_mean = data[-1].mean()
            delta_mean = (data[-1] - data[0]).mean()
            time_mean_value = data.mean()
            row.extend([initial_mean, final_mean, delta_mean, time_mean_value])
    summary_rows.append(row)

summary_path = os.path.join(OUTPUT_DIR, "cluster_physics_summary_spectrogram_k_auto_without_pz.csv")
np.savetxt(
    summary_path,
    np.array(summary_rows, dtype=float),
    delimiter=",",
    header=",".join(summary_header),
    comments="",
)
print("Saved physics summary:", summary_path, flush=True)

print("Done.", flush=True)
