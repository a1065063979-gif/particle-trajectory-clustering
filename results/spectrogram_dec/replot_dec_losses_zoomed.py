"""Replot saved loss observations with a clearly marked broken linear y-axis."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

folder = Path(__file__).resolve().parent
data = np.genfromtxt(folder / "dec_training_history_report_weight.csv", delimiter=",", names=True)
series = [
    ("total_loss", "total loss", "tab:blue"),
    ("reconstruction_loss", "recon loss", "tab:orange"),
    ("weighted_kl_contribution", "weighted KL contribution", "tab:green"),
]
upper_values = np.concatenate([data["total_loss"], data["reconstruction_loss"]])
assert np.isfinite(upper_values).all() and np.isfinite(data["weighted_kl_contribution"]).all()
assert upper_values.min() > 0.057 and upper_values.max() < 0.064
assert data["weighted_kl_contribution"].min() >= 0 and data["weighted_kl_contribution"].max() < 0.0048

plt.rcParams.update({"font.size": 12, "axes.titlesize": 16, "axes.labelsize": 13})
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 6), gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.10})
for ax in (top, bottom):
    for column, label, color in series:
        ax.plot(data["epoch"], data[column], label=label, color=color, linewidth=1.7)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 355)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
top.set_ylim(0.057, 0.064)
top.set_yticks(np.arange(0.057, 0.0641, 0.001))
bottom.set_ylim(0, 0.0048)
bottom.set_yticks([0, 0.002, 0.004])
top.spines["bottom"].set_visible(False)
bottom.spines["top"].set_visible(False)
top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
for ax, y in ((top, 0), (bottom, 1)):
    ax.plot([0, 1], [y, y], transform=ax.transAxes, marker=[(-1, -0.5), (1, 0.5)],
            markersize=10, linestyle="none", color="black", mec="black", mew=1.2, clip_on=False)
top.set_title("Spectrogram DEC training losses", pad=12)
top.legend(loc="upper right", framealpha=0.95)
bottom.set_xlabel("Epoch")
fig.supylabel("Loss", x=0.02)
fig.subplots_adjust(left=0.115, right=0.98, bottom=0.11, top=0.90)
stem = folder / "07_dec_training_losses_spectrogram_k_auto_without_pz_zoomed"
fig.savefig(stem.with_suffix(".png"), dpi=300)
fig.savefig(stem.with_suffix(".pdf"))
plt.close(fig)
print(f"Saved {stem}.png and .pdf; {len(data)} epochs; no smoothing or data scaling.")
