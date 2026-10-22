"""Approximate chart digitization for a scale preview, NOT original training data."""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

folder = Path(__file__).resolve().parent
source = folder / '06_ae_pretraining_loss_spectrogram_k_auto_without_pz.png'
pixels = np.asarray(Image.open(source).convert('RGB')).astype(float)
scale = pixels.shape[1] / 1920
# Calibration from visible tick locations in the source chart.
xzero, xperepoch = 249.5 * scale, (1822 - 249.5) * scale / 200
yhalf, ytenth = 155.5 * scale, 1015.5 * scale
yslope = (ytenth - yhalf) / 0.4
curves = []
for label, color in [('train loss', (31, 119, 180)), ('val loss', (255, 127, 14))]:
    mask = np.linalg.norm(pixels - np.array(color), axis=2) < 35
    mask[:int(120 * scale)] = False
    mask[int(1140 * scale):] = False
    xs, losses = [], []
    for epoch in range(200):
        column = round(xzero + epoch * xperepoch)
        rows = np.flatnonzero(mask[:, column])
        # Exclude legend in the upper right.
        if epoch > 150:
            rows = rows[rows > int(250 * scale)]
        if not len(rows):
            continue
        row = rows.min() if epoch == 0 else np.median(rows)
        loss = 0.5 - (row - yhalf) / yslope
        if loss > 0:
            xs.append(epoch)
            losses.append(loss)
    curves.append((label, color, xs, losses))

plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
for ax in axes:
    for label, color, epochs, losses in curves:
        ax.plot(epochs, losses, color=np.array(color)/255, label=label, linewidth=1.7)
    ax.set_xlabel('Epoch')
    ax.set_xlim(0, 200)
    ax.grid(True, which='major', alpha=0.25)
    ax.legend()
axes[0].set(title='Linear y-axis', ylabel='MSE loss', ylim=(0.04, 0.54))
axes[1].set_yscale('log')
axes[1].set(title='Logarithmic y-axis', ylabel='MSE loss (log scale)', ylim=(0.05, 0.55))
axes[1].yaxis.set_major_locator(FixedLocator([0.05, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5]))
axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
axes[1].yaxis.set_minor_formatter(NullFormatter())
fig.suptitle('AE pretraining loss — approximate preview digitized from the existing PNG', fontsize=13)
path = folder / '06_ae_pretraining_loss_log_scale_PREVIEW.png'
fig.savefig(path, dpi=200)
plt.close(fig)
print(path)
print('Digitized visible samples:', [(label, len(xs)) for label, _, xs, _ in curves])
