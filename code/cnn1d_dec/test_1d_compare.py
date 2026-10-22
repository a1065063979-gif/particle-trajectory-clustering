"""Small CPU tests; all synthetic artifacts are stored in temporary folders."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

import run_1d_compare as experiment


class OneDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_model_shapes_real_length_and_loss_reductions(self):
        model = experiment.Conv1DAutoencoder(2004)
        x = torch.randn(2, 6, 2004)
        recon, z = model(x)
        self.assertEqual(recon.shape, x.shape)
        self.assertEqual(z.shape, (2, 32))
        weights = torch.tensor([1, 1, 1, .2, .2, 1])
        loss = experiment.WeightedMSE(weights)(recon, x)
        expected = sum(weights[c]*(recon[:, c]-x[:, c]).square().mean() for c in range(6))/weights.sum()
        torch.testing.assert_close(loss, expected)
        dec = experiment.DECModel(model, np.zeros((5, 32), dtype=np.float32))
        _, _, q = dec(x)
        p = experiment.target_distribution(q)
        torch.testing.assert_close(q.sum(1), torch.ones(2))
        torch.testing.assert_close(p.sum(1), torch.ones(2))
        cluster = experiment.kl_sum(p, q)
        torch.testing.assert_close(cluster, F.kl_div(q.log(), p, reduction="sum"), atol=1e-6, rtol=1e-5)
        (loss + .0005*cluster).backward()
        self.assertTrue(torch.isfinite(model.fc_enc.weight.grad).all())

    def test_full_training_cache_and_resume(self):
        with tempfile.TemporaryDirectory() as temp, threadpool_limits(limits=1):
            root = Path(temp)
            reference = root / "reference"
            reference.mkdir()
            rng = np.random.RandomState(5)
            n, length = 40, 32
            ids = np.arange(n, dtype=np.uint64)+np.uint64(2**63)
            idx = np.arange(0, length*2, 2)
            y = np.arange(n) % 5
            raw = rng.normal(size=(6, length*2, n)).astype(np.float64) + y[None, None, :]*2
            saved = {"unique_ids": ids, "source_time_indices": idx, "sampled_times": idx*400,
                     "channel_names": np.array(experiment.CHANNELS), "time_mean": np.arange(6, dtype=np.float32),
                     "time_std": np.ones(6, dtype=np.float32)*2,
                     "selected_k": np.array([5]), "latent_vectors": rng.normal(size=(n, 32)).astype(np.float32),
                     "labels": y, "reconstruction_channel_weights": np.array([1, 1, 1, .2, .2, 1], dtype=np.float32)}
            np.savez(reference / experiment.REFERENCE_NAME, **saved)
            permutation = rng.permutation(n)
            h5_path = root / "data.h5"
            with h5py.File(h5_path, "w") as f:
                f["Times"] = np.arange(length*2)*400
                f["unique_Ids"] = ids[permutation]
                for c, channel in enumerate(experiment.CHANNELS):
                    f[channel] = raw[c][:, permutation]
            base = ["run_1d_compare.py", "--results-dir", str(reference), "--h5", str(h5_path),
                    "--cache-dir", str(root / "cache"), "--device", "cpu", "--threads", "1", "--ae-epochs", "2",
                    "--batch-size", "8", "--n-init", "2", "--sample-size", "40", "--monitor-sample-size", "40"]
            out = root / "resumed"
            with patch("sys.argv", base+["--output-dir", str(out), "--dec-epochs", "2"]):
                experiment.main()
            cache = np.load(root / "cache" / "standardized_timeseries.npy")
            expected = np.stack([((raw[c][idx].astype(np.float32)-saved["time_mean"][c])/saved["time_std"][c]).T for c in range(6)], axis=1)
            assert_array_equal(cache, expected)
            with patch("sys.argv", base+["--output-dir", str(out), "--dec-epochs", "3", "--resume", str(out / "dec_last.pt")]):
                experiment.main()
            continuous = root / "continuous"
            with patch("sys.argv", base+["--output-dir", str(continuous), "--dec-epochs", "3"]):
                experiment.main()
            a = torch.load(out / "dec_last.pt", map_location="cpu", weights_only=True)
            b = torch.load(continuous / "dec_last.pt", map_location="cpu", weights_only=True)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            assert_allclose(a["history"], b["history"], rtol=0, atol=0)
            for row in a["history"]:
                self.assertAlmostEqual(row[1]+.0005*row[3], row[1]+row[4])
                self.assertAlmostEqual(row[1], row[2]+row[4], places=6)
            metadata = json.loads((out / "experiment_metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["dec_epochs_executed"], 3)
            result = experiment.npz_read(out / "h5_1d_cnn_dec_results.npz")
            assert_array_equal(result["unique_ids"], ids)
            self.assertEqual(result["latent_vectors"].shape, (n, 32))
            self.assertTrue((out / "silhouette_comparison.csv").exists())
            self.assertTrue((out / "final_latent_pca.pdf").exists())
            # Changed normalization must invalidate a cache, not silently reuse it.
            saved["time_mean"][0] += 1
            with self.assertRaises(ValueError):
                experiment.time_cache(h5_path, root / "cache", saved)


if __name__ == "__main__":
    unittest.main()
