"""Correctness tests using temporary synthetic data only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import spectrogram

import run_pca_kmeans as baseline


def fixture():
    rng = np.random.RandomState(10)
    n = 60
    labels = np.arange(n) % 5
    features = rng.normal(size=(n, 32)).astype(np.float32)
    features[:, 0] += labels * 10
    return {"unique_ids": np.arange(n, dtype=np.uint64) + np.uint64(2**63),
            "labels": labels, "latent_vectors": features,
            "channel_names": np.array(baseline.CHANNELS),
            "source_time_indices": np.arange(0, 16, 2), "sampled_times": np.arange(0, 16, 2) * 400,
            "time_mean": np.arange(6, dtype=np.float32), "time_std": np.ones(6, dtype=np.float32) * 2,
            "spec_mean": np.arange(6, dtype=np.float32) / 10, "spec_std": np.ones(6, dtype=np.float32) * 3,
            "stft_nperseg": np.array([4]), "stft_hop": np.array([2]),
            "spec_freq_bins": np.array([2]), "spec_time_bins": np.array([4]), "selected_k": np.array([5])}


class BaselineTests(unittest.TestCase):
    def test_id_order_and_signature(self):
        saved = fixture()
        ids = saved["unique_ids"]
        assert_array_equal(baseline.align_ids(ids, ids[::-1]), np.arange(len(ids))[::-1])
        with self.assertRaises(ValueError):
            baseline.align_ids(ids, np.repeat(ids[:1], len(ids)))
        sig = baseline.preprocessing_signature(saved)
        saved["spec_std"][0] *= 2
        self.assertNotEqual(sig, baseline.preprocessing_signature(saved))

    def test_complete_cli_with_cached_synthetic_input(self):
        saved = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results, cache, output = root / "reference", root / "cache", root / "output"
            results.mkdir()
            cache.mkdir()
            np.savez(results / baseline.FINAL_NAME, **saved)
            # Deliberately reverse initial particle order to test alignment.
            np.savez(results / baseline.INITIAL_NAME, unique_ids=saved["unique_ids"][::-1],
                     latent_vectors=saved["latent_vectors"][::-1], kmeans_labels=saved["labels"][::-1])
            data = np.random.RandomState(1).normal(size=(60, 6, 2, 4)).astype(np.float32)
            data[:, 0] += saved["labels"][:, None, None] * 20
            np.save(cache / "standardized_spectrograms.npy", data)
            (cache / "cache_metadata.json").write_text(json.dumps({
                "signature": baseline.preprocessing_signature(saved), "h5": "synthetic test fixture"}))
            argv = ["run_pca_kmeans.py", "--results-dir", str(results), "--cache-dir", str(cache),
                    "--output-dir", str(output), "--sample-size", "60", "--threads", "1", "--n-init", "2"]
            with patch("sys.argv", argv):
                baseline.main()
            metadata = json.loads((output / "experiment_metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            with np.load(output / "pca_kmeans_results.npz") as result:
                assert_array_equal(result["unique_ids"], saved["unique_ids"])
                self.assertEqual(result["latent_vectors"].shape, (60, 32))
                self.assertEqual(len(np.unique(result["labels"])), 5)
            with np.load(output / "sample_silhouettes.npz") as result:
                assert_allclose(result["AE_KMeans"], result["DEC"], atol=1e-7)
            for name in ("silhouette_comparison.csv", "silhouette_comparison.pdf", "pca_clusters.pdf"):
                self.assertTrue((output / name).is_file())
            # Read-only cached input must remain unchanged by PCA centering.
            assert_array_equal(np.load(cache / "standardized_spectrograms.npy"), data)
            saved["time_mean"][0] += 1
            with self.assertRaises(ValueError):
                baseline.get_spectrogram_cache(root / "missing.h5", cache, saved, 8, 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable locally; run on server for exact preprocessing check")
    def test_cache_matches_original_preprocessing_with_shuffled_h5_ids(self):
        import torch
        import torch.nn.functional as functional
        saved = fixture()
        raw = np.random.RandomState(3).normal(size=(6, 16, 60)).astype(np.float64)
        permutation = np.random.RandomState(4).permutation(60)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with h5py.File(root / "data.h5", "w") as f:
                f["unique_Ids"] = saved["unique_ids"][permutation]
                f["Times"] = np.arange(16) * 400
                for c, name in enumerate(baseline.CHANNELS):
                    f[name] = raw[c][:, permutation]
            cached = baseline.get_spectrogram_cache(root / "data.h5", root / "cache", saved, 7, 1)
            try:
                for c in range(6):
                    arr = raw[c][saved["source_time_indices"]].astype(np.float32)
                    scaled = ((arr - saved["time_mean"][c]) / saved["time_std"][c]).T.astype(np.float32)
                    _, _, sxx = spectrogram(scaled, fs=1.0, nperseg=4, noverlap=2,
                                            scaling="density", mode="magnitude", axis=-1)
                    expected = functional.interpolate(torch.from_numpy(np.log1p(sxx).astype(np.float32)).unsqueeze(1),
                                                      size=(2, 4), mode="bilinear", align_corners=False).squeeze(1).numpy()
                    expected -= saved["spec_mean"][c]
                    expected /= saved["spec_std"][c]
                    assert_allclose(cached[:, c], expected, rtol=0, atol=1e-7)
            finally:
                cached._mmap.close()


if __name__ == "__main__":
    unittest.main()
