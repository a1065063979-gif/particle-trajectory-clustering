"""Small synthetic H5 integration checks; no real PIC data or training needed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np


SCRIPT = Path(__file__).with_name("h5_spectrogram_cnn_dec_k_auto_without_pz_report_weight.py")
CHANNELS = ["x", "px", "py", "Ex", "Ey", "Bz"]


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="pic_uniform_sampling_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.h5_path = self.root / "synthetic.h5"
        self.times = np.arange(6010, dtype=np.int64) * 400
        self.ids = np.arange(6, dtype=np.uint64) + np.uint64(2**60)
        self.raw = {}
        with h5py.File(self.h5_path, "w") as handle:
            handle["Times"] = self.times
            handle["unique_Ids"] = self.ids
            t = np.arange(6010, dtype=np.float32)[:, None]
            p = np.arange(6, dtype=np.float32)[None, :]
            for j, channel in enumerate(CHANNELS):
                values = (j + 1) * np.sin(t / 37 + p) + t / 1000 + p * 2
                if channel == "Bz":
                    values = np.full((6010, 6), 0.031, dtype=np.float32)
                self.raw[channel] = values.astype(np.float32)
                handle[channel] = self.raw[channel]

    def run_preprocessing(self, **overrides):
        env = os.environ.copy()
        env.update({
            "H5_PATH": str(self.h5_path), "OUTPUT_DIR": str(self.root / "output"),
            "PREPROCESS_ONLY": "1", "TARGET_TIME_POINTS": "2048", "TIME_STRIDE": "0",
            "STFT_NPERSEG": "256", "STFT_HOP": "128",
            "K_MIN": "2", "K_MAX": "12", "FIGURE2_PARTICLE_INDICES": "1,4",
            "PYTHONIOENCODING": "utf-8",
        })
        env.update(overrides)
        return subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True,
                              text=True, encoding="utf-8", timeout=60)

    def test_uniform_sampling_and_figure_data(self):
        result = self.run_preprocessing()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = self.root / "output"
        meta = json.loads((output / "sampling_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["sampled_time_points"], 2004)
        self.assertEqual(meta["time_stride"], 3)
        self.assertEqual(meta["sampled_time_step"], 1200)
        self.assertEqual(meta["trailing_records_omitted"], 0)
        stem = "figure2_standardized_six_channel_timeseries"
        self.assertTrue((output / f"{stem}.png").read_bytes().startswith(b"\x89PNG"))
        self.assertTrue((output / f"{stem}.pdf").read_bytes().startswith(b"%PDF"))
        with np.load(output / f"{stem}_data.npz") as data:
            indices = np.arange(0, 6010, 3)
            np.testing.assert_array_equal(data["source_time_indices"], indices)
            np.testing.assert_array_equal(np.diff(data["sampled_times"]), 1200)
            np.testing.assert_array_equal(data["unique_ids"], self.ids[[1, 4]])
            self.assertEqual(data["standardized_traces"].shape, (2, 6, 2004))
            for j, name in enumerate(CHANNELS):
                arr = self.raw[name][indices]
                mean = np.float32(arr.mean(dtype=np.float64))
                std = np.float32(arr.std(dtype=np.float64))
                if std < 1e-8:
                    std = np.float32(1)
                expected = ((arr[:, [1, 4]] - mean) / std).T
                np.testing.assert_allclose(data["standardized_traces"][:, j], expected)
        self.assertFalse(list(output.glob("*results.npz")), "Preprocessing must not train.")

    def test_override_does_not_append_irregular_endpoint(self):
        result = self.run_preprocessing(TIME_STRIDE="4")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        meta = json.loads((self.root / "output/sampling_metadata.json").read_text())
        self.assertEqual(meta["sampled_time_points"], 1503)
        self.assertEqual(meta["sampled_time_step"], 1600)
        self.assertEqual(meta["trailing_records_omitted"], 1)

    def test_rejects_nonuniform_source_times(self):
        with h5py.File(self.h5_path, "r+") as handle:
            handle["Times"][3000] += 1
        result = self.run_preprocessing()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Original times are not equally spaced", result.stderr)


if __name__ == "__main__":
    unittest.main()
