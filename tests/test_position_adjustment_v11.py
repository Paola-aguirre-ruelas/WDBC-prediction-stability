"""Focused checks independent of the existing expensive RF fitting pipeline."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_probability_position_v11.py"
spec = importlib.util.spec_from_file_location("v11", SCRIPT)
v11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v11)


class PositionTests(unittest.TestCase):
    def test_rank_residualization_removes_linear_rank_relation(self):
        x = np.arange(1.0, 100.0)
        residual = v11.residualize_rank(x, x)
        np.testing.assert_allclose(residual, 0.0, atol=1e-11)

    def test_rolling_excess_preserves_ids_and_has_zero_excess_for_constant_width(self):
        df = pd.DataFrame({
            "observation_id": [f"WDBC_{i:03d}" for i in range(11)],
            "position_geometry": np.linspace(0.01, 0.25, 11),
            "resampling_width": np.repeat(0.1, 11),
        })
        result = v11.rolling_excess(df, 5)
        self.assertEqual(len(result), 11)
        self.assertEqual(set(result.observation_id), set(df.observation_id))
        np.testing.assert_allclose(result.excess_width, 0.0)

    def test_spearman_ties(self):
        a = np.array([1, 1, 2, 2, 3, 3])
        self.assertAlmostEqual(v11.spearman(a, a), 1.0)


if __name__ == "__main__":
    unittest.main()
