import numpy as np
import pytest
import torch

from sdofmv2.core.datamodule import (
    get_dtype_from_precision,
    inverse_log_norm,
    inverse_zscore_norm,
    log_norm,
    min_max_norm,
    zscore_norm,
)


class TestGetDtypeFromPrecision:
    @pytest.mark.parametrize(
        "precision,expected_dtype",
        [
            ("16", torch.float16),
            ("16-mixed", torch.float16),
            ("bf16", torch.bfloat16),
            ("bf16-mixed", torch.bfloat16),
            ("64", torch.float64),
            ("64-true", torch.float64),
            ("32", torch.float32),
            ("default", torch.float32),
        ],
    )
    def test_get_dtype_from_precision(self, precision, expected_dtype):
        result = get_dtype_from_precision(precision)
        assert result == expected_dtype


class TestZscoreNorm:
    def test_zscore_norm_basic(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        normalization_stat = {"AIA": {"171": {"mean": 1.0, "std": 2.0}}}
        result = zscore_norm(data, "AIA", "171", normalization_stat, clip_value=None)
        expected = np.array([[0.0, 0.5], [1.0, 1.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_zscore_norm_with_clipping(self):
        data = np.array([[0.0, 5.0], [10.0, 15.0]])
        normalization_stat = {"AIA": {"171": {"mean": 5.0, "std": 5.0}}}
        result = zscore_norm(data, "AIA", "171", normalization_stat, clip_value=(2.0, 12.0))
        np.testing.assert_array_almost_equal(result[1, 0], 1.4)


class TestMinMaxNorm:
    def test_min_max_norm_basic(self):
        data = np.array([[0.0, 50.0], [100.0, 150.0]])
        normalization_stat = {"AIA": {"171": {"min": 0.0, "max": 150.0}}}
        result = min_max_norm(data, "AIA", "171", normalization_stat)
        expected = np.array([[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestLogNorm:
    def test_log_norm_basic(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        normalization_stat = {"AIA": {"171": {"mean": 0.5, "std": 0.5}}}
        result = log_norm(data, normalization_stat, "AIA", "171", scaler_factor=1.0)
        assert result.shape == data.shape
        assert not np.any(np.isnan(result))

    def test_log_norm_without_scaler(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        normalization_stat = {"AIA": {"171": {"mean": 0.5, "std": 0.5}}}
        result = log_norm(data, normalization_stat, "AIA", "171", scaler_factor=None)
        assert result.shape == data.shape


class TestInverseZscoreNorm:
    def test_inverse_zscore_norm(self):
        data = np.array([[0.0, 0.5], [1.0, 1.5]])
        normalization_stat = {"AIA": {"171": {"mean": 1.0, "std": 2.0}}}
        result = inverse_zscore_norm(data, "AIA", "171", normalization_stat)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestInverseLogNorm:
    def test_inverse_log_norm_basic(self):
        data_transformed = np.array([[0.0, 0.5], [1.0, 1.5]])
        normalization_stat = {"AIA": {"171": {"mean": 0.5, "std": 0.5}}}
        result = inverse_log_norm(
            data_transformed, normalization_stat, "AIA", "171", scaler_factor=1.0
        )
        assert result.shape == data_transformed.shape
        assert np.all(result >= 0)

    def test_inverse_log_norm_without_scaler(self):
        data_transformed = np.array([[0.0, 0.5], [1.0, 1.5]])
        normalization_stat = {"AIA": {"171": {"mean": 0.5, "std": 0.5}}}
        result = inverse_log_norm(
            data_transformed, normalization_stat, "AIA", "171", scaler_factor=None
        )
        assert result.shape == data_transformed.shape
