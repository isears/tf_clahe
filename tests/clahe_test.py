"""Tests of clahe"""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_addons as tfa

import tf_clahe

_DTYPES = {
    np.uint8,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
}

_SHAPES = {(5, 5), (5, 5, 1), (5, 5, 3), (4, 5, 5), (4, 5, 5, 1), (4, 5, 5, 3)}


@pytest.mark.parametrize("gpu_optimized", [True, False])
def test_clahe_with_equalize(gpu_optimized):
    np.random.seed(0)
    image = np.random.randint(low=0, high=255, size=(5, 100, 100, 3), dtype=np.uint8)
    # CLAHE w/grid size 1x1 and no clip limit should in theory just be global equalization
    clahed = tf_clahe.clahe(image, clip_limit=0, tile_grid_size=(1, 1), gpu_optimized=gpu_optimized)
    equalized = tfa.image.equalize(image)

    # Atol 1 to account for rounding differences between two methods
    np.testing.assert_allclose(clahed, equalized, atol=1)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("gpu_optimized", [True, False])
def test_clahe_dtype_shape(dtype, shape, gpu_optimized):
    image = np.ones(shape=shape, dtype=dtype)
    clahed = tf_clahe.clahe(
        tf.constant(image), clip_limit=2.0, tile_grid_size=(2, 2), gpu_optimized=gpu_optimized
    ).numpy()
    assert clahed.dtype == image.dtype
    assert clahed.shape == image.shape
