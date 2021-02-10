from functools import partial
from typing import Optional, Tuple, Union, List

import tensorflow as tf
from tensorflow_addons.image.utils import to_4D_image, from_4D_image
from tensorflow_addons.utils.types import TensorLike, Number


def _clahe(
        image: TensorLike, clip_limit: Number, tile_grid_size: Union[List[int], Tuple[int]]
) -> tf.Tensor:
    """Implements CLAHE as tf ops"""
    original_2d_shape = (tf.shape(image)[0], tf.shape(image)[1])
    original_dtype = image.dtype

    # Need image in int32 format for later gather_nd ops
    image = tf.cast(image, tf.int32)

    tile_shape = tf.truediv(original_2d_shape, tile_grid_size)
    tile_shape = tf.cast(tf.math.ceil(tile_shape), tf.int32)

    # Reflection-pad image
    pad_y = 0
    pad_x = 0

    if original_2d_shape[0] % tile_shape[0] != 0:
        pad_y = tile_shape[0] - (original_2d_shape[0] % tile_shape[0])

    if original_2d_shape[1] % tile_shape[1] != 0:
        pad_x = tile_shape[1] - (original_2d_shape[1] % tile_shape[1])

    image_padded = tf.pad(image, [[0, pad_y], [0, pad_x], [0, 0]], "REFLECT")

    all_tiles = tf.space_to_batch(
        input=tf.expand_dims(image_padded, axis=0),
        block_shape=tile_shape,
        paddings=[[0, 0], [0, 0]],
    )

    # Compute per-tile histogram
    hists = tf.math.reduce_sum(
        tf.one_hot(all_tiles, depth=256, on_value=1, off_value=0, axis=0), axis=1
    )

    if clip_limit > 0:
        clip_limit_actual = tf.cast(
            clip_limit * ((tile_shape[0] * tile_shape[1]) / 256), tf.int32
        )
        clipped_hists = tf.clip_by_value(
            hists, clip_value_min=0, clip_value_max=clip_limit_actual
        )
        clipped_px_count = tf.math.reduce_sum(hists - clipped_hists, axis=0)
        clipped_hists = tf.cast(clipped_hists, tf.float32)
        clipped_px_count = tf.cast(clipped_px_count, tf.float32)
        clipped_hists = clipped_hists + tf.math.truediv(clipped_px_count, 256)
    else:
        clipped_hists = tf.cast(hists, tf.float32)

    cdf = tf.math.cumsum(clipped_hists, axis=0)
    cdf_min = tf.math.reduce_min(cdf, axis=0)

    numerator = cdf - cdf_min
    denominator = tf.cast(tile_shape[0] * tile_shape[1], tf.float32) - cdf_min

    cdf_normalized = tf.round(tf.math.divide_no_nan(numerator, denominator) * (255))
    cdf_normalized = tf.cast(cdf_normalized, tf.int32)

    # Reflection-pad the cdf functions so that we don't have to explicitly deal with corners/edges
    cdf_padded = tf.pad(
        cdf_normalized, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC"
    )

    coords = tf.stack(
        tf.meshgrid(
            tf.range(tf.shape(image_padded)[0]),
            tf.range(tf.shape(image_padded)[1]),
            tf.range(tf.shape(image_padded)[2]),
            indexing="ij",
        )
    )

    y_coords = coords[0, :, :]
    x_coords = coords[1, :, :]
    z_coords = coords[2, :, :]

    half_tile_shape = tf.math.floordiv(tile_shape, 2)

    nw_y_component = tf.math.floordiv(y_coords - half_tile_shape[0], tile_shape[0])
    nw_x_component = tf.math.floordiv(x_coords - half_tile_shape[1], tile_shape[1])

    # Need to correct negative values because negative-indexing for gather_nd ops
    # not supported on all processors (cdf is padded to account for this)
    nw_y_component = nw_y_component + 1
    nw_x_component = nw_x_component + 1

    ne_y_component = nw_y_component
    ne_x_component = nw_x_component + 1

    sw_y_component = nw_y_component + 1
    sw_x_component = nw_x_component

    se_y_component = sw_y_component
    se_x_component = sw_x_component + 1

    def cdf_transform(x_comp, y_comp):
        gatherable = tf.stack([image_padded, y_comp, x_comp, z_coords], axis=-1)
        return tf.cast(tf.gather_nd(cdf_padded, gatherable), tf.float32)

    nw_transformed = cdf_transform(nw_x_component, nw_y_component)
    ne_transformed = cdf_transform(ne_x_component, ne_y_component)
    sw_transformed = cdf_transform(sw_x_component, sw_y_component)
    se_transformed = cdf_transform(se_x_component, se_y_component)

    a = (y_coords - half_tile_shape[0]) % tile_shape[0]
    a = tf.cast(tf.math.truediv(a, tile_shape[0]), tf.float32)
    b = (x_coords - half_tile_shape[1]) % tile_shape[1]
    b = tf.cast(tf.math.truediv(b, tile_shape[1]), tf.float32)

    # Interpolate
    interpolated = (a * (b * se_transformed + (1 - b) * sw_transformed)) + (1 - a) * (
            b * ne_transformed + (1 - b) * nw_transformed
    )

    # Return image to original size and dtype
    interpolated = interpolated[0: original_2d_shape[0], 0: original_2d_shape[1], :]
    interpolated = tf.cast(tf.round(interpolated), original_dtype)

    return interpolated


@tf.function(experimental_compile=True)
def clahe(
        image: TensorLike,
        clip_limit: Number = 4.0,
        tile_grid_size: Union[List[int], Tuple[int]] = (8, 8),
        name: Optional[str] = None,
) -> tf.Tensor:
    """
    Args:
        image: A tensor of shape
            `(num_images, num_rows, num_columns, num_channels)` or
            `(num_rows, num_columns, num_channels)`
        clip_limit: A floating point value or Tensor.
            0 will result in no clipping (AHE only).
            Limits the noise amplification in near-constant regions.
            Default 4.0.
        tile_grid_size: A tensor of shape
            `(tiles_in_x_direction, tiles_in_y_direction)`
            Specifies how many tiles to break the image into.
            Default (8x8).
        name: (Optional) The name of the op. Default `None`.
    Returns:
        Contrast-limited, adaptive-histogram-equalized image
    """
    with tf.name_scope(name or "clahe"):
        image_dims = tf.rank(image)
        image = to_4D_image(image)
        fn = partial(lambda x: _clahe(x, clip_limit, tile_grid_size))
        image = tf.map_fn(fn, image)
        return from_4D_image(image, image_dims)
