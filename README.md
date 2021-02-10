# Tensorflow CLAHE

![Pytest](https://github.com/isears/tf_clahe/workflows/Pytest/badge.svg)

Contrast-limited adaptive histogram equalization implemented in tensorflow ops.

## Setup

```bash
pip install tf_clahe
```

## Use

```python
import tensorflow as tf
import tf_clahe

img = tf.io.decode_image(tf.io.read_file('./path/to/your/img'))
img_clahe = tf_clahe.clahe(img)
```
![sidebyside example](./example.png)

## References

- [S. M. Pizer, E. P. Amburn, J. D. Austin, et al.: Adaptive Histogram Equalization and Its Variations. Computer Vision, Graphics, and Image Processing 39 (1987) 355-368.](https://www.cs.unc.edu/Research/Image/MIDAG/pubs/papers/Adaptive%20Histogram%20Equalization%20and%20Its%20Variations.pdf)
- [Wikipedia: Adaptive Histogram Equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE)