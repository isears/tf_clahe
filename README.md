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