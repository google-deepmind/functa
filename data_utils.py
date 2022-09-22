# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils for loading and processing datasets."""

from typing import Mapping, Optional
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


Array = jnp.ndarray
Batch = Mapping[str, np.ndarray]
DATASET_ATTRIBUTES = {
    'celeb_a_hq_custom': {
        'num_channels': 3,
        'resolution': 64,
        'type': 'image',
        'train_size': 27_000,
        'test_size': 3_000,
    },
    'srn_cars': {
        'num_channels': 4,
        'resolution': 128,
        'type': 'scene',
        'render_config': {
            'near': 0.8,
            'far': 1.8,
            'white_background': True,
        },
        'train_size': 2458,
        'test_size': 703,
    },
}


def load_dataset(dataset_name: str,
                 subset: str,
                 batch_size: Optional[int] = None,
                 shuffle: bool = False,
                 repeat: bool = False,
                 num_examples: Optional[int] = None,
                 shuffle_buffer_size: int = 10000):
  """Tensorflow dataset loaders.

  Args:
    dataset_name (string): One of elements of DATASET_NAMES.
    subset (string): One of 'train', 'test'.
    batch_size (int):
    shuffle (bool): Whether to shuffle dataset.
    repeat (bool): Whether to repeat dataset.
    num_examples (int): If not -1, returns only the first num_examples of the
      dataset.
    shuffle_buffer_size (int): Buffer size to use for shuffling dataset.

  Returns:
    Tensorflow dataset iterator.
  """

  # Load dataset
  if dataset_name.startswith('srn'):
    ds = tfds.load(dataset_name, split=subset)
    # Filter corrupted scenes that contain views with only white background
    ds = ds.filter(filter_srn)
    # Map pixels in [0,255] to [0,1] range
    ds = ds.map(process_srn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  elif dataset_name.startswith('celeb_a_hq'):
    # CelebAHQ does not have a test dataset, so do a 90/10
    # split on training data to create train and test sets
    if subset == 'train':
      subset = 'train[:90%]'
    elif subset == 'test':
      subset = 'train[90%:]'
    ds = tfds.load(dataset_name, split=subset)
    # Map pixels in [0,255] to [0,1] range and map resolution from 128 to 64.
    ds = ds.map(process_celeba,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Optionally subsample dataset
  if num_examples is not None:
    ds = ds.take(num_examples)

  # Optionally shuffle dataset
  if shuffle:
    ds = ds.shuffle(shuffle_buffer_size)

  # Optionally repeat dataset if repeat
  if repeat:
    ds = ds.repeat()

  if batch_size is not None:
    ds = ds.batch(batch_size)

  # Convert from tf.Tensor to numpy arrays for use with Jax
  return iter(tfds.as_numpy(ds))


def filter_srn(batch):
  views = batch['images']   # shape [num_views, H, W, 3]
  # Take min and max for each view
  min_val = tf.math.reduce_min(views, axis=(1, 2, 3))  # [num_views]
  max_val = tf.math.reduce_max(views, axis=(1, 2, 3))  # [num_views]
  # Take the difference then the minimum across views
  # Some views have only white background iff this min == 0
  min_diff = tf.math.reduce_min(max_val - min_val)  # scalar
  return tf.math.not_equal(min_diff, 0)


def process_srn(batch: Batch):
  batch['images'] = tf.cast(batch['images'], tf.float32) / 255.
  return batch


def process_celeba(batch: Batch):
  image = tf.cast(batch['image'], tf.float32) / 255.
  # Resize from 128 to 64 resolution.
  image = tf.image.resize(image, [64, 64])  # [64, 64, 3]
  return {'array': image}





