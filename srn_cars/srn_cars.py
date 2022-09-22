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

"""tensorflow dataset (tfds) for srn_cars."""
import imageio
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Convert srn_cars dataset into a tensorflow dataset (tfds).

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """
@article{dupont2020equivariant,
  title={Equivariant Neural Rendering},
  author={Dupont, Emilien and Miguel Angel, Bautista and Colburn, Alex and Sankar, Aditya and Guestrin, Carlos and Susskind, Josh and Shan, Qi},
  journal={arXiv preprint arXiv:2006.07630},
  year={2020}
}
"""


class SrnCars(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for srn_cars dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'images':
                tfds.features.Tensor(shape=(None, 128, 128, 3), dtype=tf.uint8),
            'poses':
                tfds.features.Tensor(shape=(None, 4, 4), dtype=tf.float32),
            'focal':
                tfds.features.Scalar(dtype=tf.float32)
        }),
        supervised_keys=None,  # Set to `None` to disable
        disable_shuffling=True,  # Fix ordering as scenes are already shuffled
        homepage='https://github.com/apple/ml-equivariant-neural-rendering',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract(
        'https://drive.google.com/uc?export=download&confirm=9iBg'
        '&id=19yDsEJjx9zNpOKz9o6AaK-E8ED6taJWU')

    return {
        'train': self._generate_examples(path / 'cars_train'),
        'test': self._generate_examples(path / 'cars_test'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # Coordinate transform matrix required to make SRN poses consistent
    # with other NeRF models
    coord_trans = np.diag(np.array([1., -1., -1., 1.], dtype=np.float32))

    for key, scene_path in enumerate(sorted(path.iterdir())):
      # Load intrinsics
      intrinsics_path = scene_path / 'intrinsics.txt'
      with intrinsics_path.open() as f:
        lines = f.readlines()
        # Extract focal length (required to obtain rays)
        focal = float(lines[0].split()[0])

      # Load images and their associated pose
      pose_dir = scene_path / 'pose'
      rgb_dir = scene_path / 'rgb'
      pose_paths = sorted(list(pose_dir.iterdir()))
      rgb_paths = sorted(list(rgb_dir.iterdir()))

      all_imgs = []
      all_poses = []
      for rgb_path, pose_path in zip(rgb_paths, pose_paths):
        img = imageio.imread(str(rgb_path))[..., :3]
        pose = np.loadtxt(str(pose_path), dtype=np.float32).reshape(4, 4)
        pose = pose @ coord_trans  # Fix coordinate system
        all_imgs.append(img)
        all_poses.append(pose)

      all_imgs = np.stack(all_imgs)
      all_poses = np.stack(all_poses)

      yield key, {
          'images': all_imgs,
          'poses': all_poses,
          'focal': focal,
      }
