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

"""Customized tensorflow dataset (tfds) for celeb_a_hq/128."""
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Convert celeb_a_hq dataset into a tensorflow dataset (tfds).

This is necessary due to issues with the necessary manual preparation of the
celeb_a_hq dataset at:
https://www.tensorflow.org/datasets/catalog/celeb_a_hq
reported in the open issue:
https://github.com/tensorflow/datasets/issues/1496

Note that the resulting dataset has a different ordering to the tfds version,
hence any train/test split further down the line may be different.
Also note that the celeb_a_hq dataset at the given google drive link contains
images at 128 resolution.
"""

_CITATION = """
@article{karras2017progressive,
  title={Progressive growing of gans for improved quality, stability, and variation},
  author={Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  journal={arXiv preprint arXiv:1710.10196},
  year={2017}
}
"""


class CelebAHqCustom(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for custom celeb_a_hq dataset."""

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
            'image':
                tfds.features.Image(shape=(128, 128, 3), dtype=tf.uint8),
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://github.com/tkarras/progressive_growing_of_gans',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract(
        'https://drive.google.com/uc?export=download&confirm=9iBg'
        '&id=107vh6Tibfs1p8pbc3gql-eVwxiqCD2o4')
    # Note that CelebAHQ does not come with a train/test split.
    # This is later split into 90:10 during data processing to create a
    # train/test split.
    path = path / 'data128x128'
    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # The path contains images at 128 resolution.
    for img_path in path.glob('*.jpg'):
      # Yields (key, example)
      yield img_path.name, {'image': img_path}
