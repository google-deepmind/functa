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

"""Create modulation dataset for celeba."""

import os

from absl import app
from absl import flags
import dill
import haiku as hk
import numpy as np
import optax

from functa import data_utils
from functa import function_reps
from functa import helpers
from functa import pytree_conversions

flags.DEFINE_integer('mod_dim', 64,
                     'The dimensionality of modulation dimension to use.'
                     'Choose one of: 64, 128, 256, 512, 1024.')
flags.DEFINE_string('pretrained_weights_dir', '',
                    'Path to directory containing pre-trained weights.')
flags.DEFINE_string('save_to_dir', '',
                    'Path to directory where modulations should be saved.')

FLAGS = flags.FLAGS


# Define function that creates a dict of modulations & psnrs for each dataset
def create_modulation_dataset(model, params, ds, num_steps, coords, lr,
                              l2_weight, noise_std):
  """Creates a dataset of modulations and corresponding psnr values.

  Args:
    model: Haiku transformed model that outputs rgb given 2d pixel coord inputs.
    params: Parameters of ModulatedSiren or LatentModulatedSiren model.
    ds: Dataset iterator that gives a single image at each iteration.
    num_steps: Number of SGD steps to use for fitting each modulation.
    coords: 2D pixel coordinates of shape (H, W, 2).
    lr: Learning rate of SGD optimizer.
    l2_weight: Weight for L2 regularisation of modulations.
    noise_std: standard deviation of Gaussian noise applied to modulations.

  Returns:
    mod_data: Array of modulations shape (data_size, mod_dim).
    psnr_vals: Array of psnrs shape (data_size,).
    psnr_mean: psnr corresponding to the mean rec loss across the dataset.
  """
  # Define sgd optimizer that carries out 3 gradient steps wrt modulations
  opt_inner = optax.sgd(lr)
  mod_list = []
  psnr_list = []
  rec_loss_list = []
  for i, datum in enumerate(ds):
    fitted_params, _, psnr = helpers.inner_loop(
        params=params,
        model=model,
        opt_inner=opt_inner,
        inner_steps=num_steps,
        coords=coords,
        targets=datum['array'],
        return_all_psnrs=False,
        return_all_losses=False,
        l2_weight=l2_weight,
        noise_std=noise_std)
    rec_loss = helpers.inverse_psnr_fn(psnr)
    _, modulations = function_reps.partition_params(fitted_params)
    modulations, _, _ = pytree_conversions.pytree_to_array(modulations)
    mod_list.append(modulations)
    psnr_list.append(psnr)
    rec_loss_list.append(rec_loss)
    print(f'data point {(i+1):5d} has psnr {psnr:2.2f} dB')
  mod_data = np.stack(mod_list)  # [num_data, mod_dim]
  psnr_vals = np.array(psnr_list)  # [num_data]
  rec_losses = np.array(rec_loss_list)  # [num_data]
  mean_rec_loss = np.mean(rec_losses)
  psnr_mean = helpers.psnr_fn(mean_rec_loss)
  return mod_data, psnr_vals, psnr_mean


def main(_):
  # Load params of LatentModulatedSiren model
  ## Define path to checkpoint, downloaded from codebase
  mod_dim = FLAGS.mod_dim  # choose one of 64, 128, 256, 512, 1024
  assert mod_dim in [
      64, 128, 256, 512, 1024
  ], f'`mod_dim` should be one of [64, 128, 256, 512, 1024], got {mod_dim}'
  path = os.path.join(FLAGS.pretrained_weights_dir,
                      f'celeba_params_{mod_dim}_latents.npz')
  ## Check that checkpoint file exists
  assert os.path.exists(path), 'Pretrained weights file does not exist.'
  with open(path, 'rb') as f:
    ckpt = dill.load(f)
  params = ckpt['params']
  config = ckpt['config']
  assert config['model']['type'] == 'latent_modulated_siren'
  print(f'Loaded params for model with {mod_dim} latent dimensions.')
  ## Create haiku transformed model that runs the forward pass.
  ## Only keep configs needed for model construction from model config
  ## `None` below ensures no error is given when already removed
  model_config = config['model'].copy()
  model_config.pop('type', None)
  model_config.pop('l2_weight', None)
  model_config.pop('noise_std', None)
  def model_net(coords):
    hk_model = function_reps.LatentModulatedSiren(
        out_channels=config['dataset']['num_channels'], **model_config)
    return hk_model(coords)
  model = hk.without_apply_rng(hk.transform(model_net))

  # Check that user specified directory exists if specified
  if FLAGS.save_to_dir:
    assert os.path.isdir(
        FLAGS.save_to_dir
    ), f'User specified directory {FLAGS.save_to_dir} does not exist.'

  # Setup celeba dataset
  train_ds = data_utils.load_dataset('celeb_a_hq_custom', subset='train')
  test_ds = data_utils.load_dataset('celeb_a_hq_custom', subset='test')

  # Iterate across training set to produce train modulations
  train_mod_data, train_psnr_vals, train_psnr_mean = create_modulation_dataset(
      model=model,
      params=params,
      ds=train_ds,
      num_steps=config['training']['inner_steps'],
      coords=function_reps.get_coordinate_grid(config['dataset']['resolution']),
      lr=config['opt_inner']['lr'],
      l2_weight=config['model']['l2_weight'],
      noise_std=config['model']['noise_std'],
  )
  print(f'Training set psnr: {train_psnr_mean}')

  # Repeat with test set
  test_mod_data, test_psnr_vals, test_psnr_mean = create_modulation_dataset(
      model=model,
      params=params,
      ds=test_ds,
      num_steps=config['training']['inner_steps'],
      coords=function_reps.get_coordinate_grid(config['dataset']['resolution']),
      lr=config['opt_inner']['lr'],
      l2_weight=config['model']['l2_weight'],
      noise_std=config['model']['noise_std'],
  )
  print(f'Test set psnr: {test_psnr_mean}')

  # Save modulations to user specified directory
  train_dict = dict(modulation=train_mod_data, psnr=train_psnr_vals)
  test_dict = dict(modulation=test_mod_data, psnr=test_psnr_vals)
  modulation_data = dict(train=train_dict, test=test_dict)
  path = os.path.join(FLAGS.save_to_dir,
                      f'celeba_modulations_{mod_dim}_latents.npz')
  with open(path, 'wb') as f:
    dill.dump(modulation_data, f)


if __name__ == '__main__':
  app.run(main)
