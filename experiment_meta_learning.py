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

"""Jaxline meta-learning experiment for functa."""
import sys
from typing import Generator, List, Mapping, Text, Tuple, Union
from absl import app
from absl import flags
from absl import logging
import functools

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils
from ml_collections import config_dict
import optax

from functa import data_utils
from functa import function_reps
from functa import helpers
from functa import minimal_nerf
from functa import pytree_conversions

FLAGS = flags.FLAGS

Array = jnp.ndarray
Batch = Mapping[str, Array]
OptState = optax.OptState
PRNGKey = chex.PRNGKey
Scalars = Mapping[Text, Array]


def get_config():
  """Return config object for training."""
  # Several config settings are defined internally for jaxline
  # use this function to extract them
  config = base_config.get_base_config()

  # These are experiment specific config arguments
  config.experiment_kwargs = config_dict.ConfigDict()
  exp = config.experiment_kwargs.config = config_dict.ConfigDict()

  # Dataset config
  exp.dataset = config_dict.ConfigDict()
  # 'celeb_a_hq_custom' or 'srn_cars'
  exp.dataset.name = 'celeb_a_hq_custom'
  exp.dataset.num_channels = data_utils.DATASET_ATTRIBUTES[
      exp.dataset.name]['num_channels']
  exp.dataset.resolution = data_utils.DATASET_ATTRIBUTES[
      exp.dataset.name]['resolution']
  exp.dataset.type = data_utils.DATASET_ATTRIBUTES[
      exp.dataset.name]['type']

  # Define num_points_per_ray for scene data.
  if exp.dataset.type == 'scene':
    exp.dataset.num_points_per_ray = 32

  # Optimizer config
  exp.opt_inner = config_dict.ConfigDict()
  exp.opt_inner.lr = 1e-2
  exp.opt_outer = config_dict.ConfigDict()
  exp.opt_outer.lr = 3e-6

  # Model config
  exp.model = config_dict.ConfigDict()
  exp.model.type = 'latent_modulated_siren'
  exp.model.w0 = 30.
  exp.model.width = 512
  exp.model.depth = 15
  exp.model.modulate_scale = False
  exp.model.modulate_shift = True
  exp.model.l2_weight = 0.
  exp.model.noise_std = 0.
  # The following three attributes are only used if model.type is
  # 'latent_modulated_siren'
  exp.model.latent_dim = 128
  # Empty tuple below corresponds to a linear map. This always gave better PSNR
  # compared to deeper MLPs.
  exp.model.layer_sizes = ()
  exp.model.latent_init_scale = 0.01
  # The following attributes are only required if using meta-SGD
  exp.model.use_meta_sgd = True
  exp.model.meta_sgd_init_range = (0.005, 0.1)
  exp.model.meta_sgd_clip_range = (0., 1.)

  # Training config
  per_device_batch_size = 1 if exp.dataset.type == 'scene' else 16
  exp.training = config_dict.ConfigDict()
  exp.training.per_device_batch_size = per_device_batch_size
  exp.training.inner_steps = 3
  exp.training.repeat = True
  exp.training.coord_noise = False
  # Define subsampling options for scenes
  if exp.dataset.type == 'scene':
    exp.training.subsample = True
    # Number of rays to subsample per view.
    exp.training.subsample_num_points = 1024
    # Number of views to subsample.
    exp.training.subsample_num_views = 8
  else:
    exp.training.subsample = False

  # Evaluation config
  exp.evaluation = config_dict.ConfigDict()
  exp.evaluation.batch_size = per_device_batch_size
  # Number of examples used for eval logging.
  # Should be small for scenes (e.g. 10).
  # Otherwise set to -1 to evaluate on entire test set.
  exp.evaluation.num_examples = 10 if exp.dataset.type == 'scene' else -1
  exp.evaluation.inner_steps = 3
  exp.evaluation.shuffle = True

  # Training loop config: log and checkpoint every minute.
  config.training_steps = int(5e5)
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60
  config.save_checkpoint_interval = 60
  config.train_checkpoint_all_hosts = False
  config.checkpoint_dir = '/tmp/training/'
  config.eval_specific_checkpoint_dir = '/tmp/training/'

  return config


class Experiment(experiment.AbstractExperiment):
  """Meta-learning experiment."""

  # Holds a map from object properties that will be checkpointed to their name
  # within a checkpoint. Currently it is assumed that these are all sharded
  # device arrays if we use CHECKPOINT_ATTRS. Using NON_BROADCAST assumes we
  # are using a single device
  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_opt_state': 'opt_state',
  }

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""

    super().__init__(mode=mode, init_rng=init_rng)
    self.mode = mode
    self.init_rng = init_rng
    # This config holds all the experiment specific keys defined in get_config
    self.config = config
    self.num_devices = jax.local_device_count()

    # Define model and forward function
    self.forward = hk.without_apply_rng(hk.transform(self._forward_fn))
    # Define coordinate grid of image
    if config.dataset.type == 'image':
      self.coords = function_reps.get_coordinate_grid(config.dataset.resolution)
    elif config.dataset.type == 'scene':
      self.coords = jnp.ones((1, 3))
      render_config = data_utils.DATASET_ATTRIBUTES[
          self.config.dataset.name]['render_config']
      self.render_config = (
          self.config.dataset.num_points_per_ray, render_config['near'],
          render_config['far'], render_config['white_background'])
    else:
      raise f'Unrecognised data type {config.dataset.type}'

    # Inner optimizer is used both for training and validation
    self._opt_inner = optax.sgd(learning_rate=config.opt_inner.lr)

    if self.mode == 'train':
      # Broadcast RNG key so we can use same init on each device
      init_rng = utils.bcast_local_devices(self.init_rng)
      # Initialize parameters on each device using pmap
      self._params = jax.pmap(self.forward.init)(init_rng,
                                                 utils.bcast_local_devices(
                                                     self.coords))
      # Initialize optimizer
      self._opt_outer = optax.adam(learning_rate=config.opt_outer.lr)
      # Only outer optimizer has a state. Optimizer for inner loop is reset at
      # every iteration. We also broadcast optimizer state so each device gets
      # an identical copy
      weights, _ = function_reps.partition_params(self._params)
      self._opt_state = jax.pmap(self._opt_outer.init)(weights)
      # Overwrite update_func method with its pmapped version (note that pmap
      # automatically jits the function). We require an axis name as this will
      # later be used to determine which axis to average the gradients over
      # Note that all arguments will already be batched across devices.
      self._update_func = jax.pmap(self._update_func, axis_name='i')
      # Set up training dataset
      self._train_input = self._build_train_input(
          self.num_devices * self.config.training.per_device_batch_size)
    else:
      self._params = None
      self._opt_state = None
      self._eval_batch = jax.jit(self._eval_batch)

  def _forward_fn(self, coords: Array) -> Array:
    if self.config.model.type == 'modulated_siren':
      model = function_reps.ModulatedSiren(
          width=self.config.model.width,
          depth=self.config.model.depth,
          out_channels=self.config.dataset.num_channels,
          w0=self.config.model.w0,
          modulate_scale=self.config.model.modulate_scale,
          modulate_shift=self.config.model.modulate_shift,
          use_meta_sgd=self.config.model.use_meta_sgd,
          meta_sgd_init_range=self.config.model.meta_sgd_init_range,
          meta_sgd_clip_range=self.config.model.meta_sgd_clip_range)
    elif self.config.model.type == 'latent_modulated_siren':
      model = function_reps.LatentModulatedSiren(
          width=self.config.model.width,
          depth=self.config.model.depth,
          out_channels=self.config.dataset.num_channels,
          w0=self.config.model.w0,
          modulate_scale=self.config.model.modulate_scale,
          modulate_shift=self.config.model.modulate_shift,
          latent_dim=self.config.model.latent_dim,
          layer_sizes=self.config.model.layer_sizes,
          latent_init_scale=self.config.model.latent_init_scale,
          use_meta_sgd=self.config.model.use_meta_sgd,
          meta_sgd_init_range=self.config.model.meta_sgd_init_range,
          meta_sgd_clip_range=self.config.model.meta_sgd_clip_range)
    return model(coords)

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| '__/ _` | | '_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #

  def step(self, global_step, rng, *unused_args, **unused_kwargs):
    """See base class."""
    # rng has shape (num_devices, 2)
    # Get batch of training data.
    # Reshape training data and coords to have batch device dimension.
    per_device_batch_size = self.config.training.per_device_batch_size
    if self.config.dataset.type == 'scene':
      train_batch_dict = next(self._train_input)
      train_batch = train_batch_dict['images']  # [bs, num_views, H, W, C]
      poses = train_batch_dict['poses']  # [bs, num_views, 4, 4]
      focal = train_batch_dict['focal']  # [bs]
      height, width = train_batch.shape[-3], train_batch.shape[-2]
      bs = focal.shape[0]
      all_rays = []
      for i in range(bs):
        rays = minimal_nerf.get_rays_batch(height, width, focal[i],
                                           poses[i])  # [2, num_views, H, W, 3]
        all_rays.append(rays)
      coords = jnp.stack(all_rays)  # [bs, 2, num_views, H, W, 3]
      coords = coords.reshape(
          self.num_devices, per_device_batch_size,
          *coords.shape[1:]
          )  # [num_devices, per_device_bs, 2, num_views, H, W, 3]
    else:
      train_batch_dict = next(self._train_input)
      # [bs, *spatial_dims, len(spatial_dims)]
      train_batch = train_batch_dict['array']
      # self.coords has shape [*spatial_dims, len(spatial_dims)]
      # Stack and reshape as appropriate
      bs = self.num_devices * per_device_batch_size
      coords = jnp.stack([self.coords for _ in range(bs)])
      coords = coords.reshape(
          self.num_devices, per_device_batch_size,
          *self.coords.shape
          )  # [num_devices, per_device_bs, *spatial_dims, len(spatial_dims)]

    # [num_devices, per_device_bs, *spatial_dims, C] where
    # for scenes, *spatial_dims = [num_views, H, W]
    train_batch = train_batch.reshape(
        self.num_devices, per_device_batch_size,
        *train_batch.shape[1:])

    # optionally subsample coordinates for scenes
    if self.config.dataset.type == 'scene' and self.config.training.subsample:
      # flatten [H,W] dims of train_batch & coords for each view
      train_batch = train_batch.reshape(
          self.num_devices, per_device_batch_size,
          train_batch.shape[2], -1, train_batch.shape[-1])
      coords = coords.reshape(
          self.num_devices, per_device_batch_size,
          coords.shape[2], coords.shape[3], -1, coords.shape[-1])
      # Sample views
      sample_view_idx = jax.random.choice(
          utils.get_first(rng), jnp.arange(train_batch.shape[2]),
          (self.config.training.subsample_num_views,))
      # Sample [H,W] indices
      sample_idx = jax.random.choice(
          utils.get_first(rng), jnp.arange(train_batch.shape[3]),
          (self.config.training.subsample_num_points,))
      # Subsample along flattened spatial dimension
      train_batch = train_batch[:, :, sample_view_idx]
      train_batch = train_batch[:, :, :, sample_idx]
      coords = coords[:, :, :, sample_view_idx]
      coords = coords[:, :, :, :, sample_idx]

    # Update model parameters
    self._params, self._opt_state, scalars = (
        self._update_func(self._params, self._opt_state, train_batch,
                          coords, rng))

    # Scalars (and global step) have identical copies stored on each device, so
    # get these from first device (but could have chosen any device) to host
    scalars = utils.get_first(scalars)

    # Print losses, useful for debugging locally
    global_step = utils.get_first(global_step)
    print(f"Step {global_step}: train PSNR {scalars['train_psnr']:.2f}dB")

    return scalars

  def _build_train_input(self, batch_size: int) -> Generator[Array, None, None]:
    """See base class."""
    if self.config.dataset.type == 'image':
      shuffle_buffer_size = 10_000
    elif self.config.dataset.type == 'scene':
      shuffle_buffer_size = 500
    return data_utils.load_dataset(
        self.config.dataset.name,
        'train',
        batch_size=batch_size,
        shuffle=True,
        repeat=self.config.training.repeat,
        shuffle_buffer_size=shuffle_buffer_size)

  def _update_func(self, params: hk.Params, opt_outer_state: OptState,
                   train_batch: Array,
                   coords: Array,
                   rng: PRNGKey) -> Tuple[hk.Params, OptState, Scalars]:
    """Updates meta-learned init of params.

    This method assumes we are given a *batch* of data. This method is run
    individually on each device and does not know about multi-device/pmap.
    This will update weights only, and not modulations.

    Args:
      params:
      opt_outer_state:
      train_batch: Shape (bs, *spatial_dims, channels).
      coords: Shape (bs, *spatial_dims, num_spatial_dims) or flattened
        equivalent. E.g. for images (bs, height, width, 2).
      rng: Random number generator key, shape (2,).

    Returns:
      Updated params, optimization state and scalars (losses).
    """
    # Compute loss and gradients (individually on each device)
    weights, modulations = function_reps.partition_params(params)
    _, model_grad = jax.value_and_grad(self._loss_func)(
        weights, modulations, train_batch, coords, rng)

    # Average gradients across devices (as the _update_func itself is pmapped,
    # it will perform computation in parallel on each separate device. However,
    # as we want to apply the same parameter update for parameters on each
    # device, we need to communicate gradients between devices. This cannot be
    # done with pmap, so need to use jax.lax.pmean to achieve this)
    model_grad = jax.lax.pmean(model_grad, axis_name='i')
    updates, opt_outer_state = self._opt_outer.update(model_grad,
                                                      opt_outer_state)
    # Extract initial modulations (not the fitted ones), since we do not
    # update the meta-learned init
    weights, modulations = function_reps.partition_params(params)
    # Apply updates to weights only
    weights = optax.apply_updates(weights, updates)
    # Merge updated weights with initial (unchanged) modulations
    params = function_reps.merge_params(weights, modulations)
    # Track training PSNR. Need to fit params with inner loop to track training
    # psnr, as the `modulations` above are initial modulations.
    fitted_params, loss = self._fit_params(params, train_batch, coords, rng)
    _, fitted_mods = function_reps.partition_params(fitted_params)
    mods_array = jax.vmap(lambda x: pytree_conversions.pytree_to_array(x)[0])(
        fitted_mods)  # [bs, mod_dim]
    squared_l2_norm = jnp.sum(mods_array**2, axis=-1)  # [bs]
    rec_loss = jnp.mean(loss - self.config.model.l2_weight * squared_l2_norm)
    l2_norm = jnp.mean(jnp.sqrt(squared_l2_norm))
    scalars = {'train_psnr': helpers.psnr_fn(rec_loss), 'mod_l2_norm': l2_norm,
               'rec_loss': rec_loss, 'loss': jnp.mean(loss)}
    scalars = jax.lax.pmean(scalars, axis_name='i')  # Average across devices
    return params, opt_outer_state, scalars

  def _loss_func(self, weights: hk.Params, modulations: hk.Params,
                 train_batch: Array, coords: Array, rng: PRNGKey) -> Array:
    """loss function (which only meta-learns weights, not modulations).

    Taking the gradient with respect to this loss function will backpropagate
    through the entire inner loop.

    Args:
      weights: Model weights shared across train_batch.
      modulations: Modulations specific to each image.
      train_batch: Batch of data.
      coords: Batch of coords.
      rng: Random number generator key.

    Returns:
      loss.
    """
    params = function_reps.merge_params(weights, modulations)
    _, loss = self._fit_params(params, train_batch, coords, rng)
    return jnp.mean(loss)  # Take mean loss across batch

  def _fit_params(self, params: hk.Params, train_batch: Array, coords: Array,
                  rng: PRNGKey) -> Tuple[hk.Params, Array]:
    """Fits params of a model by running inner loop.

    Args:
      params: Model parameters.
      train_batch: Shape (bs, *spatial_dims, channels).
      coords: Shape (bs, *spatial_dims, num_spatial_dims) or flattened
        equivalent. E.g. for images (bs, height, width, 2).
      rng: Shape (2,)

    Returns:
      fitted_params (bs, ...)
      loss (bs)
    """
    # Apply batchified inner loop, so return a *batch* of parameters, one set of
    # parameters for each image.
    # We have a batch of data, so vmap across 0th dimension of data.
    rng = jax.random.split(rng, num=train_batch.shape[0])  # [bs, 2]
    fitted_params, loss, _ = jax.vmap(
        self._inner_loop, in_axes=[None, 0, 0, 0])(params, train_batch, coords,
                                                   rng)
    return fitted_params, loss

  def _inner_loop(self, params: hk.Params,
                  targets: Array, coords: Array,
                  rng: PRNGKey) -> Tuple[hk.Params, Array, Array]:
    """Performs MAML (Finn et al.'17) inner loop and returns all PSNRs.

    This function takes `self.inner_steps` SGD steps in the inner loop to update
    modulations while keeping weights fixed. This function is applied to a
    single image.

    Args:
      params: ModulatedSiren model params.
      targets: Data to be fitted. Shape (*spatial_dims, out_channels).
      coords: Shape (*spatial_dims, num_spatial_dims).
        For images: (height, width, 2) or (height * width, 2).
      rng:

    Returns:
      Updated params, loss, PSNR
    """
    if self.config.dataset.type == 'scene':
      is_nerf = True
      render_config = self.render_config
    else:
      is_nerf = False
      render_config = None
    return helpers.inner_loop(params, self.forward, self._opt_inner,
                              self.config.training.inner_steps, coords,
                              targets,
                              is_nerf=is_nerf,
                              render_config=render_config,
                              l2_weight=self.config.model.l2_weight,
                              noise_std=self.config.model.noise_std,
                              rng=rng,
                              coord_noise=self.config.training.coord_noise)

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #

  def evaluate(self, global_step, rng, **unused_kwargs):
    """See base class."""
    # This step is used to get global_step and scalars to host device when
    # running on multiple devices
    global_step = utils.get_first(global_step)
    log_dict = jax.device_get(self._eval_epoch(rng))
    scalars = log_dict['scalars']

    # Print losses, useful for debugging locally
    if self.config.dataset.type == 'scene':
      key = (f'4_views_novel_val_psnr_'
             f'{str(self.config.evaluation.inner_steps).zfill(2)}')
      print(f'Step {global_step}: 1 view val PSNR {scalars[key]:.2f}dB')
      logging.info('[Step %d] Eval scalars: %s', global_step, scalars[key])
    else:
      print(f"Step {global_step}: val PSNR {scalars['val_psnr']:.2f}dB")
      logging.info('[Step %d] Eval scalars: %s', global_step,
                   scalars['val_psnr'])
    return scalars

  def _eval_inner_loop(
      self, params: hk.Params, image: Array, coords: Array
  ) -> Union[Tuple[hk.Params, Array, List[Array]], Tuple[
      hk.Params, Array, List[Array], List[Array]]]:
    """Performs MAML inner loop and returns all PSNRs.

    Args:
      params: ModulatedSiren model params.
      image: Image to be fitted. Shape (height, width, out_channels).
      coords: Shape (height, width, 2) or (height * width, 2).

    Returns:
      Updated params, loss, all PSNR values.
    """
    if self.config.dataset.type == 'scene':
      is_nerf = True
      render_config = self.render_config
    else:
      is_nerf = False
      render_config = None

    return helpers.inner_loop(
        params,
        self.forward,
        self._opt_inner,
        self.config.evaluation.inner_steps,
        coords,
        image,
        return_all_psnrs=True,
        return_all_losses=True,
        is_nerf=is_nerf,
        render_config=render_config,
        l2_weight=self.config.model.l2_weight)

  def _eval_batch(self, params: hk.Params,
                  val_batch_dict: Mapping[str, Array], rng: Array) -> Scalars:
    """Evaluates a batch."""
    if self.config.dataset.type == 'scene':
      logged_scalars = {}
      novel_view_idx = (23, 55, 82, 119)  # some evenly spread out view indices
      subsample_num_points = self.config.training.subsample_num_points
      # we use the view indices used in PixelNeRF for comparison.
      for context_view_idx in [(63,), (63, 127), (0, 31, 63, 127)]:
        num_context_views = len(context_view_idx)
        images = val_batch_dict[
            'images'][:, context_view_idx]  # [bs, num_context_views, H, W, C]
        poses = val_batch_dict[
            'poses'][:, context_view_idx]  # [bs, num_context_views, 4, 4]
        focal = val_batch_dict['focal']  # [bs]
        height, width = images.shape[-3], images.shape[-2]
        bs = focal.shape[0]
        all_rays = []
        for i in range(bs):
          # [2, num_context_views, H, W, 3]
          rays = minimal_nerf.get_rays_batch(height, width, focal[i], poses[i])
          all_rays.append(rays)
        coords = jnp.stack(all_rays)  # [bs, 2, num_context_views, H, W, 3]
        # Flatten images and coords for subsampling
        images_sub = images.reshape(bs, images.shape[1], -1, images.shape[-1])
        coords_sub = coords.reshape(bs, 2, coords.shape[2], -1,
                                    coords.shape[-1])
        # Sample [H,W] indices
        sample_idx = jax.random.choice(
            rng, jnp.arange(images_sub.shape[2]), (subsample_num_points,))
        images_sub = images_sub[:, :, sample_idx]
        coords_sub = coords_sub[:, :, :, sample_idx]

        out = jax.vmap(
            self._eval_inner_loop,
            in_axes=[None, 0, 0])(params, images_sub, coords_sub)
        # Unpack outputs, which are (in order):
        # - The fitted params per batch example.
        # - loss per batch example.
        # - psnr per batch example at each inner loop iteration.
        # - loss per batch example at each inner loop iteration.
        new_params = out[0]  # Nested dict with values of shape [bs, ...]
        loss = out[1]  # Array shape [bs]
        val_psnrs = out[2]  # List of arrays shape [bs]
        val_losses = out[3]  # List of arrays shape [bs]
        # Record PSNR at every step
        for i in range(len(val_psnrs)):
          idx = str(i).zfill(2)
          logged_scalars.update({
              f'{num_context_views}_views_val_psnr_{idx}': val_psnrs[i],
              f'{num_context_views}_views_loss_{idx}': val_losses[i]
          })
        # Record validation PSNR and loss corresponding to number of inner steps
        # during training.
        num_steps = self.config.training.inner_steps
        logged_scalars.update({
            f'{num_context_views}_views_val_psnr': val_psnrs[num_steps],
            f'{num_context_views}_views_loss': val_losses[num_steps]
        })
        # Render novel views given context
        images = val_batch_dict[
            'images'][:, novel_view_idx]  # [bs, num_novel_views, H, W, C]
        poses = val_batch_dict[
            'poses'][:, novel_view_idx]  # [bs, num_novel_views, 4, 4]
        # [bs, num_novel_views, H, W, 3]
        rgb, _ = jax.vmap(
            minimal_nerf.render_pose,
            in_axes=[None, 0, None, None, 0, 0,
                     None])(self.forward, new_params, height, width, focal,
                            poses, self.render_config)
        loss = jnp.mean((rgb - images)**2, axis=(1, 2, 3, 4))  # [bs]
        val_psnr = helpers.psnr_fn(loss)  # [bs]
        idx = str(self.config.evaluation.inner_steps).zfill(2)
        logged_scalars.update({
            f'{num_context_views}_views_novel_loss_{idx}': loss,
            f'{num_context_views}_views_novel_val_psnr_{idx}': val_psnr
        })
      # Record modulation norms
      _, mods = function_reps.partition_params(new_params)
      mods_array = jax.vmap(lambda x: pytree_conversions.pytree_to_array(x)[0])(
          mods)  # [bs, mod_dim]
      l2_norm = jnp.sqrt(jnp.sum(mods_array**2, axis=-1))  # [bs]
      logged_scalars.update({'mod_l2_norm': l2_norm})
      # Returned scalars will be summed and finally divided by num_samples
      log_dict = {'scalars': logged_scalars}
    else:
      val_batch = val_batch_dict['array']  # [bs, *spatial_dims, C]
      out = jax.vmap(
          self._eval_inner_loop,
          in_axes=[None, 0, None])(params, val_batch, self.coords)
      # Unpack outputs.
      new_params = out[0]  # params with leading dim bs
      loss = out[1]  # Array shape [bs]
      val_psnrs = out[2]  # List of arrays shape [bs]
      val_losses = out[3]
      # Record modulation norms
      scalars = {}
      _, mods = function_reps.partition_params(new_params)
      mods_array = jax.vmap(lambda x: pytree_conversions.pytree_to_array(x)[0])(
          mods)  # [bs, mod_dim]
      l2_norm = jnp.sqrt(jnp.sum(mods_array**2, axis=-1))
      scalars['mod_l2_norm'] = l2_norm  # [bs]
      # Record PSNR and losses at every step
      for i in range(len(val_psnrs)):
        scalars[f'val_psnr_{str(i).zfill(2)}'] = val_psnrs[i]
        scalars[f'loss_{str(i).zfill(2)}'] = val_losses[i]
      # Record validation PSNR corresponding to number of inner steps during
      # training and also loss for each image at the end of inner loop.
      scalars['val_psnr'] = val_psnrs[self.config.training.inner_steps]
      scalars['loss'] = loss
      # Returned scalars will be summed and finally divided by num_samples
      log_dict = {'scalars': scalars}

    return log_dict

  def _build_eval_input(self) -> Generator[Array, None, None]:
    if self.config.dataset.type == 'image':
      shuffle_buffer_size = 10000
    else:
      shuffle_buffer_size = 500
    return data_utils.load_dataset(
        self.config.dataset.name,
        'test',
        batch_size=self.config.evaluation.batch_size,
        shuffle=self.config.evaluation.shuffle,
        num_examples=self.config.evaluation.num_examples,
        shuffle_buffer_size=shuffle_buffer_size)

  def _eval_epoch(self, rng: Array):
    """Evaluates an epoch."""
    num_samples = 0.
    summed_scalars = None
    rng = rng[0]
    params = utils.get_first(self._params)

    for i, val_batch_dict in enumerate(self._build_eval_input()):
      rng, _ = jax.random.split(rng)  # use new rng for each batch.
      if self.config.dataset.type == 'scene':
        views = val_batch_dict['images']  # [bs, 251, H, W, C]
        num_samples += views.shape[0]
        log_dict = self._eval_batch(params, val_batch_dict, rng)
      else:
        val_batch = val_batch_dict['array']
        num_samples += val_batch.shape[0]
        log_dict = self._eval_batch(params, val_batch_dict, rng)
      scalars = log_dict['scalars']

      # Accumulate the sum of scalars for each step
      scalars = jax.tree_map(lambda x: jnp.sum(x, axis=0), scalars)
      if summed_scalars is None:
        summed_scalars = scalars
      else:
        summed_scalars = jax.tree_map(jnp.add, summed_scalars, scalars)
      print(f'{i} eval iterations done')
      logging.info('%d eval iterations done', i)

    mean_scalars = jax.tree_map(lambda x: x / num_samples, summed_scalars)
    return {'scalars': mean_scalars}


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))
