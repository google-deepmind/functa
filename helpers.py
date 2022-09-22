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

"""Helper functions."""
from typing import List, Optional, Tuple, Union

from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from functa import function_reps
from functa import pytree_conversions
from functa.minimal_nerf import render_rays

Array = jnp.ndarray
PRNGKey = chex.PRNGKey

# Helper functions to compute MSE and PSNR
mse_fn = jax.jit(lambda x, y: jnp.mean((x - y)**2))
psnr_fn = jax.jit(lambda mse: -10 * jnp.log10(mse))
inverse_psnr_fn = jax.jit(lambda psnr: jnp.exp(-psnr*jnp.log(10) / 10))


def loss_fn_image(modulations: hk.Params, weights: hk.Params, model,
                  image: Array, coords: Array, l2_weight: float) -> Array:
  """Loss function for images.

  Args:
    modulations: Modulation parameters.
    weights: Shared weight parameters.
    model: Haiku transformed model.
    image: Shape (height, width, channels).
    coords: Shape (height, width, 2) or (height * width, 2). Note the coords
      will be flattened in model call.
    l2_weight: weight for L2 regularisation of modulations.

  Returns:
    MSE between ground truth image and image reconstructed by function rep.
  """
  params = function_reps.merge_params(weights, modulations)
  generated = model.apply(params, coords)
  modulations_array, _, _ = pytree_conversions.pytree_to_array(modulations)
  l2_loss = l2_weight * jnp.sum(modulations_array**2)
  rec_loss = mse_fn(generated, image)
  return rec_loss + l2_loss, rec_loss


def loss_fn_nerf(modulations: hk.Params, weights: hk.Params, model,
                 target: Array, rays: Array,
                 render_config: Tuple[int, float, float, bool],
                 l2_weight: float, rng: Union[int, PRNGKey] = 42,
                 coord_noise: bool = False):
  """Loss function for scenes.

  Args:
    modulations: Modulation parameters.
    weights: Shared weight parameters.
    model: Haiku transformed model.
    target: Target pixel values for a single or a batch of images
      *of the same scene*. Shape (H, W, 3) or (num_views, H, W, 3).
    rays: Ray origin and direction for each target value.
      Shape (2, H, W, 3) or (2, num_views, H, W, 3).
    render_config: config for nerf.
    l2_weight: weight for L2 regularisation of modulations.
    rng: PRNG key for adding coordinate noise.
    coord_noise: whether to add coordinate noise or not.

  Returns:
    loss: scalar MSE between ground truth view and image reconstructed by
      function rep.
  """
  params = function_reps.merge_params(weights, modulations)
  rgb, _ = render_rays(model, params, rays, render_config, rng, coord_noise)
  modulations_array, _, _ = pytree_conversions.pytree_to_array(modulations)
  l2_loss = l2_weight * jnp.sum(modulations_array**2)
  rec_loss = mse_fn(rgb, target)
  return rec_loss + l2_loss, rec_loss


def inner_loop(
    params: hk.Params,
    model,
    opt_inner: optax.GradientTransformation,
    inner_steps: int,
    coords: Array,
    targets: Array,
    return_all_psnrs: bool = False,
    return_all_losses: bool = False,
    is_nerf: bool = False,
    render_config: Optional[Tuple[int, float, float, bool]] = None,
    l2_weight: float = 0.,
    noise_std: float = 0.,
    rng: Union[int, PRNGKey] = 42,
    coord_noise: bool = False,
) -> Union[Tuple[hk.Params, Array, Array], Tuple[
    hk.Params, Array, Array, List[Array]], Tuple[hk.Params, Array, List[Array]],
           Tuple[hk.Params, Array, List[Array], List[Array]]]:
  """Performs MAML (Finn et al.'17) inner loop: fits modulations to target data.

  This function takes `inner_steps` SGD steps in the inner loop to fit
  modulations to image, while keeping weights fixed. This function is applied
  to a single target (e.g. image, video or 3d scene).

  Args:
    params: ModulatedSiren model params.
    model: Haiku transformed model.
    opt_inner: Optax optimizer (typically SGD).
    inner_steps: Number of SGD steps to take to fit modulations to image.
    coords: Coordinates at which function rep will be evaluated.
    targets: Data to be fitted. Not batched. For example, a single image of
      shape (height, width, 3).
    return_all_psnrs: If True, returns a list of PSNRs at every step during
      fitting, otherwise returns only final PSNR.
    return_all_losses: If True, returns a list of losses at every step during
      fitting. Only comes into effect when return_all_psnrs=True.
    is_nerf: If True, uses nerf inner loop.
    render_config: config for nerf.
    l2_weight: weight for L2 regularisation of modulations.
    noise_std: standard deviation of Gaussian noise applied to modulations.
    rng:
    coord_noise: whether to add coordinate noise or not. Only used if
      `is_nerf=True`.

  Returns:
    Fitted params, loss and either final PSNR or all PSNR values.
  """
  if isinstance(rng, int):
    rng = jax.random.PRNGKey(rng)
  # Partition params into trainable modulations and non-trainable weights
  weights, modulations = function_reps.partition_params(params)

  # Check if 'meta_sgd_lrs' is inside a key in weights. If it is, use meta-SGD
  # to fit the data
  use_meta_sgd = False
  for key in weights:
    if 'meta_sgd_lrs' in key:
      use_meta_sgd = True

  if use_meta_sgd:
    # Extract learning rates
    _, lrs = function_reps.partition_shared_params(weights)
    # Flatten lrs so they can easily be multiplied with modulations when
    # performing meta-SGD update
    flat_lrs, _, _ = pytree_conversions.pytree_to_array(lrs)

  # Inner optimizer should have no memory of its state, every time we do inner
  # loop optimization we are solving a new problem from scratch, so optimizer
  # should be reinitialized. As we only update modulations with opt_inner,
  # initialize with modulations and not all params
  # Only use optimizer if we are not using meta-SGD (where we learn learning
  # rates per parameter)
  if not use_meta_sgd:
    opt_inner_state = opt_inner.init(modulations)

  # Optionally store PSNR at every step
  if return_all_psnrs:
    psnr_vals = []

  if return_all_losses:
    loss_vals = []

  # Only update modulations in inner loop
  for _ in range(inner_steps):
    # jax.grad takes gradient with respect to first positional argument only
    if is_nerf:
      (loss, rec_loss), modulations_grad = jax.value_and_grad(
          loss_fn_nerf, has_aux=True)(modulations, weights, model, targets,
                                      coords, render_config, l2_weight,
                                      rng, coord_noise)
    else:
      (loss, rec_loss), modulations_grad = jax.value_and_grad(
          loss_fn_image, has_aux=True)(modulations, weights, model, targets,
                                       coords, l2_weight)
    # Update modulations
    if use_meta_sgd:
      # modulations_grad is a pytree with the same keys as modulations. lrs is
      # a pytree containing all learning rates as a single array in a single
      # leaf. Flatten both to multiply them together and then reconstruct tree
      # Note, learning rate flattening operation is done above, and we therefore
      # apply flat_lrs here
      # Note, the following two lines are awkward, but are required to satisfy
      # linter (line-too-long).
      out = pytree_conversions.pytree_to_array(modulations_grad)
      flat_modulations_grads, concat_idx, tree_def = out
      flat_modulations_updates = -flat_lrs * flat_modulations_grads
      modulation_updates = pytree_conversions.array_to_pytree(
          flat_modulations_updates, concat_idx, tree_def)
    else:
      modulation_updates, opt_inner_state = opt_inner.update(
          modulations_grad, opt_inner_state)
    # Apply gradient update
    modulations = optax.apply_updates(modulations, modulation_updates)

    # Optionally calculate PSNR value
    if return_all_psnrs:
      psnr_vals.append(psnr_fn(rec_loss))
    if return_all_losses:
      loss_vals.append(loss)

  # Optionally add noise to fitted modulations, to make downstream task less
  # sensitive to exact value of modulations.
  if noise_std > 0.:
    modulations_array, concat_idx, tree_def = pytree_conversions.pytree_to_array(
        modulations)
    modulations_array += noise_std * jax.random.normal(
        rng, shape=modulations_array.shape)
    modulations = pytree_conversions.array_to_pytree(modulations_array,
                                                     concat_idx, tree_def)

  # Compute final loss using updated modulations
  if is_nerf:
    loss, rec_loss = loss_fn_nerf(modulations, weights, model, targets, coords,
                                  render_config, l2_weight, rng, coord_noise)
  else:
    loss, rec_loss = loss_fn_image(modulations, weights, model, targets, coords,
                                   l2_weight)

  total_loss = loss

  if return_all_psnrs:
    psnr_vals.append(psnr_fn(rec_loss))

  if return_all_losses:
    loss_vals.append(loss)

  # Merge weights and modulations and return
  params = function_reps.merge_params(weights, modulations)

  if return_all_psnrs and not return_all_losses:
    return params, total_loss, psnr_vals
  elif return_all_psnrs and return_all_losses:
    return params, total_loss, psnr_vals, loss_vals
  else:
    return params, total_loss, psnr_fn(rec_loss)


def image_grid_from_batch(images: Array) -> Array:
  """Simple helper to generate a single image from a mini batch.

  Args:
    images: Batch of images of shape (batch_size, height, width, channels)

  Returns:
    A single image of shape (img_grid_height, img_grid_width, channels).
  """
  batch_size = images.shape[0]
  grid_size = int(np.floor(np.sqrt(batch_size)))

  img_iter = iter(images[0:grid_size**2])
  return jnp.squeeze(
      jnp.vstack([
          jnp.hstack([next(img_iter)
                      for _ in range(grid_size)][::-1])
          for _ in range(grid_size)
      ]))


def log_params_info(params):
  """Log information about parameters."""
  logging.info('Parameter shapes')
  logging.info(jax.tree_map(jnp.shape, params))
  num_params = hk.data_structures.tree_size(params)
  byte_size = hk.data_structures.tree_bytes(params)
  logging.info('%d params, size: %.2f MB', num_params, byte_size / 1e6)
  # print each parameter and its shape
  for mod, name, value in hk.data_structures.traverse(params):
    logging.info('%s/%s: %s', mod, name, value.shape)
