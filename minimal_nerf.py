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

"""Minimal NeRF implementation.

A simplified version of:
 - https://github.com/tancik/learnit/blob/main/Experiments/shapenet.ipynb
 - https://github.com/bmild/nerf/blob/master/tiny_nerf.ipynb
"""
import functools
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp


Array = jnp.ndarray
PRNGKey = chex.PRNGKey

MAX_DENSITY = 10.


def get_rays(height: int, width: int, focal: float, pose: Array):
  """Converts pose information to ray origins and directions for NeRF.

  Args:
    height: Height of image.
    width: Width of image.
    focal: Focal length.
    pose: Pose (camera to world matrix) of shape (4, 4).

  Returns:
    Rays array of shape (2, H, W, 3), where rays[0] corresponds to ray
    origins and rays[1] to ray directions.
  """
  i, j = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing='xy')
  # use pixel center coordinates instead of corner coordinates.
  extra_shift = .5
  dirs = jnp.stack([(i - width * .5 + extra_shift) / focal,
                    -(j - height * .5 + extra_shift) / focal,
                    -jnp.ones_like(i)], -1)
  rays_d = jnp.sum(dirs[..., jnp.newaxis, :] * pose[:3, :3], -1)
  rays_o = jnp.broadcast_to(pose[:3, -1], rays_d.shape)
  return jnp.stack([rays_o, rays_d], 0)  # (2, H, W, 3)


# This batched function will output arrays of shape (2, B, H, W, 3) since we use
# out_axes=1 (i.e. batching is over 1st dimension *not* 0th dimension). Note
# that this is all for a *single scene*.
get_rays_batch = jax.vmap(get_rays, in_axes=[None, None, None, 0], out_axes=1)


def volumetric_rendering(rgb: Array, density: Array, z_vals: Array,
                         rays_d: Array, white_background: bool):
  """Volumetric rendering.

  Args:
    rgb: rgb at 3D coordinates. Array shape (..., num_points_per_ray, 3).
    density: density at 3D coordinates. Array shape (..., num_points_per_ray).
    z_vals: distances to 3D coordinates from ray origin.
      Array shape (..., num_points_per_ray).
    rays_d: ray directions. Array shape (..., 3)
    white_background: If True sets default RGB value to be 1, otherwise will be
      set to 0 (black).

  Returns:
    rgb_map: Rendered view(s). Array of shape (..., 3).
    depth_map: Depth map of view(s). Array of shape (...).
    weights: Weights for rendering rgb_map from rgb values.
      Array of shape (..., num_points_per_ray).
  """

  # Calculate distance between consecutive points along ray.
  distance_between_points = z_vals[..., 1:] - z_vals[..., :-1]
  # The following line is a slightly convoluted way of adding a single extra
  # element to the distances array (since we made it 1 element shorter than
  # full ray). This will now have the same length as opacities.
  distances = jnp.concatenate([
      distance_between_points,
      1e-3 * jnp.ones_like(distance_between_points[..., :1])
  ], -1)  # (..., num_points_per_ray)
  # Correct distances by magnitude of ray direction
  distances = distances * jnp.linalg.norm(rays_d[..., None, :], axis=-1)

  # Alpha will have a value between 0 and 1
  alpha = 1. - jnp.exp(-density * distances)  # (..., num_points_per_ray)
  # Ensure transmittance is <= 1 (and greater than 1e-10)
  trans = jnp.minimum(1., 1. - alpha + 1e-10)

  # Make the first transmittance value along the ray equal to 1 for every ray
  trans = jnp.concatenate([jnp.ones_like(trans[..., :1]), trans[..., :-1]],
                          -1)  # (..., num_points_per_ray)
  cum_trans = jnp.cumprod(trans, -1)  # T_i in Equation (3) of Nerf paper.
  weights = alpha * cum_trans  # (..., num_points_per_ray)

  # Sum RGB values along the ray
  rgb_map = jnp.sum(weights[..., None] * rgb, -2)  # (..., 3)
  # Optionally make background white
  if white_background:
    acc_map = jnp.sum(weights, -1)  # Accumulate weights   (...)
    rgb_map = rgb_map + (1. - acc_map[..., None])  # Add white background
  # Weigh distance along ray to get depth - weighted average of distances to
  # points on ray
  depth_map = jnp.sum(weights * z_vals, -1)
  return rgb_map, depth_map, weights


def render_rays(model, params, rays: Array,
                render_config: Tuple[int, float, float, bool],
                rng: Union[int, PRNGKey] = 42, coord_noise: bool = False):
  """Renders rays through model of a single scene (with possibly many views).

  Args:
    model: Haiku transformed model, with input_size = 3, output_size = 4 (3
      for RGB and 1 for density.)
    params: Model params.
    rays: Array of shape (2, ..., 3) containing ray origin and ray direction.
      This is quite similar to coords in our other models. The ellipsis refers
      to spatial dimensions and optional batch dimensions when using multiple
      views. E.g. for a single view (H, W) or (H*W) and for B views (B, H, W)
      or (B, H*W) or (B*H*W). Note that these can also be subsamples.
    render_config: Tuple containing rendering configuration for NeRF.
      This includes the following:
        - num_points_per_ray (int): Number of coarse points per ray. Splits rays
            into equally spaced points.
        - near (float): Point nearest to the camera where ray starts.
        - far (float): Point furthest from the camera where ray ends.
        - white_background (bool): If True sets default RGB value to be 1,
            otherwise will be set to 0 (black).
    rng: PRNG key for adding coordinate noise.
    coord_noise: whether to add coordinate noise or not.

  Returns:
    rgb_map: Rendered view(s). Array of shape (..., 3).
    depth_map: Depth map of view(s). Array of shape (...).
  """
  if isinstance(rng, int):
    rng = jax.random.PRNGKey(rng)
  # Unpack render config
  num_points_per_ray, near, far, white_background = render_config

  # Split rays into ray origins and ray directions
  rays_o, rays_d = rays  # both [..., 3]

  # Compute 3D query coordinates
  z_vals = jnp.linspace(near, far, num_points_per_ray)

  # Optionally add coord noise (randomized stratified sampling)
  if coord_noise:
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = jnp.concatenate([mids, z_vals[..., -1:]], -1)
    lower = jnp.concatenate([z_vals[..., :1], mids], -1)
    t_rand = jax.random.uniform(rng, shape=(*rays_o.shape[:-1],
                                            num_points_per_ray))
    z_vals = lower + (upper - lower) * t_rand
  else:
    # broadcast to make returned shape consistent (..., num_points_per_ray)
    z_vals = jnp.broadcast_to(z_vals[None, :],
                              (*rays_o.shape[:-1], num_points_per_ray))

  # The below line uses (a lot of) broadcasting. In terms of shapes:
  # (...,1,3) + (...,1,3) * (num_points_per_ray,1) = (...,num_points_per_ray,3)
  coords = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

  # Should be an instance of (Latent)ModulatedSiren that outputs 4-dim vector
  out = model.apply(params, coords)  # (..., num_points_per_ray, 4)
  # Compute colors and volume densities
  rgb, density = out[..., :3], out[
      ..., 3]  # (..., num_points_per_ray, 3), (..., num_points_per_ray)
  # Ensure density is positive (..., num_points_per_ray)
  # This is different to the usual relu, but we found that this leads to more
  # stable training for meta-learning.
  density = jax.nn.elu(density, alpha=0.1) + 0.1
  density = jnp.clip(density, 0., MAX_DENSITY)  # upper bound density at 10

  # Do volumetric rendering
  rgb_map, depth_map, _ = volumetric_rendering(rgb, density, z_vals,
                                               rays_d, white_background)

  return rgb_map, depth_map


@functools.partial(
    jax.jit, static_argnames=['model', 'height', 'width', 'render_config'])
def render_pose(model, params, height: int, width: int, focal: float,
                pose: Array, render_config: Tuple[int, float, float, bool]):
  """Renders NeRF scene in a given pose.

  Args:
    model: Haiku transformed model, with input_size = 3, output_size = 4 (3
      for RGB and 1 for density.)
    params: Model params.
    height: Height of image.
    width: Width of image.
    focal: Focal length.
    pose: Can either contain a single pose or a batch of poses, i.e. an
      array of shape (4, 4) or (B, 4, 4).
    render_config: Tuple containing rendering configuration for NeRF.
      This includes the following:
        - num_points_per_ray (int): Number of points per ray. Splits rays
            into equally spaced points.
        - near (float): Point nearest to the camera where ray starts.
        - far (float): Point furthest from the camera where ray ends.
        - white_background (bool): If True sets default RGB value to be 1,
            otherwise will be set to 0 (black).

  Returns:
    rgb_map: Rendered view(s). Array of shape (..., 3).
    depth_map: Depth map of view(s). Array of shape (...).
  """
  if pose.ndim == 3:
    rays = get_rays_batch(height, width, focal, pose)
  else:
    rays = get_rays(height, width, focal, pose)
  return render_rays(model, params, rays, render_config)
