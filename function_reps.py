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

"""SIREN models with FiLM modulations."""

from typing import Any, Callable, Dict, Mapping, Optional, Tuple
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from functa import pytree_conversions

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
OptState = Any


class Sine(hk.Module):
  """Applies a scaled sine transform to input: out = sin(w0 * in)."""

  def __init__(self, w0: float = 1.):
    """Constructor.

    Args:
      w0 (float): Scale factor in sine activation (omega_0 factor from SIREN).
    """
    super().__init__()
    self.w0 = w0

  def __call__(self, x: Array) -> Array:
    return jnp.sin(self.w0 * x)


class FiLM(hk.Module):
  """Applies a FiLM modulation: out = scale * in + shift.

  Notes:
    We currently initialize FiLM layers as the identity. However, this may not
    be optimal. In pi-GAN for example they initialize the layer with a random
    normal.
  """

  def __init__(self,
               f_in: int,
               modulate_scale: bool = True,
               modulate_shift: bool = True):
    """Constructor.

    Args:
      f_in: Number of input features.
      modulate_scale: If True, modulates scales.
      modulate_shift: If True, modulates shifts.
    """
    super().__init__()
    # Must modulate at least one of shift and scale
    assert modulate_scale or modulate_shift
    self.f_in = f_in
    # Initialize FiLM layers as identity
    self.scale = 1.
    self.shift = 0.
    if modulate_scale:
      self.scale = hk.get_parameter('scale', [self.f_in], init=jnp.ones)
    if modulate_shift:
      self.shift = hk.get_parameter('shift', [self.f_in], init=jnp.zeros)

  def __call__(self, x: Array) -> Array:
    return self.scale * x + self.shift


class ModulatedSirenLayer(hk.Module):
  """Applies a linear layer followed by a modulation and sine activation."""

  def __init__(self,
               f_in: int,
               f_out: int,
               w0: float = 1.,
               is_first: bool = False,
               is_last: bool = False,
               modulate_scale: bool = True,
               modulate_shift: bool = True,
               apply_activation: bool = True):
    """Constructor.

    Args:
      f_in (int): Number of input features.
      f_out (int): Number of output features.
      w0 (float): Scale factor in sine activation.
      is_first (bool): Whether this is first layer of model.
      is_last (bool): Whether this is last layer of model.
      modulate_scale: If True, modulates scales.
      modulate_shift: If True, modulates shifts.
      apply_activation: If True, applies sine activation.
    """
    super().__init__()
    self.f_in = f_in
    self.f_out = f_out
    self.w0 = w0
    self.is_first = is_first
    self.is_last = is_last
    self.modulate_scale = modulate_scale
    self.modulate_shift = modulate_shift
    self.apply_activation = apply_activation
    # Follow initialization scheme from SIREN
    self.init_range = 1 / f_in if is_first else jnp.sqrt(6 / f_in) / w0

  def __call__(self, x: Array) -> Array:
    # Shape (n, f_in) -> (n, f_out)
    x = hk.Linear(
        output_size=self.f_out,
        w_init=hk.initializers.RandomUniform(-self.init_range,
                                             self.init_range))(x)
    # Apply non-linearities
    if self.is_last:
      # We assume target data (e.g. RGB values of pixels) lies in [0, 1]. To
      # learn zero-centered features we therefore shift output by .5
      return x + .5
    else:
      # Optionally apply modulation
      if self.modulate_scale or self.modulate_shift:
        x = FiLM(
            self.f_out,
            modulate_scale=self.modulate_scale,
            modulate_shift=self.modulate_shift)(x)
      # Optionally apply activation
      if self.apply_activation:
        x = Sine(self.w0)(x)
      return x


class MetaSGDLrs(hk.Module):
  """Module storing learning rates for meta-SGD.

  Notes:
    This module does not apply any transformation but simply stores the learning
    rates. Since we also learn the learning rates we treat them the same as
    model params.
  """

  def __init__(self,
               num_lrs: int,
               lrs_init_range: Tuple[float, float] = (0.005, 0.1),
               lrs_clip_range: Tuple[float, float] = (-5., 5.)):
    """Constructor.

    Args:
      num_lrs: Number of learning rates to learn.
      lrs_init_range: Range from which initial learning rates will be
        uniformly sampled.
      lrs_clip_range: Range at which to clip learning rates. Default value will
        effectively avoid any clipping, but typically learning rates should
        be positive and small.
    """
    super().__init__()
    self.num_lrs = num_lrs
    self.lrs_init_range = lrs_init_range
    self.lrs_clip_range = lrs_clip_range
    # Initialize learning rates
    self.meta_sgd_lrs = hk.get_parameter(
        'meta_sgd_lrs', [self.num_lrs],
        init=hk.initializers.RandomUniform(*self.lrs_init_range))

  def __call__(self) -> Array:
    # Clip learning rate values
    return jax.tree_map(lambda x: jnp.clip(x, *self.lrs_clip_range),
                        self.meta_sgd_lrs)


class ModulatedSiren(hk.Module):
  """SIREN model with FiLM modulations as in pi-GAN."""

  def __init__(self,
               width: int = 256,
               depth: int = 5,
               out_channels: int = 3,
               w0: float = 1.,
               modulate_scale: bool = True,
               modulate_shift: bool = True,
               use_meta_sgd: bool = False,
               meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
               meta_sgd_clip_range: Tuple[float, float] = (-5., 5.),
               name: Optional[str] = None):
    """Constructor.

    Args:
      width (int): Width of each hidden layer in MLP.
      depth (int): Number of layers in MLP.
      out_channels (int): Number of output channels.
      w0 (float): Scale factor in sine activation in first layer.
      modulate_scale: If True, modulates scales.
      modulate_shift: If True, modulates shifts.
      use_meta_sgd: Whether to use meta-SGD.
      meta_sgd_init_range: Range from which initial meta_sgd learning rates will
        be uniformly sampled.
      meta_sgd_clip_range: Range at which to clip learning rates.
      name: name.
    """
    super().__init__(name=name)
    self.width = width
    self.depth = depth
    self.out_channels = out_channels
    self.w0 = w0
    self.modulate_scale = modulate_scale
    self.modulate_shift = modulate_shift
    self.use_meta_sgd = use_meta_sgd
    self.meta_sgd_init_range = meta_sgd_init_range
    self.meta_sgd_clip_range = meta_sgd_clip_range

    # Initialize meta-SGD learning rates
    if self.use_meta_sgd:
      # Compute total number of modulations in network
      self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
      self.num_modulations = width * (depth - 1) * self.modulations_per_unit
      self.meta_sgd_lrs = MetaSGDLrs(self.num_modulations,
                                     self.meta_sgd_init_range,
                                     self.meta_sgd_clip_range)

  def __call__(self, coords: Array) -> Array:
    """Evaluates model at a batch of coordinates.

    Args:
      coords (Array): Array of coordinates. Should have shape (height, width, 2)
        for images and (depth/time, height, width, 3) for 3D shapes/videos.

    Returns:
      Output features at coords.
    """
    # Flatten coordinates
    x = jnp.reshape(coords, (-1, coords.shape[-1]))
    # Initial layer
    x = ModulatedSirenLayer(
        f_in=x.shape[-1],
        f_out=self.width,
        is_first=True,
        w0=self.w0,
        modulate_scale=self.modulate_scale,
        modulate_shift=self.modulate_shift)(x)
    # Hidden layers
    for _ in range(1, self.depth - 1):
      # Add ModulatedSirenLayers
      x = ModulatedSirenLayer(
          f_in=x.shape[-1],
          f_out=self.width,
          w0=self.w0,
          modulate_scale=self.modulate_scale,
          modulate_shift=self.modulate_shift)(x)
    # Final layer
    out = ModulatedSirenLayer(
        f_in=x.shape[-1],
        f_out=self.out_channels,
        is_last=True,
        w0=self.w0,
        modulate_scale=self.modulate_scale,
        modulate_shift=self.modulate_shift)(x)
    # Unflatten output. E.g. for images this corresponds to
    # (num_pixels, out_channels) -> (height, width, out_channels)
    return jnp.reshape(out, list(coords.shape[:-1]) + [self.out_channels])


class LatentVector(hk.Module):
  """Module that holds a latent vector.

  Notes:
    This module does not apply any transformation but simply stores a latent
    vector. This is to make sure that all data necessary to represent an image
    (or a NeRF scene or a video) is present in the model params. This also makes
    it easier to use the partition_params function.
  """

  def __init__(self, latent_dim: int, latent_init_scale: float = 0.0):
    """Constructor.

    Args:
      latent_dim: Dimension of latent vector.
      latent_init_scale: Scale at which to randomly initialize latent vector.
    """
    super().__init__()
    self.latent_dim = latent_dim
    self.latent_init_scale = latent_init_scale
    # Initialize latent vector
    self.latent_vector = hk.get_parameter(
        'latent_vector', [latent_dim],
        init=hk.initializers.RandomUniform(-latent_init_scale,
                                           latent_init_scale))

  def __call__(self) -> Array:
    return self.latent_vector


class LatentToModulation(hk.Module):
  """Function mapping latent vector to a set of modulations."""

  def __init__(self,
               latent_dim: int,
               layer_sizes: Tuple[int, ...],
               width: int,
               num_modulation_layers: int,
               modulate_scale: bool = True,
               modulate_shift: bool = True,
               activation: Callable[[Array], Array] = jax.nn.relu):
    """Constructor.

    Args:
      latent_dim: Dimension of latent vector (input of LatentToModulation
        network).
      layer_sizes: List of hidden layer sizes for MLP parameterizing the map
        from latent to modulations. Input dimension is inferred from latent_dim
        and output dimension is inferred from number of modulations.
      width: Width of each hidden layer in MLP of function rep.
      num_modulation_layers: Number of layers in MLP that contain modulations.
      modulate_scale: If True, returns scale modulations.
      modulate_shift: If True, returns shift modulations.
      activation: Activation function to use in MLP.
    """
    super().__init__()
    # Must modulate at least one of shift and scale
    assert modulate_scale or modulate_shift

    self.latent_dim = latent_dim
    self.layer_sizes = tuple(layer_sizes)  # counteract XM that converts to list
    self.width = width
    self.num_modulation_layers = num_modulation_layers
    self.modulate_scale = modulate_scale
    self.modulate_shift = modulate_shift

    # MLP outputs all modulations. We apply modulations on every hidden unit
    # (i.e on width number of units) at every modulation layer.
    # At each of these we apply either a scale or a shift or both,
    # hence total output size is given by following formula
    self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
    self.modulations_per_layer = width * self.modulations_per_unit
    self.output_size = num_modulation_layers * self.modulations_per_layer

    self.forward = hk.nets.MLP(
        self.layer_sizes + (self.output_size,), activation=activation)

  def __call__(self, latent_vector: Array) -> Dict[int, Dict[str, Array]]:
    modulations = self.forward(latent_vector)
    # Partition modulations into scales and shifts at every layer
    outputs = {}
    for i in range(self.num_modulation_layers):
      single_layer_modulations = {}
      # Note that we add 1 to scales so that outputs of MLP will be centered
      # (since scale = 1 corresponds to identity function)
      if self.modulate_scale and self.modulate_shift:
        start = 2 * self.width * i
        single_layer_modulations['scale'] = modulations[start:start +
                                                        self.width] + 1
        single_layer_modulations['shift'] = modulations[start +
                                                        self.width:start +
                                                        2 * self.width]
      elif self.modulate_scale:
        start = self.width * i
        single_layer_modulations['scale'] = modulations[start:start +
                                                        self.width] + 1
      elif self.modulate_shift:
        start = self.width * i
        single_layer_modulations['shift'] = modulations[start:start +
                                                        self.width]
      outputs[i] = single_layer_modulations
    return outputs


class LatentModulatedSiren(hk.Module):
  """SIREN model with FiLM modulations generated from a latent vector."""

  def __init__(self,
               width: int = 256,
               depth: int = 5,
               out_channels: int = 3,
               latent_dim: int = 64,
               layer_sizes: Tuple[int, ...] = (256, 512),
               w0: float = 1.,
               modulate_scale: bool = True,
               modulate_shift: bool = True,
               latent_init_scale: float = 0.01,
               use_meta_sgd: bool = False,
               meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
               meta_sgd_clip_range: Tuple[float, float] = (-5., 5.)):
    """Constructor.

    Args:
      width (int): Width of each hidden layer in MLP.
      depth (int): Number of layers in MLP.
      out_channels (int): Number of output channels.
      latent_dim: Dimension of latent vector (input of LatentToModulation
        network).
      layer_sizes: List of hidden layer sizes for MLP parameterizing the map
        from latent to modulations. Input dimension is inferred from latent_dim
        and output dimension is inferred from number of modulations.
      w0 (float): Scale factor in sine activation in first layer.
      modulate_scale: If True, modulates scales.
      modulate_shift: If True, modulates shifts.
      latent_init_scale: Scale at which to randomly initialize latent vector.
      use_meta_sgd: Whether to use meta-SGD.
      meta_sgd_init_range: Range from which initial meta_sgd learning rates will
        be uniformly sampled.
      meta_sgd_clip_range: Range at which to clip learning rates.
    """
    super().__init__()
    self.width = width
    self.depth = depth
    self.out_channels = out_channels
    self.latent_dim = latent_dim
    self.layer_sizes = layer_sizes
    self.w0 = w0
    self.modulate_scale = modulate_scale
    self.modulate_shift = modulate_shift
    self.latent_init_scale = latent_init_scale
    self.use_meta_sgd = use_meta_sgd
    self.meta_sgd_init_range = meta_sgd_init_range
    self.meta_sgd_clip_range = meta_sgd_clip_range

    # Initialize meta-SGD learning rates
    if self.use_meta_sgd:
      self.meta_sgd_lrs = MetaSGDLrs(self.latent_dim,
                                     self.meta_sgd_init_range,
                                     self.meta_sgd_clip_range)

    # Initialize latent vector and map from latents to modulations
    self.latent = LatentVector(latent_dim, latent_init_scale)
    self.latent_to_modulation = LatentToModulation(
        latent_dim=latent_dim,
        layer_sizes=layer_sizes,
        width=width,
        num_modulation_layers=depth-1,
        modulate_scale=modulate_scale,
        modulate_shift=modulate_shift)

  def modulate(self, x: Array, modulations: Dict[str, Array]) -> Array:
    """Modulates input according to modulations.

    Args:
      x: Hidden features of MLP.
      modulations: Dict with keys 'scale' and 'shift' (or only one of them)
        containing modulations.

    Returns:
      Modulated vector.
    """
    if 'scale' in modulations:
      x = modulations['scale'] * x
    if 'shift' in modulations:
      x = x + modulations['shift']
    return x

  def __call__(self, coords: Array) -> Array:
    """Evaluates model at a batch of coordinates.

    Args:
      coords (Array): Array of coordinates. Should have shape (height, width, 2)
        for images and (depth/time, height, width, 3) for 3D shapes/videos.

    Returns:
      Output features at coords.
    """
    # Compute modulations based on latent vector
    latent_vector = self.latent()
    modulations = self.latent_to_modulation(latent_vector)

    # Flatten coordinates
    x = jnp.reshape(coords, (-1, coords.shape[-1]))

    # Initial layer (note all modulations are set to False here, since we
    # directly apply modulations from latent_to_modulations output).
    x = ModulatedSirenLayer(
        f_in=x.shape[-1],
        f_out=self.width,
        is_first=True,
        w0=self.w0,
        modulate_scale=False,
        modulate_shift=False,
        apply_activation=False)(x)
    x = self.modulate(x, modulations[0])
    x = Sine(self.w0)(x)

    # Hidden layers
    for i in range(1, self.depth - 1):
      x = ModulatedSirenLayer(
          f_in=x.shape[-1],
          f_out=self.width,
          w0=self.w0,
          modulate_scale=False,
          modulate_shift=False,
          apply_activation=False)(x)
      x = self.modulate(x, modulations[i])
      x = Sine(self.w0)(x)

    # Final layer
    out = ModulatedSirenLayer(
        f_in=x.shape[-1],
        f_out=self.out_channels,
        is_last=True,
        w0=self.w0,
        modulate_scale=False,
        modulate_shift=False)(x)

    # Unflatten output
    return jnp.reshape(out, list(coords.shape[:-1]) + [self.out_channels])


# Helper functions


def get_num_weights_and_modulations(params: hk.Params) -> Tuple[int, int]:
  """Returns the number of weights and modulations of ModulatedSiren model.

  Args:
    params (hk.Params): Parameters from ModulatedSiren model.

  Returns:
    Number of weights and modulations.

  Notes:
    This relies on the partition_params function which assumes all modulations
    are stored in FiLM layers. If we change this in the future, this function
    will break.
  """
  weights, modulations = partition_params(params)
  return hk.data_structures.tree_size(weights), hk.data_structures.tree_size(
      modulations)


def partition_params(params: hk.Params) -> Tuple[hk.Params, hk.Params]:
  """Partitions ModulatedSiren parameters into weights and modulations.

  Args:
    params (hk.Params): Parameters of ModulatedSiren or LatentModulatedSiren
      model.

  Returns:
    Weights and modulations of network.
  """
  # If predicate is True, module contains FiLM parameters or a latent vector
  # mapping to FiLM parameters
  predicate = lambda module_name, name, value: 'fi_lm' in module_name or 'latent_vector' in module_name
  modulations, weights = hk.data_structures.partition(predicate, params)
  return weights, modulations


def partition_shared_params(
    shared_params: hk.Params) -> Tuple[hk.Params, hk.Params]:
  """Partitions shared params parameters into weights and learning rates.

  Args:
    shared_params (hk.Params): Shared parameters of ModulatedSiren or
      LatentModulatedSiren model, i.e. parameters that are not updated in inner
      loop and are shared across datapoints.

  Returns:
    Weights and learning rates of network.
  """
  predicate = lambda module_name, name, value: 'meta_sgd_lrs' in module_name
  lrs, weights = hk.data_structures.partition(predicate, shared_params)
  return weights, lrs


def merge_params(weights: hk.Params, modulations: hk.Params) -> hk.Params:
  """Merges weights and modulations into a single set of parameters.

  Args:
    weights (hk.Params):
    modulations (hk.Params):

  Returns:
    Parameters of ModulatedSiren model.
  """
  return hk.data_structures.merge(modulations, weights)


def update_params(params: hk.Params, modulation: Array) -> hk.Params:
  """Update ModulatedSiren parameters by only updating modulations.

  Args:
    params (hk.Params): Parameters of ModulatedSiren or LatentModulatedSiren
      model.
    modulation (Array): Array representation of modulations, shape (mod_dim,).

  Returns:
    Updated params.
  """
  # extract non-modulation weights from params and tree structure for mods
  weights, init_modulation = partition_params(params)
  _, concat_idx, tree_def = pytree_conversions.pytree_to_array(init_modulation)
  # update modulations and merge with non-modulation weights
  modulation_tree = pytree_conversions.array_to_pytree(
      modulation, concat_idx, tree_def)
  modulated_params = merge_params(weights, modulation_tree)
  return modulated_params


def get_coordinate_grid(res: int, centered: bool = True) -> Array:
  """Returns a normalized coordinate grid for a res by res sized image.

  Args:
    res (int): Resolution of image.
    centered (bool): If True assumes coordinates lie at pixel centers. This is
      equivalent to the align_corners argument in Pytorch. This should always be
      set to True as this ensures we are consistent across different
      resolutions, but keep False as option for backwards compatibility.

  Returns:
    Jnp array of shape (height, width, 2).

  Notes:
    Output will be in [0, 1] (i.e. coordinates are normalized to lie in [0, 1]).
  """
  if centered:
    half_pixel = 1. / (2. * res)  # Size of half a pixel in grid
    coords_one_dim = jnp.linspace(half_pixel, 1. - half_pixel, res)
  else:
    coords_one_dim = jnp.linspace(0, 1, res)
  # Array will have shape (height, width, 2)
  return jnp.stack(
      jnp.meshgrid(coords_one_dim, coords_one_dim, indexing='ij'), axis=-1)
