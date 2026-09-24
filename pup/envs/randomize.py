"""Provided dynamics randomization, using Playground's (model, in_axes) pattern.

Brax calls this once per training run with a batch of PRNG keys and expects a
pair: a model whose randomized fields have gained a leading batch axis, and an
`in_axes` tree marking which fields to map over (0) and which to share (None).

What gets perturbed, and why each one matters on hardware:

* **floor friction** 0.6-1.0 -- the single biggest sim2real gap for a legged
  robot. A policy trained on one friction coefficient learns to rely on it.
* **trunk mass** +/-0.5 kg -- batteries, cameras and tape are added to real
  robots constantly, and nobody updates the URDF.
* **actuator gains** +/-10 %, kp and kv together -- real motor drivers are not
  calibrated to three decimal places, and gains drift with temperature.
"""

import jax
import mujoco
from mujoco import mjx

from pup.envs.constants import MJX_SCENE

_FLOOR_GEOM_ID = mujoco.MjModel.from_xml_path(str(MJX_SCENE)).geom("floor").id


def domain_randomize(model: mjx.Model, rng: jax.Array) -> tuple[mjx.Model, mjx.Model]:
    """Batch friction, trunk mass and matching PD gains over rng keys, shape (N, 2)."""
    @jax.vmap
    def sample(key):
        friction_key, mass_key, gain_key = jax.random.split(key, 3)
        friction = model.geom_friction.at[_FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(friction_key, minval=0.6, maxval=1.0))
        mass = model.body_mass.at[1].add(jax.random.uniform(mass_key, minval=-0.5, maxval=0.5))
        factor = jax.random.uniform(gain_key, minval=0.9, maxval=1.1)
        gain = model.actuator_gainprm.at[:, 0].multiply(factor)
        # For a <position> actuator, biasprm = [0, -kp, -kv]; scale kp and kv
        # together so the damping ratio stays put.
        bias = model.actuator_biasprm.at[:, 1:3].multiply(factor)
        return friction, mass, gain, bias

    values = sample(rng)
    fields = ("geom_friction", "body_mass", "actuator_gainprm", "actuator_biasprm")
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    return (model.tree_replace(dict(zip(fields, values))),
            in_axes.tree_replace(dict.fromkeys(fields, 0)))
