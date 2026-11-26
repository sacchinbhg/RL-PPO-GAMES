#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /opt/emerald-rl
apt-get -qq update
apt-get install -y -qq --no-install-recommends python3-venv libretro-mgba git ca-certificates gcc
python3 -m venv /opt/emerald-rl/venv
/opt/emerald-rl/venv/bin/pip -q install uv
/opt/emerald-rl/venv/bin/uv pip install --python /opt/emerald-rl/venv/bin/python \
    'jax[cuda12]==0.4.33' 'numpy<2' 'elements>=3.19.1' 'granular>=0.20.3' \
    'ninjax>=3.5.1' 'portal>=3.5.0' 'scope>=0.4.4' chex optax jaxtyping einops \
    colored_traceback tqdm ruamel.yaml cloudpickle pillow av ipdb google-resumable-media
JAX_PLATFORMS=cuda /opt/emerald-rl/venv/bin/python - <<'PY'
import jax
import jax.numpy as jnp
x=jnp.ones((512,512),dtype=jnp.float32)
y=jax.jit(lambda a:a@a)(x).block_until_ready()
print("JAX_GPU_VERIFIED",jax.__version__,jax.devices(),float(y[0,0]),flush=True)
assert y.devices().pop().platform=="gpu"
PY
