# GR00T N1.6 Demo Guide

---

## How it works

```
┌──────────────────────┐              ┌──────────────────────────────────┐
│    Policy Server     │              │        Simulation Client         │
│                      │   ZMQ/TCP    │                                  │
│  run_gr00t_server.py │◄────────────►│  rollout_policy.py               │
│                      │  port 5555   │                                  │
│  GR00T N1.6 model    │              │  MuJoCo physics + rendering      │
│  Diffusion inference │              │  Sends camera + joint state      │
│  Returns action chunk│              │  Executes returned actions       │
└──────────────────────┘              └──────────────────────────────────┘
```

Per step:
1. Simulation client captures robot camera image + joint state → sends to policy server over TCP
2. Policy server runs diffusion inference → returns action chunk (~20 joint commands)
3. Client executes those commands → steps physics → renders frame → repeat

The two processes communicate over localhost or TCP and can run on the **same machine** or different machines.

---

## Part 0: Request a GPU Node (HPC)

If you are on an HPC cluster, request an interactive GPU node before doing anything else:

```bash
qsub -I -l select=1:ngpus=1 -l walltime=12:00:00 -q ic102 -P 71001002
```

Wait until you land on the compute node, then proceed with Parts 1 and 2 in separate terminals on that node.

---

## Part 1: Start the Policy Server

### 1.1 Install dependencies

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T

pip install uv   # if not already installed
uv pip install -e ".[base]"
```

Verify your GPU:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 1.2 Start the server

The server downloads the model from HuggingFace on first run (~6 GB).

```bash
cd Isaac-GR00T

uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper \
    --port 1811
```

Wait for:
```
Server is ready and listening on tcp://*:1811
```

**Leave this terminal open.**

> To pre-download the model instead of downloading at startup:
> ```bash
> huggingface-cli download nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
>     --local-dir ./models/gr00t-g1-pnp
> ```
> Then use `--model-path ./models/gr00t-g1-pnp`.

---

## Part 2: Set Up the Simulation Client

### 2.1 Install system libraries

```bash
sudo apt-get install -y \
    libgl1-mesa-dev libglu1-mesa \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libglfw3 libglfw3-dev libegl1-mesa-dev \
    git-lfs

git lfs install
```

### 2.2 Set up the G1 simulation environment

```bash
# Pull the submodule
git submodule update --init external_dependencies/GR00T-WholeBodyControl

# Install sim dependencies and download robot assets (takes a few minutes)
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
```

This creates a self-contained venv at:
```
gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/
```

Use this venv's Python to run the simulation — not your system Python.

---

## Part 3: Run the Demo

### Option A — Both server and client on the same machine (recommended for HPC)

If your laptop can't handle MuJoCo rendering, run everything on the GPU node. The simulation renders headlessly via EGL and saves videos to disk.

```bash
# Open a second terminal on the same compute node (server should already be running)
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --env_name "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc" \
    --policy_client_host localhost \
    --policy_client_port 1811 \
    --n_episodes 10 \
    --n_envs 1 \
    --n_action_steps 20 \
    --max_episode_steps 1440
```

Note: `--n_envs` defaults to 8, and the client internally runs `n_episodes = max(n_episodes, n_envs)`, so you may see `Episodes: 0/8` even if you set `--n_episodes 1`. Setting `--n_envs 1` avoids `AsyncVectorEnv` and can help surface the underlying exception if a worker would otherwise crash.

No live window appears, but episode videos are saved to the `Video saved to:` path printed by the client (typically under `/scratch/users/$USER/`, e.g. `/scratch/users/ntu/cong045/sim_eval_videos_gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc_ac20_df508bb0-e631-40ae-82a6-6c09965ba894`). Copy them back with `scp` or `rsync` to review.

Verify EGL is available before running:
```bash
ldconfig -p | grep -E "libEGL|libnvidia-egl"
```
On a DGX/H100 node this is almost always present. If the setup script's sanity check passed, you're good.

---

### Option B — Server on HPC, client on your laptop (split setup)

Use this if your laptop can handle MuJoCo rendering and you want a live window.

#### If the server is behind a jump host (SSH tunneling)

Forward the ZMQ port through your SSH config. Example for NSCC:

```bash
# Run this on your laptop, keep the terminal open
ssh -L 5559:a2ap-dgx026:5559 jhnscc
```

This maps `localhost:5559` on your laptop to the compute node through the jump chain. Then confirm it works:
```bash
nc -zv localhost 5559   # should say "succeeded"
```

#### Test the connection

```bash
python -c "
from gr00t.policy.server_client import PolicyClient
c = PolicyClient('localhost', 5559)
print('Connected:', c.ping())
"
# Expected: Connected: True
```

#### Run the demo

```bash
export MUJOCO_GL=glfw

gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --env_name "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc" \
    --policy_client_host localhost \
    --policy_client_port 5559 \
    --n_episodes 10 \
    --n_action_steps 20 \
    --max_episode_steps 1440
```

A live window opens showing the Unitree G1 humanoid robot picking an apple and placing it on a plate. Each episode resets automatically.

---

## Part 3.5: Render pre-generated actions (no inference)

If you only want to validate that the Unitree G1 simulation can **execute and render action chunks** (without running GR00T inference), run the server in **ReplayPolicy** mode using a pre-generated dataset.

### 3.5.1 Download the Unitree G1 dataset (one-time)

```bash
cd Isaac-GR00T/examples/GR00T-WholeBodyControl

git clone --filter=blob:none --no-checkout https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
cd PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
git sparse-checkout init --cone
git sparse-checkout set unitree_g1.LMPnPAppleToPlateDC
git checkout
git lfs pull
```

### 3.5.2 Terminal 1: start a ReplayPolicy server (localhost:1811)

```bash
cd Isaac-GR00T

uv run python gr00t/eval/run_gr00t_server.py \
    --dataset-path "examples/GR00T-WholeBodyControl/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/unitree_g1.LMPnPAppleToPlateDC" \
    --embodiment-tag UNITREE_G1 \
    --execution-horizon 20 \
    --video-backend ffmpeg \
    --use-sim-policy-wrapper \
    --port 1811
```

### 3.5.3 Terminal 2: run the simulation client and render

On some clusters, `CUDA_VISIBLE_DEVICES` is set to a GPU UUID (e.g. `GPU-...`). For RoboSuite/MuJoCo EGL, force numeric device IDs:

```bash
cd Isaac-GR00T
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --env_name "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc" \
    --policy_client_host localhost \
    --policy_client_port 1811 \
    --n_episodes 1 \
    --n_action_steps 20 \
    --max_episode_steps 1440 \
    --n_envs 1
```

Videos are saved to the printed `Video saved to:` directory.

Example:
```
Video saved to:  /scratch/users/ntu/cong045/sim_eval_videos_gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc_ac20_df508bb0-e631-40ae-82a6-6c09965ba894
```

## Part 4: Troubleshooting

### Cannot connect to server

```bash
# Test raw TCP
nc -zv localhost 1811          # same machine
nc -zv 192.168.1.42 1811       # remote

# On the server, confirm it's listening
ss -tlnp | grep 1811
```

If `nc` fails, the issue is network or firewall — not GR00T.

### No window / "cannot open display" (Option B)

You must run the client directly on your laptop, not over SSH. If you must use SSH:

```bash
ssh -X user@your-laptop-ip
export DISPLAY=:0
export MUJOCO_GL=glfw
```

### EGL not found (Option A)

```bash
sudo apt-get install -y libegl1-mesa-dev libglu1-mesa
```

Then re-run the setup script.

### "git-lfs: command not found" during setup

```bash
sudo apt-get install -y git-lfs && git lfs install
```

Then re-run the setup script.

### Robot barely moves

`--n_action_steps` must match what the model was trained with. For G1, always use `--n_action_steps 20`.

### Robot motion is jerky / stop-and-go

Normal — this is the gap between action chunks. The diffusion policy predicts steps ahead; when one chunk ends before the next arrives, motion briefly pauses.

### "CUDA out of memory"

```bash
nvidia-smi   # check nothing else is holding GPU memory
```

The 3B model needs ~10 GB VRAM. H100 80GB has ample headroom.

---

## Quick Reference

```bash
# ── Request GPU node (HPC) ────────────────────────────────────
qsub -I -l select=1:ngpus=1 -l walltime=12:00:00 -q ic102 -P 71001002

# ── Terminal 1: policy server ─────────────────────────────────
cd Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper \
    --port 1811

# ── Terminal 2: simulation client (same machine, headless) ────
export MUJOCO_GL=egl
gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --env_name "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc" \
    --policy_client_host localhost \
    --policy_client_port 1811 \
    --n_episodes 10 \
    --n_action_steps 20 \
    --max_episode_steps 1440

# Videos saved to /tmp/sim_eval_videos_*/

# ── One-time setup (simulation client) ───────────────────────
sudo apt-get install -y libgl1-mesa-dev libglu1-mesa libglfw3 libglfw3-dev libegl1-mesa-dev git-lfs
git lfs install
git submodule update --init external_dependencies/GR00T-WholeBodyControl
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh

# ── SSH tunnel (Option B only, run on laptop) ─────────────────
ssh -L 5559:a2ap-dgx026:5559 jhnscc
```
