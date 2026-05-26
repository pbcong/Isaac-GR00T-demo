# Isaac Sim Setup Notes

## WSL2 Headless Fix

The URDF importer extension crashes on startup in WSL2 headless mode because
`omni.ui` tries to create GUI widgets without a display.

**Patched file** (in-venv, applied once):
```
.venv/lib/python3.10/site-packages/isaacsim/extscache/
  isaacsim.asset.importer.urdf-2.3.10+106.4.0.lx64.r.cp310/
  isaacsim/asset/importer/urdf/scripts/extension.py
```

The `on_startup` method now checks `/app/window/hideUi` and skips all UI
creation when headless. We load the G1 from a pre-built USD anyway, so the
URDF importer GUI is not needed.

## Running the demo

```bash
# Always use the venv Python, not system python3
.venv/bin/python3 -m isaac_nav_demo.run_demo \
    --policy-path g1_nav_demo/walk_policy/motion.pt \
    --command "Smoke detected. Navigate to the fire source and inspect it."
```

## Environment variables needed

```bash
OPENROUTER_API_KEY=...
UNITREE_ISAACLAB_ROOT=~/unitree_sim_isaaclab
```

## Package install command (if re-installing)

```bash
uv pip install \
    "isaacsim==4.5.0.0" \
    "isaacsim-extscache-physics==4.5.0.0" \
    "isaacsim-extscache-kit==4.5.0.0" \
    "isaacsim-extscache-kit-sdk==4.5.0.0" \
    "isaacsim-robot-setup==4.5.0.0" \
    --find-links https://pypi.nvidia.com/isaacsim/ \
    --find-links https://pypi.nvidia.com/isaacsim-extscache-physics/ \
    --find-links https://pypi.nvidia.com/isaacsim-extscache-kit/ \
    --find-links https://pypi.nvidia.com/isaacsim-extscache-kit-sdk/ \
    --find-links https://pypi.nvidia.com/isaacsim-robot-setup/
```

Re-apply the WSL2 headless patch after reinstalling.
