# Pokémon Emerald world-model training

The agent starts at boot, sees a 60 × 40 RGB image and a compact description of game state, and selects one of nine controller actions. A decision advances 24 game frames. The trainer supports PPO, PPO with an archive of the agent’s own discoveries, and a world model with short imagination rollouts. Frozen-policy evaluations start new games with archive resets disabled.

## Run with mGBA

Use Python 3.11, NumPy, Pillow, and PyTorch. Install a CUDA-enabled PyTorch build when using an NVIDIA GPU. The laptop run used PyTorch 2.7.1 with CUDA 12.8, NumPy 2.2.6, and Pillow 11.3. Supply the mGBA libretro core and an English Emerald revision 0 ROM matching SHA-1 `f3ae088181bf583e55daf962a92bb46f4f1d07b7`. Copy `local.example.json` to `local.json` and set the paths for your machine.

```powershell
.\run.ps1 train runs/emerald-world -Mode world -Envs 8 -Episode 512 -Steps 65536
.\run.ps1 eval runs/emerald-eval -Mode world -Envs 4 -Episode 8192 -Weights runs/emerald-world/policy.pt
```

The `world` mode fits dynamics on real transitions, then updates the policy through three imagined actions. Reports distinguish real controller decisions from imagined transitions. Training can resume from a compatible local `checkpoint.pt`; evaluation requires only the much smaller `policy.pt`. For comparison, use `-Mode archive` with matching seeds and real-decision budgets.

## Build the native backend

The second backend compiles the game C code from [pokeemerald-wasm](https://github.com/tripplyons/pokeemerald-wasm/tree/fd83f5b6e609b61b0b10777e32c74343e1a81b41) as a shared library. Each worker owns one engine. It advances the simulation 24 frames and renders one frame per decision. The GPU trains the neural networks. State observations come from typed game variables, and the map must match the currently loaded map header. The starter signal uses the actual party count.

Prepare a disposable checkout with Git, Python 3.11+, GNU make, Clang, LLVM lld, GNU ARM binutils, libpng, zlib, the usual C/C++ build tools, and uv available. The bridge setup adds a small number of introspection functions to this checkout. Run the following from the Emerald training directory in a shell with the toolchain on its path.

```sh
git clone https://github.com/tripplyons/pokeemerald-wasm.git upstream-emerald
git -C upstream-emerald checkout fd83f5b6e609b61b0b10777e32c74343e1a81b41
python native/prepare.py upstream-emerald                  # add --windows for MSYS2 UCRT64
cd upstream-emerald
make -j4 -f rl-build.mk native-bench rl-core NATIVE_CC=clang NATIVE_AST_CC=clang NATIVE_CFLAGS="-O2 -DNDEBUG -fPIC" NATIVE_LDFLAGS=-fuse-ld=lld
```

For Windows use a local MSYS2 toolchain with MSYS host tools, `/ucrt64/bin/clang` for both compiler variables, and `NATIVE_CFLAGS="-O2 -DNDEBUG"`. Set `native_core` to the resulting `build/native/emerald_rl.dll` or `build/native/emerald_rl.so`. The trainer verifies the supplied reference ROM identity for both backends. Native gameplay executes the compiled source.

```powershell
python verify_native.py --trace upstream-emerald/tools/wasm_replays/mudkip_starter.txt --out runs/backend-check
.\run.ps1 train runs/native-world -Backend native -Mode world -Envs 8 -Episode 512 -Steps 65536
.\run.ps1 eval runs/native-eval -Backend native -Mode world -Envs 4 -Episode 8192 -Weights runs/native-world/policy.pt
```

The native archive stores controller histories tagged with the engine binary’s hash. A restore replays those buttons from a clean boot, including each decision’s render. This works across processes and incurs a cost that grows with route length. The verification script checks cross-worker continuation and times it separately. A complete checkpoint must use the same backend and binary. Policy weights can be transferred between the two for a separate experiment.

## Evaluate progress

Use `verified_event_decision` and `verified_map_decision` in each fresh trial to report the clock, rival, Route 101, and party milestones. The raw map ID can briefly change while mGBA loads a room. Training archives record discoveries across reset branches. A fresh-game success measures whether a frozen policy can reproduce a milestone. The upstream Mudkip controller recording is a compatibility test and is never loaded into training.

Run `python -m unittest -v test_world_model` to check the learning code. To run a timed continuation with hourly fresh evaluations, use `python overnight.py --out runs/night --restart-from runs/native-world/checkpoint.pt --hours 8`. The supervisor detects the checkpoint’s backend and saves a recovery checkpoint every ten minutes.
