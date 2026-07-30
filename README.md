<div align="center">
<img src="https://github.com/rl-tools/raptor-media/blob/ef2dfc6ec650ad0226d74b9db33083cb9b39b4f7/logo.svg"></img>
</div>

<h1 align="center">A Foundation Policy for Quadrotor Control</h1>

<div align="center">
<a href="https://youtu.be/hVzdWRFTX3k" rel="Link to video"><img src="https://github.com/rl-tools/raptor-media/blob/ef2dfc6ec650ad0226d74b9db33083cb9b39b4f7/thumbnail.jpg" width='450'/></a>
    </br>
<a href="https://raptor.rl.tools" rel="Link to Project Page"><img src="https://github.com/rl-tools/raptor-media/blob/ef2dfc6ec650ad0226d74b9db33083cb9b39b4f7/raptor.rl.tools.gif" width='450'/></a>
</div>

## For Narma intern work reproduction, please refer to documentation in docs_narma/

## Usage
If you want to use your own simulator:
```bash
pip install foundation-policy==1.0.1
```
```python
from foundation_policy import Raptor
policy = Raptor()
policy.reset()
for simulation_step in range(1000):
    observation = np.array([[*sim.position, *R(sim.orientation).flatten(), *sim.linear_velocity, *sim.angular_velocity, *sim.action]])
    action = policy.evaluate_step(observation)[0] # the policy works on batches by default
    simulation.step(action) # simulation dt=10 ms
```
Note that the axis conventions are FLU (x = forward, y = left, z = up). Please convert position, orientation, linear velocity and angular velocity into these conventions. 

- **Position**: Absolute position in meter. Can be relative offset to a target trajectory as well.
- **Orientation**: Flattened (row-major) rotation matrix.
- **Linear Velocity**: Linear velocity in m/s (can be relative offset to target trajectory as well).
- **Angular Velocity**: Angular velocity in the body frame and in rad/s.
- **Previous Action**: The `sim.action` is just the previous action (same normalization).
- **Action**: The action motor conventions are [front-right, back-right, back-left, front-left] and the motor commands are normalized in the range [-1, 1]. `rpm(action) = (max_rpm - min_rpm) * (action + 1)/2 + min_rpm`

Note that the poicy is trained for control at 100 Hz. While it can handle deviations from that, we expect best results around the nominal value. The control can be run at higher frequencies using the technique implemented in the L2F inference executor: `rl_tools/inference/applications/l2f/c_backend.h` by just advancing the latent state of the network every N steps as described later in this document. 


### Usage: L2F
The following instructions show how to use [l2f](https://github.com/rl-tools/l2f), the simulator used for training the foundation policy:
```bash
pip install l2f==2.0.18 ui-server==0.0.13 foundation-policy==1.0.1
```
Run `ui-server` in background and open [http://localhost:13337](http://localhost:13337)
```bash
ui-server
```
Then run the following code:
```python
from copy import copy
import numpy as np
import asyncio, websockets, json
import l2f
from l2f import vector8 as vector
from foundation_policy import Raptor

policy = Raptor()
device = l2f.Device()
rng = vector.VectorRng()
env = vector.VectorEnvironment()
ui = l2f.UI()
params = vector.VectorParameters()
state = vector.VectorState()
observation = np.zeros((env.N_ENVIRONMENTS, env.OBSERVATION_DIM), dtype=np.float32)
next_state = vector.VectorState()

vector.initialize_rng(device, rng, 0)
vector.initialize_environment(device, env)
vector.sample_initial_parameters(device, env, params, rng)
vector.sample_initial_state(device, env, params, state, rng)

def configure_3d_model(parameters_message):
    parameters_message = json.loads(parameters_message)
    for d in parameters_message["data"]:
        d["ui"] = {
            "model": "95d22881d444145176db6027d44ebd3a15e9699a",
            "name": "x500"
        }
    return json.dumps(parameters_message)

async def render(websocket, state, action):
    ui_state = copy(state)
    for i, s in enumerate(ui_state.states):
        s.position[0] += i * 0.1 # Spacing for visualization
    state_action_message = vector.set_state_action_message(device, env, params, ui, ui_state, action)
    await websocket.send(state_action_message)

async def main():
    uri = "ws://localhost:13337/backend" # connection to the UI server
    async with websockets.connect(uri) as websocket:
        handshake = json.loads(await websocket.recv(uri))
        assert(handshake["channel"] == "handshake")
        namespace = handshake["data"]["namespace"]
        ui.ns = namespace
        ui_message = vector.set_ui_message(device, env, ui)
        parameters_message = vector.set_parameters_message(device, env, params, ui)
        # parameters_message = configure_3d_model(parameters_message) # use this for a more realistic 3d model
        await websocket.send(ui_message)
        await websocket.send(parameters_message)
        await asyncio.sleep(1)
        await render(websocket, state, np.zeros((8, 4)))
        await asyncio.sleep(2)
        policy.reset()
        for _ in range(500):
            vector.observe(device, env, params, state, observation, rng)
            action = policy.evaluate_step(observation[:, :22])
            dts = vector.step(device, env, params, state, action, next_state, rng)
            state.assign(next_state)
            await render(websocket, state, action)
            await asyncio.sleep(dts[-1])

if __name__ == "__main__":
    asyncio.run(main())
```

## Reproducing the paper on a headless server

The repository now includes a deterministic, checkpoint-based reproduction of
RAPTOR's robust position-keeping behavior. It runs eight L2F simulators and the
released recurrent policy on the server, while the browser performs all
Three.js/WebGL rendering. No remote display, GPU, or Docker daemon is required.

Detailed implementation notes:

- [Checkpoint-based paper reproduction](docs/checkpoint_reproduction.md)
- [Hexacopter inference and visualization pipeline](docs/hexacopter_pipeline.md)
- [Full RAPTOR training and Meta-Imitation Learning pipeline](docs/raptor_training_pipeline.md)

### How RAPTOR works

RAPTOR is trained in two stages:

1. `src/foundation_policy/pre_training` trains 1,000 independent SAC teachers,
   one for each fixed quadrotor configuration. A teacher is Markovian and sees
   privileged motor-speed information.
2. `src/foundation_policy/post_training` distills the teachers with
   Meta-Imitation Learning. After 10 teacher-forcing epochs, the student is
   rolled out on-policy so it learns to correct its own errors. The released
   network is a 22-to-16 dense layer, a 16-unit GRU, and a 16-to-4 output layer:
   2,084 parameters in total.

The 22 policy inputs are position (3), row-major rotation matrix (9), linear
velocity (3), body angular velocity (3), and previous action (4). The GRU's
hidden state uses the resulting action/observation history to infer relevant
dynamics online, including thrust-to-weight ratio, torque-to-inertia ratio, and
motor lag.

Teachers train on a 50/50 mixture of position holding and smooth
second-order-Langevin trajectories. Episodes last 500 steps at 100 Hz and begin
with adverse states including orientations up to 90 degrees.

### Where drone configurations are created

The entry point is
[`rl-tools/src/foundation_policy/pre_training/sample_dynamics_parameters.cpp`](rl-tools/src/foundation_policy/pre_training/sample_dynamics_parameters.cpp).
It seeds the sampler with zero, creates 1,000 configurations in
`src/foundation_policy/dynamics_parameters`, and creates named Crazyflie, X500,
MRS, FS, race, and Flightmare presets in `src/foundation_policy/registry`.
Each resulting JSON disables further domain randomization so its teacher sees
one fixed vehicle.

The ancestral transformations are implemented in
[`rl-tools/include/rl_tools/rl/environments/l2f/operations_generic/10_sample_initial_parameters.h`](rl-tools/include/rl_tools/rl/environments/l2f/operations_generic/10_sample_initial_parameters.h):

- mass is sampled uniformly in cube-root size over 0.02--5 kg;
- thrust-to-weight is sampled over 1.5--5 and scales all thrust-curve terms;
- torque-to-inertia is sampled over 40--1,200 and determines diagonal inertia;
- arm length scales with cube-root mass plus a size-deviation factor;
- motor rise/fall constants are sampled over 0.03--0.10 s and 0.03--0.30 s;
- rotor torque constant is sampled over 0.005--0.05;
- disturbance-force variance is limited by surplus thrust.

The size-deviation helper unintentionally draws a normal variate with mean
`-range` and standard deviation `range`; the reproduction intentionally keeps
that behavior.

### Where visualizations are created

There are two related visualization paths:

- The project-page `l2f-studio` gitlink runs the L2F simulator and HDF5
  checkpoint in browser WebAssembly. `index.js` constructs the 22 observations,
  `l2f.js` advances the simulation, and Three.js renders the drones.
- The Python path used here runs simulation and inference remotely. L2F
  serializes parameters, state, and action to WebSocket messages. `ui-server`
  relays those messages, and its generic browser page dynamically imports the
  renderer returned by
  [`rl-tools/include/rl_tools/rl/environments/l2f/operations_cpu.h`](rl-tools/include/rl_tools/rl/environments/l2f/operations_cpu.h).
  That renderer constructs the body, arms, rotors, spin-direction colors, and
  thrust arrows directly from each dynamics configuration.

The UI relay does not render frames. It only serves static JavaScript and
relays JSON, which is why it works on a server without a display.

### Install and verify

The required source submodule is pinned and can be initialized with:

```bash
git submodule update --init rl-tools
```

The `data` and `media` submodules are not needed for this checkpoint-based
demo. Create a Python 3.10 environment and install the exact released stack:

```bash
/home/wqfang/miniconda3/envs/aero_vla/bin/python -m venv .venv
.venv/bin/python -m pip install -r requirements-demo.txt
```

Download the current paper revision and official checkpoint, then verify their
checksums:

```bash
./scripts/fetch_release_artifacts.sh
```

Run the 16-second accelerated acceptance scenario:

```bash
.venv/bin/python -m demo.position_hold --check
```

It tests eight deterministic vehicles. They first recover from randomized
initial states, receive a `[1.0, -0.7, 0.4]` m/s velocity kick at 4 seconds,
and receive a 25% mass/inertia payload increase at 8 seconds without resetting
the recurrent policy. Results are written to
`artifacts/position_hold/metrics.json` and `trajectory.npz`.

### Testing the unchanged policy on a hexacopter

[`demo/hexacopter.py`](demo/hexacopter.py) evaluates the released checkpoint on
the open RotorS [AscTec Firefly
model](https://github.com/ethz-asl/rotors_simulator/blob/master/rotors_description/urdf/firefly.xacro).
It uses the published 1.5 kg mass, inertia matrix, six 0.215 m arms, rotor
coordinates and spin directions, thrust/yaw constants, 838 rad/s limit, and
12.5/25 ms asymmetric motor lag.

RAPTOR is not modified and still emits four normalized virtual-motor commands
in its native order. The adapter:

1. interprets those commands through an equivalent X-quad mixer;
2. obtains the requested `[collective thrust, roll, pitch, yaw]` wrench;
3. solves a bounded allocation over the Firefly's six physical motors; and
4. feeds the previous four virtual commands, not the six allocated commands,
   back into RAPTOR's next observation.

Run the headless acceptance test with:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m demo.hexacopter --check
```

The scenario starts at a 30-degree tilt, applies the same velocity kick at 4
seconds, and adds 25% mass/inertia at 8 seconds. It records policy outputs, all
six motor commands/speeds, allocation residuals, poses, and pass/fail metrics
under `artifacts/hexacopter`.

This is a lightweight 6-DoF rigid-body and actuator test derived from RotorS
parameters. It does not execute Gazebo's full aerodynamics, sensor/estimator
stack, contacts, PX4 scheduling, or a real vehicle, so it is an interface
feasibility result rather than hardware validation.

### Browser visualization through SSH

Start both continuously replaying pages:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m demo.visualizers
```

The combined launcher exposes one remote-loopback website on port 13337. A
path-aware gateway keeps the two simulation backends isolated while presenting
them as subpages of the same browser origin.

From the local computer, forward that one HTTP/WebSocket endpoint:

```bash
ssh -N -L 13337:127.0.0.1:13337 user@remote-host
```

Then open either page:

- <http://127.0.0.1:13337/quadcopter/?L2FDisplayActions=true>
- <http://127.0.0.1:13337/hexacopter/?L2FDisplayActions=true>

The site root redirects to the quadcopter page. Navigation controls on both
dashboards switch between the subpages. The same tunnel carries static assets
and live websocket updates. Adding
`L2FDisplayActions=true` displays motor-force arrows; drag the canvas to orbit
the camera.

The quadcopter dashboard explains the experiment, highlights the active
recovery/kick/payload phase, lists all eight configurations, and labels the
visualization-only 3-by-3 grid. The Firefly dashboard shows the
four-command-to-six-motor allocation, episode phase, live position and attitude
error, allocation residual, saturation count, source parameters, and all six
motor coordinates/spin directions.

For a persistent remote process containing both pages:

```bash
tmux new-session -d -s raptor-visualizers \
  "cd /home/wqfang/raptor && OMP_NUM_THREADS=1 .venv/bin/python -m demo.visualizers"
tmux attach -t raptor-visualizers
```

Stop it with `tmux kill-session -t raptor-visualizers`.

Each page can still be run independently when desired:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m demo.position_hold --loop --port 13337
OMP_NUM_THREADS=1 .venv/bin/python -m demo.hexacopter --loop --port 13338
```

## Deployment
For deployment onto flight controllers please refer to the reference implementations for PX4, Betaflight, Crazyflie and M5StampFly which are tracked in the [embedded_platforms](https://github.com/rl-tools/rl-tools/tree/3dea1bc877a8593dcd8349f6fdc4e362f025a0ca/embedded_platforms) folder.

The general idea is to use the provided adapter like [this](https://github.com/rl-tools/px4/blob/8350ec059af043dc794361b778794446808b2dea/external_modules/src/modules/rl_tools_policy/rl_tools_adapter.cpp):
```c++
#include <rl_tools/operations/arm.h>

#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/inference/executor/executor.h>

#include "blob/policy.h"

namespace rlt = rl_tools;

namespace other{
    using DEV_SPEC = rlt::devices::DefaultARMSpecification;
    using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
}

struct RL_TOOLS_INFERENCE_APPLICATIONS_L2F_CONFIG{
    using DEVICE = other::DEVICE;
    using TI = typename other::DEVICE::index_t;
    using RNG = other::DEVICE::SPEC::RANDOM::ENGINE<>;
    static constexpr TI TEST_SEQUENCE_LENGTH_ACTUAL = 5;
    static constexpr TI TEST_BATCH_SIZE_ACTUAL = 2;
    using ACTOR_TYPE_ORIGINAL = rlt::checkpoint::actor::TYPE;
    using POLICY_TEST = rlt::checkpoint::actor::TYPE::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_SEQUENCE_LENGTH<TI, 1>;
    using POLICY = ACTOR_TYPE_ORIGINAL::template CHANGE_BATCH_SIZE<TI, 1>;
    using T = typename POLICY::SPEC::T;
    static auto& policy() {
        return rlt::checkpoint::actor::module;
    }
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr TI CONTROL_INTERVAL_INTERMEDIATE_NS = 2.5 * 1000 * 1000; // Inference is at 500hz
    static constexpr TI CONTROL_INTERVAL_NATIVE_NS = 10 * 1000 * 1000; // Training is 100hz
    static constexpr TI TIMING_STATS_NUM_STEPS = 100;
    static constexpr bool FORCE_SYNC_INTERMEDIATE = true;
    static constexpr TI FORCE_SYNC_NATIVE = 4;
    static constexpr bool DYNAMIC_ALLOCATION = false;
    using WARNING_LEVELS = rlt::inference::executor::WarningLevelsDefault<T>;
};

// #define RL_TOOLS_DISABLE_TEST
#include <rl_tools/inference/applications/l2f/c_backend.h>
```

Make sure to configure the control interval properly. The `CONTROL_INTERVAL_NATIVE_NS` should always be $10$ ms because the foundation policy was trained at $100$ Hz. The policy can be run at higher frequencies, though (e.g. $400$ Hz) in which case the native state progression in the policy is only triggered every $4$ steps as signified by `FORCE_SYNC_NATIVE`. 

This interface provides simple functions for inference like e.g.
```c++
auto executor_status = rl_tools_inference_applications_l2f_control(current_time * 1000, &observation, &action);
```
`rl_tools_inference_applications_l2f_control` should be called at the interval configured in `CONTROL_INTERVAL_INTERMEDIATE_NS`. 

The foundation policy that has been exported as C++ code should be included (e.g. `#include <blob/policy.h>` in the example) and it can be downloaded [here](https://github.com/rl-tools/px4-blob/blob/d081d8ca4a558d90f864e352442da14a4c7dd866/policy.h).


## Training

```bash
git clone https://github.com/rl-tools/raptor.git
cd raptor
git submodule update --init rl-tools data
cd rl-tools
git submodule update --init --recursive -- external/highfive external/json external/tensorboard
cd ..
```

```bash
cat data/foundation-policy-v1-data.tar.gz.part_* > data/foundation-policy-v1-data.tar.gz
cd rl-tools
tar -xvf ../data/foundation-policy-v1-data.tar.gz
cd ..
```

The following can be skipped because the image is also available the Docker Hub and will be automatically downloaded by `docker run` we just include it here for complete reproducibility.
```bash
cd rl-tools/docker
./build_all.sh
cd ../../
```

```bash
docker run -it --rm -v $(pwd)/rl-tools:/rl-tools -v data:/data rltools/rltools:ubuntu24.04_mkl_gcc_base
```

```bash
mkdir build
MKL_ROOT=/opt/intel/oneapi/mkl/latest cmake -S /rl-tools -B /build -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_MKL=ON -DRL_TOOLS_ENABLE_TARGETS=ON -DRL_TOOLS_EXPERIMENTAL=ON -DRL_TOOLS_ENABLE_HDF5=ON -DRL_TOOLS_ENABLE_JSON=ON -DRL_TOOLS_ENABLE_TENSORBOARD=ON
cmake --build /build --target foundation_policy_pre_training_sample_dynamics_parameters --target foundation_policy_pre_training --target foundation_policy_post_training -j$(nproc)
cd /rl-tools
export MKL_NUM_THREADS=1
export RL_TOOLS_EXTRACK_EXPERIMENT=$(date '+%Y-%m-%d_%H-%M-%S')
echo "Experiment: $RL_TOOLS_EXTRACK_EXPERIMENT"
/build/src/foundation_policy/foundation_policy_pre_training_sample_dynamics_parameters
seq 0 999 | xargs -I {} /build/src/foundation_policy/foundation_policy_pre_training ./src/foundation_policy/dynamics_parameters/{}.json
/build/src/foundation_policy/foundation_policy_post_training
```
Note, the pre-training of the 1000 teachers will take a long time. You can add `-P $(nproc)` to run multiple teacher training processes in parallel. By default `foundation_policy_post_training` uses the teacher checkpoints from `foundation-policy-v1-data` so you can also skip the `seq 0 999 | xarts ...` by sending SIGINT by pressing CTRL+C while it is running. 

If you want to use the teacher checkpoints trained by `foundation_policy_pre_training`:
```bash
pip install p2s==0.1.14
cd rl-tools/src/foundation_policy
./extract_checkpoints.sh > checkpoints_$RL_TOOLS_EXTRACK_EXPERIMENT.txt
```
Then change the experiment name to the content of `$RL_TOOLS_EXTRACK_EXPERIMENT` in `post_training/main.cpp` and the experiment directory from `1k-experiments` to `experiments` such that it will find the newly trained checkpoints referred to by the `checkpoint_xxx.txt`.


### macOS
On macOS, use Accelerate instead of MKL and build natively replacing the CMake configure and build commands with:
```
cd rl-tools
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_BACKEND_ENABLE_ACCELERATE=ON -DRL_TOOLS_ENABLE_TARGETS=ON -DRL_TOOLS_EXPERIMENTAL=ON -DRL_TOOLS_ENABLE_HDF5=ON -DRL_TOOLS_ENABLE_JSON=ON -DRL_TOOLS_ENABLE_TENSORBOARD=ON
cmake --build . --target foundation_policy_pre_training_sample_dynamics_parameters --target foundation_policy_pre_training --target foundation_policy_post_training
cd ..
./build/src/foundation_policy/foundation_policy_post_training
```

### CMake 4+

Use `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`
