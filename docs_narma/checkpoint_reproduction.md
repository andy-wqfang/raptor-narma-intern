# Checkpoint-Based RAPTOR Reproduction

## Scope

This repository contains a headless, checkpoint-based reproduction of the
robust position-keeping behavior described in
[RAPTOR: A Foundation Policy for Quadrotor Control](https://arxiv.org/abs/2509.11481).
It runs the released recurrent policy against eight L2F quadrotor simulations,
records quantitative results, and streams the live state to a browser-based
Three.js visualizer.

This is a behavioral reproduction using the authors' released checkpoint. It
does not claim to repeat every real-flight experiment or retrain the 1,000 SAC
teachers. The full training workflow is described in
[raptor_training_pipeline.md](raptor_training_pipeline.md).

## Clone and environment setup

The checkpoint reproduction requires Git, `curl`, and Python 3.10 with virtual
environment support. Clone the repository and initialize the source submodule:

```bash
git clone https://github.com/andy-wqfang/raptor-narma-intern.git
cd raptor
git submodule update --init rl-tools
```

Only the pinned `rl-tools` submodule is required for checkpoint inference and
the web visualizers. The much larger `data` and `media` submodules are not
needed unless reproducing the full training dataset or paper media.

Create an isolated Python environment and install the pinned demo stack:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
```

The requirements install the released CPU inference and simulation packages:

```text
foundation-policy==1.0.1
l2f==2.0.18
ui-server==0.0.13
```

Download the official arXiv paper and checkpoint archive and record their
checksums:

```bash
./scripts/fetch_release_artifacts.sh
```

Run the automated suite and the headless quadcopter acceptance scenario:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 python -m demo.position_hold --check
```

When the shell is reopened later, return to the repository and reactivate the
environment with:

```bash
cd raptor
source .venv/bin/activate
```

## Reproduction components

| Component | Implementation |
|---|---|
| Paper and checkpoint download | [`scripts/fetch_release_artifacts.sh`](../scripts/fetch_release_artifacts.sh) |
| Quadrotor scenario and web backend | [`demo/position_hold.py`](../demo/position_hold.py) |
| Released policy loader | `foundation-policy==1.0.1` |
| Dynamics and browser renderer | `l2f==2.0.18` and `ui-server==0.0.13` |
| Automated verification | [`tests/test_position_hold.py`](../tests/test_position_hold.py) |
| Recorded results | `artifacts/position_hold/metrics.json` and `trajectory.npz` |

The download script retrieves arXiv revision `2509.11481v2` and the official
checkpoint archive from the paper's Zenodo record. It validates the published
checkpoint archive MD5 and records SHA-256 hashes for the downloaded artifacts.

## Inference path

The released student policy has the following structure:

```text
22-value observation
        |
        v
Dense(22 -> 16), ReLU
        |
        v
GRU(16 -> 16)
        |
        v
Dense(16 -> 4)
        |
        v
four normalized motor-effort commands
```

The 22 inputs are:

| Signal | Values |
|---|---:|
| Position error | 3 |
| Orientation as a row-major rotation matrix | 9 |
| Linear-velocity error | 3 |
| Body angular velocity | 3 |
| Previous motor commands | 4 |
| **Total** | **22** |

The recurrent state is reset at the beginning of an episode. It is deliberately
not reset for the velocity kick or payload change, allowing the demo to test
online adaptation through the GRU state. Policy evaluation and L2F dynamics run
at 100 Hz.

## Eight simulated quadrotors

[`build_runtime`](../demo/position_hold.py) creates eight deterministic
quadrotors with seed 42. The configurations are sampled using the same physical
relationships and ranges used by the RAPTOR training distribution:

- mass: 0.02--5 kg;
- thrust-to-weight ratio: 1.5--5;
- torque-to-inertia ratio: 40--1,200;
- motor rise time: 0.03--0.10 s;
- motor fall time: 0.03--0.30 s;
- rotor reaction-torque constant: 0.005--0.05; and
- arm length, inertia, thrust coefficients, and disturbance strength derived
  from correlated physical quantities.

These eight configurations were generated for this reproduction. They are not
the eight named vehicles from the paper and are not a hand-selected subset of
the authors' 1,000 teacher configurations. The automated tests verify mass,
thrust-to-weight, inertia, geometry, motor ordering, and finite trajectories.

## Robust position-keeping episode

Each episode lasts 16 seconds:

| Time | Phase |
|---|---|
| 0--4 s | Recover from randomized adverse initial states |
| 4--8 s | Recover from a `[1.0, -0.7, 0.4]` m/s velocity kick |
| 8--16 s | Adapt after mass and inertia increase by 25% |

All eight vehicles physically target their own origin. Their 3-by-3 grid
placement is a visualization-only offset; it does not alter policy observations
or dynamics.

Run the headless acceptance scenario with:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m demo.position_hold --check
```

The command fails if the configured recovery thresholds are missed and writes
the complete trajectory and metrics under `artifacts/position_hold`.

## Headless web visualizer

The remote server never renders a pixel. The data path is:

```text
L2F simulation + RAPTOR inference
                |
                | JSON parameters, state, action
                v
       loopback ui-server relay
                |
                | HTTP + WebSocket through SSH
                v
      local browser / Three.js / WebGL
```

At connection time, the backend sends the renderer source and the eight
dynamics configurations. During the episode it sends current state, policy
actions, phase, event, and position-error telemetry. The browser creates each
body, arm, rotor, coordinate frame, and thrust arrow from those messages.

The dashboard added by [`_dashboard_html`](../demo/position_hold.py) explains
the experiment, labels the three episode phases and display-grid dimensions,
and presents the parameters and live error of every simulated drone.

## Running both visualizers

Use the combined launcher to keep the quadcopter and Firefly pages alive at the
same time:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m demo.visualizers
```

It binds one website to remote loopback on port 13337. The visualizers remain
isolated behind path-aware proxy routes while appearing under one browser
origin.

Forward the site from the local computer:

```bash
ssh -N -L 13337:127.0.0.1:13337 user@remote-host
```

Then open:

- <http://127.0.0.1:13337/quadcopter/?L2FDisplayActions=true>
- <http://127.0.0.1:13337/hexacopter/?L2FDisplayActions=true>

Each page contains a navigation control for switching to the other visualizer.

## Limitations

- The experiment uses simulation state directly and therefore assumes
  odometry-like position, orientation, linear velocity, and angular velocity.
- It does not model a complete estimator, communication stack, firmware
  scheduler, sensor delays, contacts, or full aerodynamics.
- Passing the deterministic scenario demonstrates checkpoint inference and
  disturbance recovery for these configurations; it is not hardware
  certification or a statistical reproduction of all paper results.
