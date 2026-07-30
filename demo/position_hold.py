"""Robust RAPTOR position keeping with a loopback-only browser UI.

The simulation and recurrent policy run on the remote machine.  The browser
receives JSON state updates over a WebSocket and performs all Three.js/WebGL
rendering locally.
"""

from __future__ import annotations

import os

# Tiny vector environments are much faster without OpenMP thread launch
# overhead.  This must be set before importing NumPy or the L2F extension.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import asyncio
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import l2f
from foundation_policy import QuadrotorPolicy
from l2f import vector8 as vector


CONTROL_HZ = 100
DT = 1.0 / CONTROL_HZ
TOTAL_STEPS = 16 * CONTROL_HZ
KICK_STEP = 4 * CONTROL_HZ
PAYLOAD_STEP = 8 * CONTROL_HZ
KICK_VELOCITY = np.array([1.0, -0.7, 0.4], dtype=np.float32)
PAYLOAD_FACTOR = 1.25
N_DRONES = 8
STATE_SEED = 7
GRID_SIZE = math.ceil(math.sqrt(N_DRONES))
GRID_SPACING_M = 0.9
GRID_CENTER_SPAN_M = (GRID_SIZE - 1) * GRID_SPACING_M

THRESHOLDS = {
    "pre_kick_error_m": 0.12,
    "kick_peak_error_m": 0.40,
    "pre_payload_error_m": 0.12,
    "payload_peak_error_m": 0.15,
    "final_error_m": 0.10,
}


@dataclass
class Runtime:
    device: Any
    single_environment: Any
    environments: Any
    parameters: Any
    states: Any
    next_states: Any
    rng: Any
    policy: QuadrotorPolicy
    observations: np.ndarray
    configurations: list[dict[str, Any]]
    configuration_metadata: list[dict[str, Any]]


def _scale_matrix(matrix: list[list[float]], factor: float) -> list[list[float]]:
    return [[float(value * factor) for value in row] for row in matrix]


def _configuration_from_baseline(
    baseline: dict[str, Any],
    rng: np.random.Generator,
    index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the dynamics sampler used by sample_dynamics_parameters.cpp.

    NumPy supplies the deterministic random stream; ranges and ancestral
    transformations match the pinned C++ source.
    """

    config = copy.deepcopy(baseline)
    dynamics = config["dynamics"]
    randomization = {
        "thrust_to_weight_min": 1.5,
        "thrust_to_weight_max": 5.0,
        "torque_to_inertia_min": 40.0,
        "torque_to_inertia_max": 1200.0,
        "mass_min": 0.02,
        "mass_max": 5.0,
        "mass_size_deviation": 0.1,
        "motor_rise_min": 0.03,
        "motor_rise_max": 0.10,
        "motor_fall_min": 0.03,
        "motor_fall_max": 0.30,
        "rotor_torque_min": 0.005,
        "rotor_torque_max": 0.05,
        "disturbance_force_max": 0.3,
    }

    gravity_norm = float(np.linalg.norm(dynamics["gravity"]))
    max_action = float(dynamics["action_limit"]["max"])
    maximum_nominal_thrust = sum(
        coefficients[0]
        + coefficients[1] * max_action
        + coefficients[2] * max_action * max_action
        for coefficients in dynamics["rotor_thrust_coefficients"]
    )
    nominal_mass = float(dynamics["mass"])
    nominal_thrust_to_weight = maximum_nominal_thrust / (
        nominal_mass * gravity_norm
    )

    thrust_to_weight = float(
        rng.uniform(
            randomization["thrust_to_weight_min"],
            randomization["thrust_to_weight_max"],
        )
    )
    thrust_factor = thrust_to_weight / nominal_thrust_to_weight

    relative_size_min = np.cbrt(randomization["mass_min"])
    relative_size_max = np.cbrt(randomization["mass_max"])
    sampled_size = float(rng.uniform(relative_size_min, relative_size_max))
    mass = float(
        np.clip(
            sampled_size**3,
            randomization["mass_min"],
            randomization["mass_max"],
        )
    )
    relative_scale = float(np.cbrt(mass / nominal_mass))
    mass_factor = mass / nominal_mass
    dynamics["mass"] = mass
    thrust_coefficient_factor = thrust_factor * mass_factor
    dynamics["rotor_thrust_coefficients"] = [
        [float(value * thrust_coefficient_factor) for value in coefficients]
        for coefficients in dynamics["rotor_thrust_coefficients"]
    ]

    maximum_rotor_thrust = thrust_to_weight * mass * gravity_norm / 4
    first_rotor_x = abs(float(dynamics["rotor_positions"][0][0]))
    maximum_torque = first_rotor_x * math.sqrt(2.0) * maximum_rotor_thrust
    nominal_torque_to_inertia = maximum_torque / float(dynamics["J"][0][0])
    torque_to_inertia = float(
        rng.uniform(
            randomization["torque_to_inertia_min"],
            randomization["torque_to_inertia_max"],
        )
    )
    torque_to_inertia_factor = torque_to_inertia / nominal_torque_to_inertia

    # This intentionally preserves the pinned source's accidental Normal
    # distribution: mean=-range, standard deviation=range.
    raw_size_deviation = float(
        rng.normal(
            -randomization["mass_size_deviation"],
            randomization["mass_size_deviation"],
        )
    )
    size_deviation_factor = (
        1.0 / (1.0 - raw_size_deviation)
        if raw_size_deviation < 0
        else 1.0 + raw_size_deviation
    )
    rotor_distance_factor = relative_scale * size_deviation_factor
    inertia_factor = torque_to_inertia_factor / rotor_distance_factor

    dynamics["J"] = _scale_matrix(dynamics["J"], 1.0 / inertia_factor)
    dynamics["J_inv"] = _scale_matrix(dynamics["J_inv"], inertia_factor)
    dynamics["rotor_positions"] = [
        [float(value * rotor_distance_factor) for value in position]
        for position in dynamics["rotor_positions"]
    ]

    maximum_rotor_distance = max(
        float(np.linalg.norm(position)) for position in dynamics["rotor_positions"]
    )
    config["mdp"]["termination"]["position_threshold"] = (
        maximum_rotor_distance * 20.0
    )
    config["mdp"]["init"]["max_position"] = maximum_rotor_distance * 10.0
    # Termination is an RL episode-management signal, not part of the flight
    # physics.  Disable automatic episode endings so the demo can observe
    # recovery continuously without resetting the policy or simulator.
    config["mdp"]["termination"]["enabled"] = False

    rotor_torque = float(
        rng.uniform(
            randomization["rotor_torque_min"],
            randomization["rotor_torque_max"],
        )
    )
    motor_rise = float(
        rng.uniform(
            randomization["motor_rise_min"],
            randomization["motor_rise_max"],
        )
    )
    motor_fall = float(
        rng.uniform(
            randomization["motor_fall_min"],
            randomization["motor_fall_max"],
        )
    )
    dynamics["rotor_torque_constants"] = [rotor_torque] * 4
    dynamics["rotor_time_constants_rising"] = [motor_rise] * 4
    dynamics["rotor_time_constants_falling"] = [motor_fall] * 4

    surplus_thrust_to_weight = max(thrust_to_weight - 1.0, 0.0)
    disturbance_multiple = float(
        rng.uniform(
            0.0,
            surplus_thrust_to_weight
            * randomization["disturbance_force_max"],
        )
    )
    disturbance_force_std = (
        disturbance_multiple * thrust_to_weight * mass / 3.0
    )
    config["disturbances"]["random_force"] = {
        "mean": 0.0,
        "std": disturbance_force_std,
    }

    metadata = {
        "index": index,
        "mass_kg": mass,
        "thrust_to_weight": thrust_to_weight,
        "torque_to_inertia": torque_to_inertia,
        "rotor_span_m": maximum_rotor_distance * 2.0,
        "motor_rise_s": motor_rise,
        "motor_fall_s": motor_fall,
        "rotor_torque_constant": rotor_torque,
        "disturbance_force_std": disturbance_force_std,
        "size_deviation_raw": raw_size_deviation,
    }
    return config, metadata


def generate_configurations(
    device: Any,
    single_environment: Any,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_parameters = l2f.Parameters()
    l2f.initial_parameters(device, single_environment, baseline_parameters)
    baseline = json.loads(
        l2f.parameters_to_json(device, single_environment, baseline_parameters)
    )
    rng = np.random.default_rng(seed)
    pairs = [
        _configuration_from_baseline(baseline, rng, index)
        for index in range(N_DRONES)
    ]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _parameters_from_configuration(
    device: Any,
    single_environment: Any,
    configuration: dict[str, Any],
) -> Any:
    parameters = l2f.Parameters()
    l2f.parameters_from_json(
        device,
        single_environment,
        json.dumps(configuration),
        parameters,
    )
    return copy.copy(parameters)


def build_runtime(seed: int) -> Runtime:
    device = l2f.Device()
    single_environment = l2f.Environment()
    l2f.initialize_environment(device, single_environment)

    environments = vector.VectorEnvironment()
    parameters = vector.VectorParameters()
    states = vector.VectorState()
    next_states = vector.VectorState()
    rng = vector.VectorRng()

    vector.initialize_environment(device, environments)
    # Keep the state/disturbance stream independent from the configuration
    # stream so the published acceptance scenario is stable.
    vector.initialize_rng(device, rng, STATE_SEED)

    configurations, metadata = generate_configurations(
        device,
        single_environment,
        seed,
    )
    parameters.parameters = [
        _parameters_from_configuration(device, single_environment, configuration)
        for configuration in configurations
    ]
    vector.sample_initial_state(device, environments, parameters, states, rng)

    policy = QuadrotorPolicy()
    policy.reset()
    observations = np.zeros(
        (N_DRONES, environments.OBSERVATION_DIM),
        dtype=np.float32,
    )
    return Runtime(
        device=device,
        single_environment=single_environment,
        environments=environments,
        parameters=parameters,
        states=states,
        next_states=next_states,
        rng=rng,
        policy=policy,
        observations=observations,
        configurations=configurations,
        configuration_metadata=metadata,
    )


def ui_message(runtime: Runtime, ui: Any) -> str:
    payload = json.loads(
        vector.set_ui_message(runtime.device, runtime.environments, ui)
    )
    render_function = payload["data"]["render_function"]
    replacements = {
        "const grid_distance = 0.0": (
            f"const grid_distance = {GRID_SPACING_M}"
        ),
        "const x = (i % grid_size) * grid_distance": (
            "const x = ((i % grid_size) - (grid_size - 1) / 2) * grid_distance"
        ),
        "const y = Math.floor(i / grid_size) * grid_distance": (
            "const y = (Math.floor(i / grid_size) - "
            "(grid_size - 1) / 2) * grid_distance"
        ),
    }
    for original, replacement in replacements.items():
        if render_function.count(original) != 1:
            raise RuntimeError(f"Unexpected L2F renderer; missing marker: {original}")
        render_function = render_function.replace(original, replacement, 1)
    payload["data"]["render_function"] = render_function
    payload["data"]["options"] = {
        "showAxes": True,
        "camera_position": [0.7, 0.7, 1.2],
    }
    payload["latch"] = True
    return json.dumps(payload)


def parameters_message(runtime: Runtime, ui: Any) -> str:
    payload = json.loads(
        vector.set_parameters_message(
            runtime.device,
            runtime.environments,
            runtime.parameters,
            ui,
        )
    )
    for drone_index, (parameters, metadata) in enumerate(
        zip(payload["data"], runtime.configuration_metadata)
    ):
        parameters["demo"] = {
            "label": f"Drone {drone_index + 1}",
            **metadata,
        }
    payload["latch"] = True
    return json.dumps(payload)


def _phase_at_step(completed_steps: int) -> tuple[str, str]:
    if completed_steps < KICK_STEP:
        return (
            "initial_recovery",
            "Initial recovery and dynamics identification",
        )
    if completed_steps < PAYLOAD_STEP:
        return (
            "kick_recovery",
            "Recovering from the velocity kick",
        )
    return (
        "payload_adaptation",
        "Adapting to the 25% payload increase",
    )


def state_message(
    runtime: Runtime,
    ui: Any,
    actions: np.ndarray,
    completed_steps: int,
    episode: int,
) -> str:
    payload = json.loads(
        vector.set_state_action_message(
            runtime.device,
            runtime.environments,
            runtime.parameters,
            ui,
            runtime.states,
            actions,
        )
    )
    phase_id, phase_name = _phase_at_step(completed_steps)
    event = None
    if completed_steps == KICK_STEP:
        event = "Velocity kick applied"
    elif completed_steps == PAYLOAD_STEP:
        event = "25% mass and inertia payload applied"
    payload["data"]["demo"] = {
        "episode": episode,
        "step": completed_steps,
        "time_s": completed_steps * DT,
        "duration_s": TOTAL_STEPS * DT,
        "progress": completed_steps / TOTAL_STEPS,
        "phase_id": phase_id,
        "phase_name": phase_name,
        "event": event,
        "position_errors_m": [
            float(np.linalg.norm(np.asarray(state.position)))
            for state in runtime.states.states
        ],
    }
    payload["latch"] = True
    return json.dumps(payload)


def _apply_velocity_kick(states: Any) -> None:
    state_list = states.states
    for state in state_list:
        state.linear_velocity = (
            np.asarray(state.linear_velocity, dtype=np.float32) + KICK_VELOCITY
        )
    states.states = state_list


def _apply_payload(runtime: Runtime) -> None:
    for configuration in runtime.configurations:
        dynamics = configuration["dynamics"]
        dynamics["mass"] = float(dynamics["mass"] * PAYLOAD_FACTOR)
        dynamics["J"] = _scale_matrix(dynamics["J"], PAYLOAD_FACTOR)
        dynamics["J_inv"] = _scale_matrix(
            dynamics["J_inv"],
            1.0 / PAYLOAD_FACTOR,
        )
    runtime.parameters.parameters = [
        _parameters_from_configuration(
            runtime.device,
            runtime.single_environment,
            configuration,
        )
        for configuration in runtime.configurations
    ]


def _recovery_time(
    errors: np.ndarray,
    start: int,
    stop: int,
    threshold: float,
    stable_steps: int = 50,
) -> float | None:
    last_start = stop - stable_steps
    for index in range(start, last_start + 1):
        if float(np.max(errors[index : index + stable_steps])) <= threshold:
            return (index + 1) * DT
    return None


def _build_metrics(
    runtime: Runtime,
    position_errors: np.ndarray,
    terminated: np.ndarray,
    seed: int,
    elapsed_s: float,
    maximum_action_magnitude: float,
) -> dict[str, Any]:
    per_drone = []
    for drone_index in range(N_DRONES):
        errors = position_errors[:, drone_index]
        pre_kick = float(errors[KICK_STEP - 1])
        kick_peak = float(np.max(errors[KICK_STEP:PAYLOAD_STEP]))
        pre_payload = float(errors[PAYLOAD_STEP - 1])
        payload_peak = float(np.max(errors[PAYLOAD_STEP:]))
        final = float(errors[-1])
        recovery_time = _recovery_time(
            errors,
            KICK_STEP,
            PAYLOAD_STEP,
            THRESHOLDS["pre_payload_error_m"],
        )
        passed = (
            pre_kick <= THRESHOLDS["pre_kick_error_m"]
            and kick_peak <= THRESHOLDS["kick_peak_error_m"]
            and pre_payload <= THRESHOLDS["pre_payload_error_m"]
            and payload_peak <= THRESHOLDS["payload_peak_error_m"]
            and final <= THRESHOLDS["final_error_m"]
            and not bool(terminated[drone_index])
            and bool(np.isfinite(errors).all())
        )
        per_drone.append(
            {
                **runtime.configuration_metadata[drone_index],
                "pre_kick_error_m": pre_kick,
                "kick_peak_error_m": kick_peak,
                "kick_recovery_time_s": recovery_time,
                "pre_payload_error_m": pre_payload,
                "payload_peak_error_m": payload_peak,
                "final_error_m": final,
                "terminated": bool(terminated[drone_index]),
                "finite": bool(np.isfinite(errors).all()),
                "passed": passed,
            }
        )

    return {
        "run": {
            "seed": seed,
            "state_seed": STATE_SEED,
            "control_hz": CONTROL_HZ,
            "simulated_duration_s": TOTAL_STEPS * DT,
            "wall_time_s": elapsed_s,
            "n_drones": N_DRONES,
            "observation_dim_used": 22,
            "action_dim": 4,
            "maximum_absolute_action": maximum_action_magnitude,
            "checkpoint_package": "foundation-policy==1.0.1",
            "simulator_package": "l2f==2.0.18",
            "ui_package": "ui-server==0.0.13",
        },
        "events": [
            {
                "time_s": KICK_STEP * DT,
                "type": "velocity_kick",
                "delta_m_per_s": KICK_VELOCITY.tolist(),
            },
            {
                "time_s": PAYLOAD_STEP * DT,
                "type": "payload",
                "mass_and_diagonal_inertia_factor": PAYLOAD_FACTOR,
            },
        ],
        "thresholds": THRESHOLDS,
        "per_drone": per_drone,
        "passed": all(drone["passed"] for drone in per_drone),
    }


async def run_episode(
    seed: int,
    *,
    websocket: Any | None = None,
    ui: Any | None = None,
    realtime: bool = False,
    episode: int = 1,
) -> tuple[dict[str, Any], np.ndarray]:
    runtime = build_runtime(seed)
    if websocket is not None:
        if ui is None:
            raise ValueError("ui is required when websocket is provided")
        await websocket.send(ui_message(runtime, ui))
        await websocket.send(parameters_message(runtime, ui))

    position_errors = np.zeros((TOTAL_STEPS, N_DRONES), dtype=np.float32)
    trajectory = np.zeros((TOTAL_STEPS, N_DRONES, 3), dtype=np.float32)
    actions = np.zeros((N_DRONES, 4), dtype=np.float32)
    terminated_flags = np.zeros(N_DRONES, dtype=np.bool_)
    ever_terminated = np.zeros(N_DRONES, dtype=np.bool_)
    maximum_action_magnitude = 0.0
    start = time.perf_counter()
    deadline = start

    for step_index in range(TOTAL_STEPS):
        vector.observe(
            runtime.device,
            runtime.environments,
            runtime.parameters,
            runtime.states,
            runtime.observations,
            runtime.rng,
        )
        actions = np.asarray(
            runtime.policy.evaluate_step(runtime.observations[:, :22]),
            dtype=np.float32,
        )
        if actions.shape != (N_DRONES, 4):
            raise RuntimeError(f"Unexpected action shape: {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError(f"Non-finite policy action at step {step_index}")
        step_action_magnitude = float(np.max(np.abs(actions)))
        maximum_action_magnitude = max(
            maximum_action_magnitude,
            step_action_magnitude,
        )

        vector.step(
            runtime.device,
            runtime.environments,
            runtime.parameters,
            runtime.states,
            actions,
            runtime.next_states,
            runtime.rng,
        )
        runtime.states.assign(runtime.next_states)

        completed_steps = step_index + 1
        if completed_steps == KICK_STEP:
            _apply_velocity_kick(runtime.states)
        if completed_steps == PAYLOAD_STEP:
            _apply_payload(runtime)

        vector.terminated(
            runtime.device,
            runtime.environments,
            runtime.parameters,
            runtime.states,
            terminated_flags,
            runtime.rng,
        )
        ever_terminated |= terminated_flags

        state_list = runtime.states.states
        for drone_index, state in enumerate(state_list):
            position = np.asarray(state.position, dtype=np.float32)
            trajectory[step_index, drone_index] = position
            position_errors[step_index, drone_index] = np.linalg.norm(position)

        if websocket is not None:
            message = state_message(
                runtime,
                ui,
                actions,
                completed_steps,
                episode,
            )
            await websocket.send(message)

        if realtime:
            deadline += DT
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

    elapsed_s = time.perf_counter() - start
    metrics = _build_metrics(
        runtime,
        position_errors,
        ever_terminated,
        seed,
        elapsed_s,
        maximum_action_magnitude,
    )
    return metrics, trajectory


def write_results(
    output_dir: Path,
    metrics: dict[str, Any],
    trajectory: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "trajectory.npz",
        position=trajectory,
        dt=np.array(DT),
    )


def _dashboard_html(
    original: str,
    *,
    peer_href: str | None = None,
) -> str:
    dashboard_css = """
    <style id="raptor-dashboard-style">
        :root {
            color-scheme: dark;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        body {
            align-items: stretch !important;
            background: #07111f;
            color: #dbeafe;
            display: flex !important;
            justify-content: flex-start !important;
            overflow: hidden;
        }
        .canvas-container {
            background:
                radial-gradient(circle at 50% 42%, #17304a 0, #0a1728 52%, #050b14 100%);
            flex: 1 1 auto;
            height: 100vh !important;
            min-width: 0;
            width: auto !important;
        }
        #canvas {
            box-shadow: none !important;
        }
        #raptor-dashboard {
            background: rgba(6, 15, 27, 0.98);
            border-left: 1px solid #223b55;
            box-sizing: border-box;
            flex: 0 0 min(540px, 44vw);
            height: 100vh;
            overflow-y: auto;
            padding: 22px 22px 32px;
        }
        #raptor-dashboard h1 {
            font-size: 22px;
            letter-spacing: -0.02em;
            margin: 0 0 5px;
        }
        #raptor-dashboard h2 {
            color: #93c5fd;
            font-size: 12px;
            letter-spacing: 0.1em;
            margin: 24px 0 10px;
            text-transform: uppercase;
        }
        .visualizer-nav {
            align-items: center;
            display: flex;
            gap: 7px;
            margin: 12px 0 4px;
        }
        .visualizer-nav a,
        .visualizer-nav span {
            border: 1px solid #28425f;
            border-radius: 999px;
            color: #9fb8d1;
            font-size: 11px;
            padding: 6px 10px;
            text-decoration: none;
        }
        .visualizer-nav span {
            background: #1d4ed8;
            border-color: #3b82f6;
            color: #eff6ff;
            font-weight: 700;
        }
        .visualizer-nav a:hover {
            background: #102b43;
            border-color: #38bdf8;
            color: #e0f2fe;
        }
        #raptor-dashboard p {
            color: #a9bdd2;
            font-size: 13px;
            line-height: 1.55;
            margin: 8px 0;
        }
        .connection-line {
            align-items: center;
            color: #7f96ad;
            display: flex;
            font-size: 12px;
            gap: 7px;
        }
        .connection-dot {
            background: #f59e0b;
            border-radius: 50%;
            height: 8px;
            width: 8px;
        }
        .connection-dot.connected {
            background: #34d399;
            box-shadow: 0 0 10px rgba(52, 211, 153, 0.55);
        }
        .phase-card {
            background: #0d1d30;
            border: 1px solid #28425f;
            border-radius: 10px;
            padding: 13px 14px;
        }
        #phase-name {
            color: #f8fafc;
            font-size: 15px;
            font-weight: 700;
        }
        #phase-time {
            color: #60a5fa;
            float: right;
            font-size: 13px;
            font-variant-numeric: tabular-nums;
        }
        #phase-event {
            color: #fbbf24;
            font-size: 12px;
            height: 18px;
            margin-top: 5px;
        }
        .timeline {
            display: flex;
            height: 9px;
            margin: 14px 0 7px;
            position: relative;
        }
        .timeline-segment:first-child {
            border-radius: 5px 0 0 5px;
        }
        .timeline-segment:nth-child(1) {
            background: #2563eb;
            width: 25%;
        }
        .timeline-segment:nth-child(2) {
            background: #d97706;
            width: 25%;
        }
        .timeline-segment:nth-child(3) {
            background: #059669;
            border-radius: 0 5px 5px 0;
            width: 50%;
        }
        #timeline-marker {
            background: white;
            border: 2px solid #07111f;
            border-radius: 50%;
            height: 13px;
            left: 0;
            position: absolute;
            top: -4px;
            transform: translateX(-50%);
            transition: left 80ms linear;
            width: 13px;
        }
        .timeline-labels {
            color: #7890a8;
            display: grid;
            font-size: 10px;
            grid-template-columns: 1fr 1fr 2fr;
        }
        .explanation {
            background: #091827;
            border-left: 3px solid #3b82f6;
            border-radius: 4px;
            padding: 8px 12px;
        }
        .grid-spec {
            align-items: baseline;
            background: #10233a;
            border: 1px solid #28425f;
            border-radius: 7px;
            display: flex;
            flex-wrap: wrap;
            gap: 5px 9px;
            margin-top: 12px;
            padding: 9px 10px;
        }
        .grid-spec span {
            color: #7f9bb8;
            font-size: 11px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .grid-spec strong {
            color: #bfdbfe;
            font-size: 12px;
        }
        .table-wrap {
            border: 1px solid #223b55;
            border-radius: 8px;
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            font-size: 11px;
            min-width: 790px;
            width: 100%;
        }
        th {
            background: #10233a;
            color: #8fb4d8;
            font-weight: 600;
            padding: 8px 7px;
            position: sticky;
            text-align: right;
            top: 0;
            white-space: nowrap;
        }
        td {
            border-top: 1px solid #183049;
            color: #c1d4e6;
            font-variant-numeric: tabular-nums;
            padding: 7px;
            text-align: right;
            white-space: nowrap;
        }
        th:first-child, td:first-child {
            color: #f8fafc;
            font-weight: 600;
            left: 0;
            position: sticky;
            text-align: left;
        }
        th:first-child {
            background: #10233a;
        }
        td:first-child {
            background: #091827;
        }
        .error-good {
            color: #34d399;
            font-weight: 700;
        }
        .legend {
            color: #7890a8;
            font-size: 10px;
            line-height: 1.45;
            margin-top: 8px;
        }
        @media (max-width: 900px) {
            body {
                flex-direction: column;
                overflow: auto;
            }
            .canvas-container {
                flex: 0 0 62vh;
                width: 100vw !important;
            }
            #raptor-dashboard {
                border-left: 0;
                border-top: 1px solid #223b55;
                flex: none;
                height: auto;
                width: 100%;
            }
        }
    </style>
    """
    dashboard_body = """
    <aside id="raptor-dashboard">
        <h1>RAPTOR position keeping</h1>
        __VISUALIZER_NAV__
        <div class="connection-line">
            <span id="connection-dot" class="connection-dot"></span>
            <span id="connection-text">Waiting for live telemetry</span>
        </div>

        <h2>What this demonstrates</h2>
        <div class="explanation">
            <p>
                One 2,084-parameter recurrent policy controls all eight
                quadrotors without being told their physical parameters.
                It infers thrust, inertia, geometry, and motor response from
                the recent action/observation history.
            </p>
            <p>
                Every vehicle physically targets the origin. The grid offsets
                and coordinate frames are visual only. Each 16-second episode
                starts from adverse states, applies a velocity kick at 4 s,
                and adds 25% mass and inertia at 8 s without resetting the GRU.
            </p>
            <div class="grid-spec">
                <span>Display grid</span>
                <strong>__GRID_SPEC__</strong>
            </div>
        </div>

        <h2>Episode phase</h2>
        <div class="phase-card">
            <span id="phase-name">Waiting for episode</span>
            <span id="phase-time">0.00 / 16.00 s</span>
            <div id="phase-event"></div>
            <div class="timeline">
                <div class="timeline-segment"></div>
                <div class="timeline-segment"></div>
                <div class="timeline-segment"></div>
                <div id="timeline-marker"></div>
            </div>
            <div class="timeline-labels">
                <span>0–4 s · recover</span>
                <span>4–8 s · kick</span>
                <span>8–16 s · payload adaptation</span>
            </div>
        </div>

        <h2>Drone configurations</h2>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Drone</th>
                        <th>Mass<br>kg</th>
                        <th>Thrust/<br>weight</th>
                        <th>Torque/<br>inertia</th>
                        <th>Span<br>m</th>
                        <th>Rise<br>ms</th>
                        <th>Fall<br>ms</th>
                        <th>Rotor<br>torque k</th>
                        <th>Force σ</th>
                        <th>Error<br>m</th>
                    </tr>
                </thead>
                <tbody id="configuration-rows">
                    <tr><td colspan="10">Waiting for parameters…</td></tr>
                </tbody>
            </table>
        </div>
        <div class="legend">
            Mass, thrust/weight, torque/inertia, frame span, motor time
            constants, rotor torque constant, and random-force standard
            deviation are fixed per drone. “Error” is the live distance from
            that drone’s physical origin.
        </div>
    </aside>
    <script id="raptor-dashboard-script">
    (() => {
        const rows = document.getElementById("configuration-rows");
        const phaseName = document.getElementById("phase-name");
        const phaseTime = document.getElementById("phase-time");
        const phaseEvent = document.getElementById("phase-event");
        const marker = document.getElementById("timeline-marker");
        const connectionDot = document.getElementById("connection-dot");
        const connectionText = document.getElementById("connection-text");
        let latestParameters = [];
        let eventTimer = null;

        const fixed = (value, digits) =>
            Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";

        function renderConfigurations(parameters) {
            latestParameters = parameters;
            rows.innerHTML = parameters.map((parameter, index) => {
                const demo = parameter.demo || {};
                return `<tr>
                    <td>${demo.label || `Drone ${index + 1}`}</td>
                    <td>${fixed(demo.mass_kg, 3)}</td>
                    <td>${fixed(demo.thrust_to_weight, 2)}</td>
                    <td>${fixed(demo.torque_to_inertia, 0)}</td>
                    <td>${fixed(demo.rotor_span_m, 3)}</td>
                    <td>${fixed(1000 * demo.motor_rise_s, 0)}</td>
                    <td>${fixed(1000 * demo.motor_fall_s, 0)}</td>
                    <td>${fixed(demo.rotor_torque_constant, 4)}</td>
                    <td>${fixed(demo.disturbance_force_std, 3)}</td>
                    <td id="error-${index}" class="error-good">—</td>
                </tr>`;
            }).join("");
        }

        function updateState(demo) {
            if (!demo) return;
            phaseName.textContent = `Episode ${demo.episode} · ${demo.phase_name}`;
            phaseTime.textContent =
                `${fixed(demo.time_s, 2)} / ${fixed(demo.duration_s, 2)} s`;
            marker.style.left =
                `${Math.max(0, Math.min(100, demo.progress * 100))}%`;
            (demo.position_errors_m || []).forEach((error, index) => {
                const cell = document.getElementById(`error-${index}`);
                if (cell) cell.textContent = fixed(error, 3);
            });
            if (demo.event) {
                phaseEvent.textContent = demo.event;
                clearTimeout(eventTimer);
                eventTimer = setTimeout(() => {
                    phaseEvent.textContent = "";
                }, 1800);
            }
        }

        function connect() {
            const protocol = location.protocol === "https:" ? "wss" : "ws";
            const pagePath = location.pathname.replace(/\\/$/, "");
            const ws = new WebSocket(
                `${protocol}://${location.host}${pagePath}/ui`
            );
            ws.onopen = () => {
                connectionDot.classList.add("connected");
                connectionText.textContent = "Live telemetry connected";
            };
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                if (message.channel === "setParameters" &&
                    Array.isArray(message.data)) {
                    renderConfigurations(message.data);
                } else if (message.channel === "setState") {
                    updateState(message.data && message.data.demo);
                }
            };
            ws.onclose = () => {
                connectionDot.classList.remove("connected");
                connectionText.textContent = "Reconnecting…";
                setTimeout(connect, 500);
            };
            ws.onerror = () => ws.close();
        }
        connect();
    })();
    </script>
    """
    visualizer_nav = ""
    if peer_href is not None:
        visualizer_nav = f"""
        <nav class="visualizer-nav" aria-label="Visualizer pages">
            <span>Quadcopter</span>
            <a href="{peer_href}">
                Firefly hexacopter →
            </a>
        </nav>
        """
    dashboard_body = dashboard_body.replace(
        "__VISUALIZER_NAV__",
        visualizer_nav,
        1,
    )
    if "</head>" not in original or "</body>" not in original:
        raise RuntimeError("Unexpected ui-server index document")
    dashboard_body = dashboard_body.replace(
        "__GRID_SPEC__",
        (
            f"{GRID_SIZE} × {GRID_SIZE} cells · {N_DRONES} occupied · "
            f"{GRID_SPACING_M:.2f} m spacing · "
            f"{GRID_CENTER_SPAN_M:.2f} × {GRID_CENTER_SPAN_M:.2f} m "
            "center span (visual only)"
        ),
        1,
    )
    return original.replace("</head>", dashboard_css + "</head>", 1).replace(
        "</body>",
        dashboard_body + "</body>",
        1,
    )


async def start_loopback_server(
    port: int,
    *,
    index_transform: Any = _dashboard_html,
) -> tuple[Any, Any, int]:
    from aiohttp import web
    from ui_server import ui_server as upstream

    state = upstream.State()
    state.scenario = "generic"
    static_path = (
        Path(upstream.__file__).with_suffix("").parent
        / "rl-tools"
        / "static"
        / "ui_server"
        / "generic"
    )
    app = web.Application()
    app["state"] = state
    app["static_path"] = static_path

    async def dashboard_index(_request: Any) -> Any:
        original = (static_path / "index.html").read_text(encoding="utf-8")
        return web.Response(
            text=index_transform(original),
            content_type="text/html",
        )

    app.add_routes(
        [
            web.get("/", dashboard_index),
            web.get("/ui", upstream.websocket_handler),
            web.get("/backend", upstream.websocket_handler),
            web.get("/scenario", upstream.handle_scenario),
            web.get("/conta/{hash}", upstream.handle_conta),
            web.get("/debug/{tail:.*}", upstream.handle_debug),
            web.get("/{tail:.*}", upstream.handle_static),
        ]
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    sockets = site._server.sockets if site._server is not None else []
    if not sockets:
        await runner.cleanup()
        raise RuntimeError("UI server did not create a listening socket")
    actual_port = int(sockets[0].getsockname()[1])
    return runner, site, actual_port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAPTOR robust position-hold simulation and browser UI",
    )
    parser.add_argument("--port", type=int, default=13337)
    parser.add_argument(
        "--hex-href",
        default=None,
        help="show a navigation link to this hexacopter page URL",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--loop",
        action="store_true",
        help="continuously replay the 16-second scenario",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="run simulation as fast as possible",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run headlessly, write results, and fail if thresholds are missed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/position_hold"),
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.check:
        metrics, trajectory = await run_episode(
            args.seed,
            realtime=False,
        )
        write_results(args.output_dir, metrics, trajectory)
        print(json.dumps(metrics, indent=2))
        return 0 if metrics["passed"] else 1

    index_transform = (
        _dashboard_html
        if args.hex_href is None
        else lambda original: _dashboard_html(
            original,
            peer_href=args.hex_href,
        )
    )
    runner, _site, actual_port = await start_loopback_server(
        args.port,
        index_transform=index_transform,
    )
    if getattr(args, "announce", True):
        print(f"RAPTOR UI listening on http://127.0.0.1:{actual_port}")
        print(
            "Forward it from your local machine with: "
            f"ssh -N -L {actual_port}:127.0.0.1:{actual_port} "
            "user@remote-host"
        )
        print(
            "Then open: "
            f"http://127.0.0.1:{actual_port}/?L2FDisplayActions=true"
        )

    import websockets

    try:
        uri = f"ws://127.0.0.1:{actual_port}/backend"
        async with websockets.connect(uri, max_size=100_000_000) as websocket:
            handshake = json.loads(await websocket.recv())
            if handshake.get("channel") != "handshake":
                raise RuntimeError(f"Unexpected UI handshake: {handshake}")
            ui = l2f.UI()
            ui.ns = handshake["data"]["namespace"]
            episode = 0
            while True:
                episode += 1
                metrics, trajectory = await run_episode(
                    args.seed,
                    websocket=websocket,
                    ui=ui,
                    realtime=not args.no_realtime,
                    episode=episode,
                )
                metrics["run"]["episode"] = episode
                write_results(args.output_dir, metrics, trajectory)
                print(
                    f"episode={episode} passed={metrics['passed']} "
                    f"metrics={args.output_dir / 'metrics.json'}"
                )
                if not args.loop:
                    return 0 if metrics["passed"] else 1
                if not args.no_realtime:
                    await asyncio.sleep(1.0)
    finally:
        await runner.cleanup()


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
