"""Evaluate the unchanged RAPTOR quadrotor policy on a six-rotor model.

The rigid-body and actuator parameters come from ETH Zurich's RotorS AscTec
Firefly model.  RAPTOR still emits its native four virtual motor commands.
Those commands are converted to a desired collective-thrust/body-torque
wrench and a bounded allocator maps that wrench to the Firefly's six motors.

This is deliberately a lightweight 6-DoF dynamics test, not a claim that the
full RotorS/Gazebo model (aerodynamics, sensors, contacts, and estimator) is
being executed.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import asyncio
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import l2f
from foundation_policy import QuadrotorPolicy

from demo.position_hold import start_loopback_server


CONTROL_HZ = 100
DT = 1.0 / CONTROL_HZ
TOTAL_STEPS = 16 * CONTROL_HZ
KICK_STEP = 4 * CONTROL_HZ
PAYLOAD_STEP = 8 * CONTROL_HZ
KICK_VELOCITY = np.array([1.0, -0.7, 0.4], dtype=np.float64)
PAYLOAD_FACTOR = 1.25

# These acceptance limits intentionally match the existing quadrotor demo.
THRESHOLDS = {
    "pre_kick_error_m": 0.12,
    "kick_peak_error_m": 0.40,
    "pre_payload_error_m": 0.12,
    "payload_peak_error_m": 0.15,
    "final_error_m": 0.10,
}

FIREFLY_SOURCE = (
    "https://github.com/ethz-asl/rotors_simulator/blob/master/"
    "rotors_description/urdf/firefly.xacro"
)


@dataclass(frozen=True)
class FireflyModel:
    """RotorS Firefly parameters expressed in RAPTOR's FLU convention."""

    mass_kg: float
    inertia_kg_m2: np.ndarray
    arm_length_m: float
    rotor_z_m: float
    rotor_radius_m: float
    motor_constant_n_per_rad_s_sq: float
    moment_constant_m: float
    time_constant_up_s: float
    time_constant_down_s: float
    max_rotor_speed_rad_s: float
    rotor_positions_m: np.ndarray
    spin_directions: tuple[str, ...]
    yaw_signs: np.ndarray

    @property
    def max_rotor_thrust_n(self) -> float:
        return (
            self.motor_constant_n_per_rad_s_sq
            * self.max_rotor_speed_rad_s**2
        )

    @property
    def thrust_to_weight(self) -> float:
        return (
            6 * self.max_rotor_thrust_n / (self.mass_kg * 9.81)
        )

    @property
    def hover_rotor_speed_rad_s(self) -> float:
        return math.sqrt(
            self.mass_kg
            * 9.81
            / (6 * self.motor_constant_n_per_rad_s_sq)
        )


def firefly_model() -> FireflyModel:
    """Return the parameters published in RotorS ``firefly.xacro``."""

    arm = 0.215
    sin30 = 0.5
    cos30 = 0.866025403784
    positions = np.array(
        [
            [cos30 * arm, sin30 * arm, 0.037],  # front left, CCW
            [0.0, arm, 0.037],                  # left, CW
            [-cos30 * arm, sin30 * arm, 0.037], # back left, CCW
            [-cos30 * arm, -sin30 * arm, 0.037],# back right, CW
            [0.0, -arm, 0.037],                 # right, CCW
            [cos30 * arm, -sin30 * arm, 0.037], # front right, CW
        ],
        dtype=np.float64,
    )
    spins = ("CCW", "CW", "CCW", "CW", "CCW", "CW")
    # Positive body-z reaction torque for CCW, negative for CW.
    yaw_signs = np.array([1, -1, 1, -1, 1, -1], dtype=np.float64)
    return FireflyModel(
        mass_kg=1.5,
        inertia_kg_m2=np.diag([0.0347563, 0.0458929, 0.0977]),
        arm_length_m=arm,
        rotor_z_m=0.037,
        rotor_radius_m=0.1,
        motor_constant_n_per_rad_s_sq=8.54858e-06,
        moment_constant_m=0.016,
        time_constant_up_s=0.0125,
        time_constant_down_s=0.025,
        max_rotor_speed_rad_s=838.0,
        rotor_positions_m=positions,
        spin_directions=spins,
        yaw_signs=yaw_signs,
    )


def allocation_matrix(
    positions: np.ndarray,
    yaw_signs: np.ndarray,
    moment_constant_m: float,
) -> np.ndarray:
    """Map upward rotor thrusts to [Fz, tau_x, tau_y, tau_z]."""

    return np.vstack(
        [
            np.ones(len(positions), dtype=np.float64),
            positions[:, 1],
            -positions[:, 0],
            yaw_signs * moment_constant_m,
        ]
    )


class BoundedAllocator:
    """Small bounded least-squares allocator specialized to six motors."""

    def __init__(self, model: FireflyModel) -> None:
        self.model = model
        self.physical_matrix = allocation_matrix(
            model.rotor_positions_m,
            model.yaw_signs,
            model.moment_constant_m,
        )

        # RAPTOR's motor order is front-right, back-right, back-left,
        # front-left.  A virtual X quad with the same total maximum thrust
        # provides a concrete interpretation for its four outputs.
        corner = model.arm_length_m / math.sqrt(2.0)
        virtual_positions = np.array(
            [
                [corner, -corner, 0.0],
                [-corner, -corner, 0.0],
                [-corner, corner, 0.0],
                [corner, corner, 0.0],
            ],
            dtype=np.float64,
        )
        virtual_yaw_signs = np.array([-1, 1, -1, 1], dtype=np.float64)
        self.virtual_matrix = allocation_matrix(
            virtual_positions,
            virtual_yaw_signs,
            model.moment_constant_m,
        )
        self.virtual_max_thrust_n = 1.5 * model.max_rotor_thrust_n
        self.physical_max_thrust_n = model.max_rotor_thrust_n
        self._physical_pinv = np.linalg.pinv(self.physical_matrix)

        # Normalize force and torque residuals before comparing allocator
        # candidates; otherwise newtons would dominate newton-metres.
        total = 6 * self.physical_max_thrust_n
        self.residual_scales = np.array(
            [
                total,
                total * model.arm_length_m,
                total * model.arm_length_m,
                total * model.moment_constant_m,
            ],
            dtype=np.float64,
        )
        self._normalized_matrix = (
            self.physical_matrix / self.residual_scales[:, None]
        )
        self._bounded_cases = self._precompute_bounded_cases()

    def _precompute_bounded_cases(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Precompute all lower/free/upper active-set pseudoinverses."""

        cases = []
        for status_tuple in itertools.product((-1, 0, 1), repeat=6):
            status = np.asarray(status_tuple, dtype=np.int8)
            free = np.flatnonzero(status == 0)
            fixed = np.flatnonzero(status != 0)
            fixed_values = np.where(
                status[fixed] > 0,
                self.physical_max_thrust_n,
                0.0,
            )
            free_pinv = (
                np.linalg.pinv(self._normalized_matrix[:, free])
                if len(free)
                else np.empty((0, 4), dtype=np.float64)
            )
            cases.append((free, fixed, fixed_values, free_pinv))
        return cases

    def desired_wrench(
        self,
        virtual_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert four raw RAPTOR outputs into a virtual-quad wrench."""

        clipped = np.clip(np.asarray(virtual_action, dtype=np.float64), -1, 1)
        normalized_speed = (clipped + 1.0) / 2.0
        virtual_thrusts = (
            self.virtual_max_thrust_n * normalized_speed**2
        )
        return self.virtual_matrix @ virtual_thrusts, virtual_thrusts

    def allocate(self, wrench: np.ndarray) -> tuple[np.ndarray, float]:
        """Allocate a wrench and return thrusts plus normalized residual."""

        wrench = np.asarray(wrench, dtype=np.float64)
        unconstrained = self._physical_pinv @ wrench
        tolerance = 1e-9
        if (
            np.all(unconstrained >= -tolerance)
            and np.all(
                unconstrained <= self.physical_max_thrust_n + tolerance
            )
        ):
            thrusts = np.clip(
                unconstrained,
                0.0,
                self.physical_max_thrust_n,
            )
        else:
            normalized_wrench = wrench / self.residual_scales
            best_cost = math.inf
            best_thrusts: np.ndarray | None = None
            for free, fixed, fixed_values, free_pinv in self._bounded_cases:
                candidate = np.zeros(6, dtype=np.float64)
                if len(fixed):
                    candidate[fixed] = fixed_values
                    rhs = (
                        normalized_wrench
                        - self._normalized_matrix[:, fixed] @ fixed_values
                    )
                else:
                    rhs = normalized_wrench
                if len(free):
                    candidate[free] = free_pinv @ rhs
                    if (
                        np.any(candidate[free] < -tolerance)
                        or np.any(
                            candidate[free]
                            > self.physical_max_thrust_n + tolerance
                        )
                    ):
                        continue
                residual = (
                    self._normalized_matrix @ candidate
                    - normalized_wrench
                )
                # A minute regularizer selects a balanced solution among
                # otherwise equivalent allocations.
                cost = float(residual @ residual)
                cost += 1e-12 * float(candidate @ candidate)
                if cost < best_cost:
                    best_cost = cost
                    best_thrusts = candidate
            if best_thrusts is None:
                raise RuntimeError("No feasible hexacopter allocation found")
            thrusts = np.clip(
                best_thrusts,
                0.0,
                self.physical_max_thrust_n,
            )

        residual = self.physical_matrix @ thrusts - wrench
        normalized_residual = float(
            np.linalg.norm(residual / self.residual_scales)
        )
        return thrusts, normalized_residual

    def command(
        self,
        virtual_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        wrench, virtual_thrusts = self.desired_wrench(virtual_action)
        physical_thrusts, residual = self.allocate(wrench)
        rotor_speed_commands = np.sqrt(
            physical_thrusts
            / self.model.motor_constant_n_per_rad_s_sq
        )
        return (
            rotor_speed_commands,
            physical_thrusts,
            virtual_thrusts,
            residual,
        )


@dataclass
class HexState:
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    rotor_speeds: np.ndarray
    last_virtual_action: np.ndarray


@dataclass
class HexRuntime:
    model: FireflyModel
    allocator: BoundedAllocator
    policy: Any
    state: HexState
    mass_kg: float
    inertia_kg_m2: np.ndarray


def _axis_angle_quaternion(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.concatenate(([math.cos(half)], axis * math.sin(half)))


def _rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
            ],
            [
                2 * x * y + 2 * w * z,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * w * x,
            ],
            [
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x * x - 2 * y * y,
            ],
        ],
        dtype=np.float64,
    )


def _quaternion_derivative(
    q: np.ndarray,
    angular_velocity: np.ndarray,
) -> np.ndarray:
    w, x, y, z = q
    wx, wy, wz = angular_velocity
    return 0.5 * np.array(
        [
            -x * wx - y * wy - z * wz,
            w * wx + y * wz - z * wy,
            w * wy + z * wx - x * wz,
            w * wz + x * wy - y * wx,
        ],
        dtype=np.float64,
    )


def build_runtime() -> HexRuntime:
    model = firefly_model()
    allocator = BoundedAllocator(model)
    hover_virtual_thrust = model.mass_kg * 9.81 / 4.0
    hover_virtual_speed = math.sqrt(
        hover_virtual_thrust / allocator.virtual_max_thrust_n
    )
    hover_virtual_action = np.full(
        4,
        2 * hover_virtual_speed - 1,
        dtype=np.float64,
    )
    state = HexState(
        position=np.array([0.12, -0.08, 0.08], dtype=np.float64),
        orientation=_axis_angle_quaternion(
            np.array([1.0, -0.6, 0.2]),
            math.radians(30),
        ),
        linear_velocity=np.zeros(3, dtype=np.float64),
        angular_velocity=np.zeros(3, dtype=np.float64),
        rotor_speeds=np.full(
            6,
            model.hover_rotor_speed_rad_s,
            dtype=np.float64,
        ),
        last_virtual_action=hover_virtual_action,
    )
    policy = QuadrotorPolicy()
    policy.reset()
    return HexRuntime(
        model=model,
        allocator=allocator,
        policy=policy,
        state=state,
        mass_kg=model.mass_kg,
        inertia_kg_m2=model.inertia_kg_m2.copy(),
    )


def observation(state: HexState) -> np.ndarray:
    """Construct RAPTOR's 22-value FLU observation."""

    value = np.concatenate(
        (
            state.position,
            _rotation_matrix(state.orientation).reshape(-1),
            state.linear_velocity,
            state.angular_velocity,
            state.last_virtual_action,
        )
    )
    if value.shape != (22,):
        raise RuntimeError(f"Unexpected observation shape: {value.shape}")
    return value.astype(np.float32)[None, :]


def _pack_state(state: HexState) -> np.ndarray:
    return np.concatenate(
        (
            state.position,
            state.orientation,
            state.linear_velocity,
            state.angular_velocity,
            state.rotor_speeds,
        )
    )


def _unpack_state(vector: np.ndarray, last_action: np.ndarray) -> HexState:
    return HexState(
        position=vector[0:3].copy(),
        orientation=vector[3:7].copy(),
        linear_velocity=vector[7:10].copy(),
        angular_velocity=vector[10:13].copy(),
        rotor_speeds=vector[13:19].copy(),
        last_virtual_action=np.asarray(last_action, dtype=np.float64).copy(),
    )


def _derivative(
    runtime: HexRuntime,
    vector: np.ndarray,
    rotor_speed_commands: np.ndarray,
) -> np.ndarray:
    state = _unpack_state(vector, runtime.state.last_virtual_action)
    model = runtime.model
    inertia = runtime.inertia_kg_m2
    inertia_inv = np.linalg.inv(inertia)

    rotor_thrusts = (
        model.motor_constant_n_per_rad_s_sq * state.rotor_speeds**2
    )
    wrench = runtime.allocator.physical_matrix @ rotor_thrusts
    thrust_body = np.array([0.0, 0.0, wrench[0]], dtype=np.float64)
    linear_acceleration = (
        _rotation_matrix(state.orientation) @ thrust_body / runtime.mass_kg
        + np.array([0.0, 0.0, -9.81], dtype=np.float64)
    )
    angular_momentum = inertia @ state.angular_velocity
    angular_acceleration = inertia_inv @ (
        wrench[1:] - np.cross(state.angular_velocity, angular_momentum)
    )
    time_constants = np.where(
        rotor_speed_commands >= state.rotor_speeds,
        model.time_constant_up_s,
        model.time_constant_down_s,
    )
    rotor_acceleration = (
        rotor_speed_commands - state.rotor_speeds
    ) / time_constants
    return np.concatenate(
        (
            state.linear_velocity,
            _quaternion_derivative(
                state.orientation,
                state.angular_velocity,
            ),
            linear_acceleration,
            angular_acceleration,
            rotor_acceleration,
        )
    )


def integrate(
    runtime: HexRuntime,
    rotor_speed_commands: np.ndarray,
) -> None:
    """Advance the six-rotor rigid body one 100 Hz step with RK4."""

    initial = _pack_state(runtime.state)
    k1 = _derivative(runtime, initial, rotor_speed_commands)
    k2 = _derivative(
        runtime,
        initial + DT * k1 / 2.0,
        rotor_speed_commands,
    )
    k3 = _derivative(
        runtime,
        initial + DT * k2 / 2.0,
        rotor_speed_commands,
    )
    k4 = _derivative(
        runtime,
        initial + DT * k3,
        rotor_speed_commands,
    )
    final = initial + DT * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    final[3:7] /= np.linalg.norm(final[3:7])
    final[13:19] = np.clip(
        final[13:19],
        0.0,
        runtime.model.max_rotor_speed_rad_s,
    )
    last_action = runtime.state.last_virtual_action
    runtime.state = _unpack_state(final, last_action)


def _phase_at_step(completed_steps: int) -> tuple[str, str]:
    if completed_steps < KICK_STEP:
        return "initial_recovery", "Recovering from a 30° initial tilt"
    if completed_steps < PAYLOAD_STEP:
        return "kick_recovery", "Recovering from the velocity kick"
    return "payload_adaptation", "Adapting to a 25% payload increase"


def _settling_time(
    errors: np.ndarray,
    start: int,
    stop: int,
    threshold: float,
    stable_steps: int = 50,
) -> float | None:
    for index in range(start, stop - stable_steps + 1):
        if float(np.max(errors[index : index + stable_steps])) <= threshold:
            return (index + 1) * DT
    return None


def build_metrics(
    position_errors: np.ndarray,
    orientation_errors: np.ndarray,
    allocation_residuals: np.ndarray,
    saturated_motors: np.ndarray,
    maximum_absolute_action: float,
    elapsed_s: float,
) -> dict[str, Any]:
    pre_kick = float(position_errors[KICK_STEP - 1])
    kick_peak = float(np.max(position_errors[KICK_STEP:PAYLOAD_STEP]))
    pre_payload = float(position_errors[PAYLOAD_STEP - 1])
    payload_peak = float(np.max(position_errors[PAYLOAD_STEP:]))
    final = float(position_errors[-1])
    finite = bool(
        np.isfinite(position_errors).all()
        and np.isfinite(orientation_errors).all()
        and np.isfinite(allocation_residuals).all()
    )
    passed = (
        finite
        and pre_kick <= THRESHOLDS["pre_kick_error_m"]
        and kick_peak <= THRESHOLDS["kick_peak_error_m"]
        and pre_payload <= THRESHOLDS["pre_payload_error_m"]
        and payload_peak <= THRESHOLDS["payload_peak_error_m"]
        and final <= THRESHOLDS["final_error_m"]
    )
    model = firefly_model()
    return {
        "run": {
            "control_hz": CONTROL_HZ,
            "simulated_duration_s": TOTAL_STEPS * DT,
            "wall_time_s": elapsed_s,
            "model": "RotorS AscTec Firefly",
            "model_source": FIREFLY_SOURCE,
            "policy_outputs": 4,
            "physical_motors": 6,
            "adapter": "virtual X-quad wrench -> bounded hex allocation",
            "checkpoint_package": "foundation-policy==1.0.1",
            "maximum_absolute_policy_action": maximum_absolute_action,
        },
        "model": {
            "mass_kg": model.mass_kg,
            "inertia_kg_m2": model.inertia_kg_m2.tolist(),
            "arm_length_m": model.arm_length_m,
            "rotor_radius_m": model.rotor_radius_m,
            "max_rotor_speed_rad_s": model.max_rotor_speed_rad_s,
            "motor_constant_n_per_rad_s_sq": (
                model.motor_constant_n_per_rad_s_sq
            ),
            "moment_constant_m": model.moment_constant_m,
            "time_constant_up_s": model.time_constant_up_s,
            "time_constant_down_s": model.time_constant_down_s,
            "thrust_to_weight": model.thrust_to_weight,
            "rotor_positions_m": model.rotor_positions_m.tolist(),
            "spin_directions": list(model.spin_directions),
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
                "mass_and_inertia_factor": PAYLOAD_FACTOR,
            },
        ],
        "thresholds": THRESHOLDS,
        "results": {
            "pre_kick_error_m": pre_kick,
            "kick_peak_error_m": kick_peak,
            "kick_recovery_time_s": _settling_time(
                position_errors,
                KICK_STEP,
                PAYLOAD_STEP,
                THRESHOLDS["pre_payload_error_m"],
            ),
            "pre_payload_error_m": pre_payload,
            "payload_peak_error_m": payload_peak,
            "final_error_m": final,
            "maximum_orientation_error_deg": float(
                np.degrees(np.max(orientation_errors))
            ),
            "final_orientation_error_deg": float(
                np.degrees(orientation_errors[-1])
            ),
            "maximum_normalized_allocation_residual": float(
                np.max(allocation_residuals)
            ),
            "steps_with_motor_saturation": int(
                np.count_nonzero(saturated_motors)
            ),
            "finite": finite,
        },
        "passed": passed,
        "scope": (
            "Lightweight rigid-body/actuator simulation using RotorS "
            "parameters; not full Gazebo, estimator, or hardware validation."
        ),
    }


def parameters_message(runtime: HexRuntime, namespace: str) -> str:
    model = runtime.model
    dynamics = {
        "mass": runtime.mass_kg,
        "J": runtime.inertia_kg_m2.tolist(),
        "J_inv": np.linalg.inv(runtime.inertia_kg_m2).tolist(),
        "gravity": [0.0, 0.0, -9.81],
        "rotor_positions": model.rotor_positions_m.tolist(),
        "rotor_thrust_directions": [[0.0, 0.0, 1.0]] * 6,
        "rotor_torque_directions": [
            [0.0, 0.0, float(sign)] for sign in model.yaw_signs
        ],
        "rotor_thrust_coefficients": [
            [0.0, 0.0, model.motor_constant_n_per_rad_s_sq]
        ] * 6,
        "rotor_torque_constants": [model.moment_constant_m] * 6,
        "rotor_time_constants_rising": [model.time_constant_up_s] * 6,
        "rotor_time_constants_falling": [model.time_constant_down_s] * 6,
        "hovering_throttle_relative": (
            model.hover_rotor_speed_rad_s / model.max_rotor_speed_rad_s
        ),
        "action_limit": {
            "min": 0.0,
            "max": model.max_rotor_speed_rad_s,
        },
    }
    payload = {
        "namespace": namespace,
        "channel": "setParameters",
        "latch": True,
        "data": {
            "dynamics": dynamics,
            "integration": {"dt": DT},
            "demo": {
                "model": "RotorS AscTec Firefly",
                "source": FIREFLY_SOURCE,
                "policy_outputs": 4,
                "physical_motors": 6,
            },
        },
    }
    return json.dumps(payload)


def ui_message(namespace: str) -> str:
    device = l2f.Device()
    environment = l2f.Environment()
    l2f.initialize_environment(device, environment)
    ui = l2f.UI()
    ui.ns = namespace
    payload = json.loads(l2f.set_ui_message(device, environment, ui))
    render_function = payload["data"]["render_function"]
    old = """export async function render(ui_state, parameters, state, action) {
    ui_state.drone.get().position.set(...clip_position(parameters.dynamics.mass, state.position))
    ui_state.drone.get().quaternion.copy(new THREE.Quaternion(state.orientation[1], state.orientation[2], state.orientation[3], state.orientation[0]).normalize())
    update_camera(ui_state)
}"""
    new = """export async function render(ui_state, parameters, state, action) {
    ui_state.drone.get().position.set(...clip_position(parameters.dynamics.mass, state.position))
    ui_state.drone.get().quaternion.copy(new THREE.Quaternion(state.orientation[1], state.orientation[2], state.orientation[3], state.orientation[0]).normalize())
    const speeds = state.rotor_speeds_normalized || []
    ui_state.drone.set_action(speeds)
    update_camera(ui_state)
}"""
    if render_function.count(old) != 1:
        raise RuntimeError("Unexpected L2F single-drone renderer")
    render_function = render_function.replace(old, new, 1)
    old_action_loop = """    set_action(action){
        for(let i = 0; i < 4; i++){
            const forceArrow = this.rotors[i].forceArrow
            const force_magnitude = action[i]
            forceArrow.setDirection(new THREE.Vector3(0, 0, force_magnitude))
            forceArrow.setLength(Math.cbrt(this.scale)/10)
        }
    }"""
    new_action_loop = """    set_action(action){
        for(let i = 0; i < this.rotors.length; i++){
            const forceArrow = this.rotors[i].forceArrow
            const force_magnitude = Math.max(0, action[i] || 0)
            forceArrow.setDirection(new THREE.Vector3(0, 0, 1))
            forceArrow.setLength(
                Math.cbrt(this.scale) / 10 * force_magnitude
            )
        }
    }"""
    if render_function.count(old_action_loop) != 1:
        raise RuntimeError("Unexpected L2F action-arrow renderer")
    render_function = render_function.replace(
        old_action_loop,
        new_action_loop,
        1,
    )
    color_expression = (
        "this.parameters.dynamics.rotor_thrust_directions"
        "[rotorIndex][2] < 0"
    )
    if render_function.count(color_expression) != 1:
        raise RuntimeError("Unexpected L2F rotor-color renderer")
    render_function = render_function.replace(
        color_expression,
        (
            "this.parameters.dynamics.rotor_torque_directions"
            "[rotorIndex][2] < 0"
        ),
        1,
    )
    payload["data"]["render_function"] = render_function
    payload["data"]["options"] = {
        # The pinned single-drone UI's showAxes branch references an undefined
        # `scale` variable in episode_init and prevents episode initialization.
        "showAxes": False,
        "camera_position": [0.8, 0.8, 1.4],
    }
    payload["latch"] = True
    return json.dumps(payload)


def state_message(
    runtime: HexRuntime,
    virtual_action: np.ndarray,
    allocation_residual: float,
    saturated_motors: int,
    completed_steps: int,
    episode: int,
) -> str:
    state = runtime.state
    phase_id, phase_name = _phase_at_step(completed_steps)
    event = None
    if completed_steps == KICK_STEP:
        event = "Velocity kick applied"
    elif completed_steps == PAYLOAD_STEP:
        event = "25% mass and inertia payload applied"
    orientation_angle = 2 * math.acos(
        float(np.clip(abs(state.orientation[0]), 0.0, 1.0))
    )
    payload = {
        "channel": "setState",
        "latch": True,
        "data": {
            "state": {
                "position": state.position.tolist(),
                "orientation": state.orientation.tolist(),
                "linear_velocity": state.linear_velocity.tolist(),
                "angular_velocity": state.angular_velocity.tolist(),
                "rotor_speeds_normalized": (
                    state.rotor_speeds
                    / runtime.model.max_rotor_speed_rad_s
                ).tolist(),
            },
            "action": np.asarray(virtual_action).tolist(),
            "demo": {
                "episode": episode,
                "step": completed_steps,
                "time_s": completed_steps * DT,
                "duration_s": TOTAL_STEPS * DT,
                "progress": completed_steps / TOTAL_STEPS,
                "phase_id": phase_id,
                "phase_name": phase_name,
                "event": event,
                "position_error_m": float(np.linalg.norm(state.position)),
                "orientation_error_deg": math.degrees(orientation_angle),
                "allocation_residual": allocation_residual,
                "saturated_motors": saturated_motors,
                "mass_kg": runtime.mass_kg,
                "virtual_action": np.asarray(virtual_action).tolist(),
                "physical_rotor_speed_rad_s": state.rotor_speeds.tolist(),
            },
        },
    }
    return json.dumps(payload)


async def run_episode(
    *,
    websocket: Any | None = None,
    namespace: str = "",
    realtime: bool = False,
    episode: int = 1,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    runtime = build_runtime()
    if websocket is not None:
        await websocket.send(ui_message(namespace))
        await websocket.send(parameters_message(runtime, namespace))

    position = np.zeros((TOTAL_STEPS, 3), dtype=np.float64)
    orientation = np.zeros((TOTAL_STEPS, 4), dtype=np.float64)
    policy_actions = np.zeros((TOTAL_STEPS, 4), dtype=np.float64)
    motor_commands = np.zeros((TOTAL_STEPS, 6), dtype=np.float64)
    motor_speeds = np.zeros((TOTAL_STEPS, 6), dtype=np.float64)
    allocation_residuals = np.zeros(TOTAL_STEPS, dtype=np.float64)
    saturated_motors = np.zeros(TOTAL_STEPS, dtype=np.int64)
    position_errors = np.zeros(TOTAL_STEPS, dtype=np.float64)
    orientation_errors = np.zeros(TOTAL_STEPS, dtype=np.float64)
    maximum_absolute_action = 0.0
    start = time.perf_counter()
    deadline = start

    for step_index in range(TOTAL_STEPS):
        virtual_action = np.asarray(
            runtime.policy.evaluate_step(observation(runtime.state))[0],
            dtype=np.float64,
        )
        if virtual_action.shape != (4,) or not np.isfinite(
            virtual_action
        ).all():
            raise RuntimeError(
                f"Invalid policy action at step {step_index}: "
                f"{virtual_action}"
            )
        maximum_absolute_action = max(
            maximum_absolute_action,
            float(np.max(np.abs(virtual_action))),
        )
        (
            rotor_speed_commands,
            physical_thrusts,
            _virtual_thrusts,
            allocation_residual,
        ) = runtime.allocator.command(virtual_action)
        saturation_count = int(
            np.count_nonzero(
                (physical_thrusts <= 1e-7)
                | (
                    physical_thrusts
                    >= runtime.model.max_rotor_thrust_n - 1e-7
                )
            )
        )
        integrate(runtime, rotor_speed_commands)
        runtime.state.last_virtual_action = virtual_action.copy()

        completed_steps = step_index + 1
        if completed_steps == KICK_STEP:
            runtime.state.linear_velocity += KICK_VELOCITY
        if completed_steps == PAYLOAD_STEP:
            runtime.mass_kg *= PAYLOAD_FACTOR
            runtime.inertia_kg_m2 *= PAYLOAD_FACTOR

        position[step_index] = runtime.state.position
        orientation[step_index] = runtime.state.orientation
        policy_actions[step_index] = virtual_action
        motor_commands[step_index] = rotor_speed_commands
        motor_speeds[step_index] = runtime.state.rotor_speeds
        allocation_residuals[step_index] = allocation_residual
        saturated_motors[step_index] = saturation_count
        position_errors[step_index] = np.linalg.norm(
            runtime.state.position
        )
        orientation_errors[step_index] = 2 * math.acos(
            float(np.clip(abs(runtime.state.orientation[0]), 0.0, 1.0))
        )

        if websocket is not None:
            await websocket.send(
                state_message(
                    runtime,
                    virtual_action,
                    allocation_residual,
                    saturation_count,
                    completed_steps,
                    episode,
                )
            )
        if realtime:
            deadline += DT
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

    elapsed_s = time.perf_counter() - start
    metrics = build_metrics(
        position_errors,
        orientation_errors,
        allocation_residuals,
        saturated_motors,
        maximum_absolute_action,
        elapsed_s,
    )
    trajectory = {
        "position": position,
        "orientation": orientation,
        "policy_action": policy_actions,
        "motor_command_rad_s": motor_commands,
        "motor_speed_rad_s": motor_speeds,
        "allocation_residual": allocation_residuals,
        "saturated_motors": saturated_motors,
        "dt": np.array(DT),
    }
    return metrics, trajectory


def write_results(
    output_dir: Path,
    metrics: dict[str, Any],
    trajectory: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "trajectory.npz", **trajectory)


def _hexacopter_dashboard_html(
    original: str,
    *,
    peer_href: str | None = None,
) -> str:
    model = firefly_model()
    rotor_rows = "\n".join(
        (
            "<tr>"
            f"<td>{index + 1}</td>"
            f"<td>{name}</td>"
            f"<td>{position[0]:+.3f}</td>"
            f"<td>{position[1]:+.3f}</td>"
            f"<td>{position[2]:+.3f}</td>"
            f"<td>{spin}</td>"
            "</tr>"
        )
        for index, (name, position, spin) in enumerate(
            zip(
                (
                    "Front left",
                    "Left",
                    "Back left",
                    "Back right",
                    "Right",
                    "Front right",
                ),
                model.rotor_positions_m,
                model.spin_directions,
            )
        )
    )
    dashboard_css = """
    <style id="raptor-hex-dashboard-style">
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
                radial-gradient(circle at 50% 42%, #17304a 0,
                    #0a1728 52%, #050b14 100%);
            flex: 1 1 auto;
            height: 100vh !important;
            min-width: 0;
            width: auto !important;
        }
        #canvas { box-shadow: none !important; }
        #raptor-hex-dashboard {
            background: rgba(6, 15, 27, 0.98);
            border-left: 1px solid #223b55;
            box-sizing: border-box;
            flex: 0 0 min(560px, 46vw);
            height: 100vh;
            overflow-y: auto;
            padding: 24px;
        }
        #raptor-hex-dashboard h1 {
            color: #f8fafc;
            font-size: 24px;
            line-height: 1.15;
            margin: 0 0 6px;
        }
        #raptor-hex-dashboard h2 {
            color: #f8fafc;
            font-size: 15px;
            margin: 0 0 10px;
        }
        #raptor-hex-dashboard p {
            color: #a9bdd2;
            font-size: 13px;
            line-height: 1.55;
            margin: 7px 0;
        }
        #raptor-hex-dashboard a { color: #67e8f9; }
        .visualizer-nav {
            align-items: center;
            display: flex;
            gap: 7px;
            margin: 12px 0 16px;
        }
        .visualizer-nav a,
        .visualizer-nav span {
            border: 1px solid #28506a;
            border-radius: 999px;
            color: #a5c4da;
            font-size: 11px;
            padding: 6px 10px;
            text-decoration: none;
        }
        .visualizer-nav span {
            background: #0e7490;
            border-color: #22d3ee;
            color: #ecfeff;
            font-weight: 700;
        }
        .visualizer-nav a:hover {
            background: #102b43;
            border-color: #38bdf8;
            color: #e0f2fe;
        }
        .hex-eyebrow {
            color: #67e8f9;
            font-size: 11px;
            font-weight: 750;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .hex-status {
            align-items: center;
            color: #9fb3c8;
            display: flex;
            font-size: 12px;
            gap: 7px;
            margin: 12px 0 18px;
        }
        .hex-status-dot {
            background: #f59e0b;
            border-radius: 50%;
            height: 8px;
            width: 8px;
        }
        .hex-status-dot.connected {
            background: #22c55e;
            box-shadow: 0 0 10px rgba(34, 197, 94, .8);
        }
        .hex-panel {
            background: #0c1b2d;
            border: 1px solid #203a53;
            border-radius: 12px;
            margin: 0 0 14px;
            padding: 15px;
        }
        .hex-flow {
            align-items: center;
            display: grid;
            gap: 5px;
            grid-template-columns: 1fr auto 1fr auto 1fr;
            margin: 12px 0;
            text-align: center;
        }
        .hex-flow-box {
            background: #10263c;
            border: 1px solid #2b4964;
            border-radius: 8px;
            color: #dbeafe;
            font-size: 11px;
            line-height: 1.35;
            padding: 9px 5px;
        }
        .hex-flow-arrow { color: #67e8f9; }
        .hex-phase-title {
            color: #e5f3ff;
            font-size: 14px;
            font-weight: 700;
        }
        .hex-phase-time {
            color: #8da7c0;
            font-size: 12px;
            margin-top: 3px;
        }
        .hex-timeline {
            background: #183047;
            border-radius: 999px;
            height: 8px;
            margin: 13px 0 7px;
            overflow: visible;
            position: relative;
        }
        .hex-timeline::before,
        .hex-timeline::after {
            background: #6b87a1;
            content: "";
            height: 14px;
            position: absolute;
            top: -3px;
            width: 1px;
        }
        .hex-timeline::before { left: 25%; }
        .hex-timeline::after { left: 50%; }
        #hex-timeline-marker {
            background: #67e8f9;
            border: 2px solid #ecfeff;
            border-radius: 50%;
            box-shadow: 0 0 10px rgba(103, 232, 249, .7);
            height: 11px;
            left: 0;
            position: absolute;
            top: -3px;
            transform: translateX(-50%);
            width: 11px;
        }
        .hex-labels {
            color: #7791aa;
            display: grid;
            font-size: 10px;
            grid-template-columns: 1fr 1fr 2fr;
        }
        .hex-event {
            color: #fbbf24;
            font-size: 12px;
            height: 18px;
            margin-top: 8px;
        }
        .hex-metrics {
            display: grid;
            gap: 8px;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin-top: 13px;
        }
        .hex-metric {
            background: #091624;
            border-radius: 8px;
            padding: 9px 10px;
        }
        .hex-metric span {
            color: #7791aa;
            display: block;
            font-size: 10px;
            margin-bottom: 3px;
            text-transform: uppercase;
        }
        .hex-metric strong {
            color: #f1f5f9;
            font-size: 16px;
        }
        .hex-note {
            border-left: 3px solid #f59e0b;
            color: #a9bdd2;
            font-size: 12px;
            line-height: 1.5;
            margin-top: 11px;
            padding-left: 10px;
        }
        .hex-table-wrap { overflow-x: auto; }
        .hex-table {
            border-collapse: collapse;
            font-size: 11px;
            width: 100%;
        }
        .hex-table th {
            color: #7894ad;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: .04em;
            padding: 6px;
            text-align: right;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .hex-table td {
            border-top: 1px solid #19334b;
            color: #c3d5e6;
            padding: 7px 6px;
            text-align: right;
            white-space: nowrap;
        }
        .hex-table th:nth-child(2),
        .hex-table td:nth-child(2) { text-align: left; }
        .hex-params {
            display: grid;
            gap: 7px 12px;
            grid-template-columns: 1fr auto;
            margin-top: 8px;
        }
        .hex-params dt { color: #829bb3; font-size: 11px; }
        .hex-params dd {
            color: #dbeafe;
            font-size: 11px;
            margin: 0;
            text-align: right;
        }
        @media (max-width: 900px) {
            body { flex-direction: column; overflow: auto; }
            .canvas-container {
                flex: 0 0 58vh;
                height: 58vh !important;
                width: 100% !important;
            }
            #raptor-hex-dashboard {
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
    <aside id="raptor-hex-dashboard">
        <div class="hex-eyebrow">Unchanged checkpoint · new airframe</div>
        <h1>RAPTOR on a Firefly hexacopter</h1>
        __VISUALIZER_NAV__
        <p>
            The left view is a six-motor rigid-body simulation using the
            published ETH Zürich RotorS AscTec Firefly parameters. RAPTOR sees
            its normal 22-value observation and still emits four outputs.
        </p>
        <div class="hex-status">
            <span id="hex-status-dot" class="hex-status-dot"></span>
            <span id="hex-status-text">Connecting to live telemetry…</span>
        </div>

        <section class="hex-panel">
            <h2>What is being tested</h2>
            <div class="hex-flow">
                <div class="hex-flow-box">RAPTOR<br>4 virtual motors</div>
                <div class="hex-flow-arrow">→</div>
                <div class="hex-flow-box">Desired wrench<br>Fz, τx, τy, τz</div>
                <div class="hex-flow-arrow">→</div>
                <div class="hex-flow-box">Bounded allocator<br>6 real motors</div>
            </div>
            <p>
                This directly tests whether a virtual-quad adapter can reuse
                the current recurrent policy on a hex frame. The policy
                receives its previous four <em>virtual</em> commands, while
                the plant evolves from all six physical rotor speeds.
            </p>
            <div class="hex-note">
                Scope: 6-DoF rigid body, rotor thrust/yaw moments, asymmetric
                motor lag, limits, and exact published mass/inertia/geometry.
                This is not yet full Gazebo aerodynamics, estimator, PX4
                timing, or hardware validation.
            </div>
        </section>

        <section class="hex-panel">
            <h2>Episode phase</h2>
            <div id="hex-phase-name" class="hex-phase-title">Waiting…</div>
            <div id="hex-phase-time" class="hex-phase-time">0 / 16 s</div>
            <div class="hex-timeline">
                <div id="hex-timeline-marker"></div>
            </div>
            <div class="hex-labels">
                <span>0–4 s tilt recovery</span>
                <span>4–8 s velocity kick</span>
                <span>8–16 s +25% payload</span>
            </div>
            <div id="hex-event" class="hex-event"></div>
            <div class="hex-metrics">
                <div class="hex-metric">
                    <span>Position error</span>
                    <strong id="hex-position-error">—</strong>
                </div>
                <div class="hex-metric">
                    <span>Attitude error</span>
                    <strong id="hex-orientation-error">—</strong>
                </div>
                <div class="hex-metric">
                    <span>Allocation residual</span>
                    <strong id="hex-allocation-residual">—</strong>
                </div>
                <div class="hex-metric">
                    <span>Motors at limit</span>
                    <strong id="hex-saturated-motors">—</strong>
                </div>
                <div class="hex-metric">
                    <span>Current mass</span>
                    <strong id="hex-mass">—</strong>
                </div>
                <div class="hex-metric">
                    <span>Physical motors</span>
                    <strong>6</strong>
                </div>
            </div>
        </section>

        <section class="hex-panel">
            <h2>Reference model parameters</h2>
            <dl class="hex-params">
                <dt>Nominal mass</dt><dd>__MASS__ kg</dd>
                <dt>Inertia diag. [Jx, Jy, Jz]</dt>
                    <dd>[__JX__, __JY__, __JZ__] kg·m²</dd>
                <dt>Arm length</dt><dd>__ARM__ m</dd>
                <dt>Rotor radius</dt><dd>__ROTOR_RADIUS__ m</dd>
                <dt>Maximum rotor speed</dt><dd>__MAX_SPEED__ rad/s</dd>
                <dt>Maximum thrust / motor</dt><dd>__MAX_THRUST__ N</dd>
                <dt>Total thrust / weight</dt><dd>__TWR__</dd>
                <dt>Motor lag, up / down</dt><dd>12.5 / 25.0 ms</dd>
                <dt>Thrust constant</dt><dd>8.54858×10⁻⁶ N/(rad/s)²</dd>
                <dt>Moment constant</dt><dd>0.016 m</dd>
            </dl>
            <p>
                Source:
                <a href="__SOURCE__" target="_blank" rel="noreferrer">
                    RotorS firefly.xacro
                </a>
            </p>
        </section>

        <section class="hex-panel">
            <h2>Six-motor geometry (FLU, metres)</h2>
            <div class="hex-table-wrap">
                <table class="hex-table">
                    <thead>
                        <tr>
                            <th>#</th><th>Location</th>
                            <th>x</th><th>y</th><th>z</th><th>Spin</th>
                        </tr>
                    </thead>
                    <tbody>__ROTOR_ROWS__</tbody>
                </table>
            </div>
            <p>
                x is forward, y is left, z is up. Alternating spin directions
                create yaw authority. The allocation matrix uses these
                coordinates and enforces 0–__MAX_THRUST__ N per motor.
            </p>
        </section>
    </aside>
    <script id="raptor-hex-dashboard-script">
    (() => {
        const fixed = (value, digits) =>
            Number.isFinite(Number(value))
                ? Number(value).toFixed(digits)
                : "—";
        const phaseName = document.getElementById("hex-phase-name");
        const phaseTime = document.getElementById("hex-phase-time");
        const marker = document.getElementById("hex-timeline-marker");
        const eventLabel = document.getElementById("hex-event");
        const statusDot = document.getElementById("hex-status-dot");
        const statusText = document.getElementById("hex-status-text");
        let eventTimer = null;

        function update(demo) {
            if (!demo) return;
            phaseName.textContent =
                `Episode ${demo.episode} · ${demo.phase_name}`;
            phaseTime.textContent =
                `${fixed(demo.time_s, 2)} / ${fixed(demo.duration_s, 2)} s`;
            marker.style.left =
                `${Math.max(0, Math.min(100, demo.progress * 100))}%`;
            document.getElementById("hex-position-error").textContent =
                `${fixed(demo.position_error_m, 3)} m`;
            document.getElementById("hex-orientation-error").textContent =
                `${fixed(demo.orientation_error_deg, 1)}°`;
            document.getElementById("hex-allocation-residual").textContent =
                `${fixed(100 * demo.allocation_residual, 2)}%`;
            document.getElementById("hex-saturated-motors").textContent =
                `${demo.saturated_motors || 0} / 6`;
            document.getElementById("hex-mass").textContent =
                `${fixed(demo.mass_kg, 3)} kg`;
            if (demo.event) {
                eventLabel.textContent = demo.event;
                clearTimeout(eventTimer);
                eventTimer = setTimeout(() => {
                    eventLabel.textContent = "";
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
                statusDot.classList.add("connected");
                statusText.textContent = "Live six-rotor telemetry connected";
            };
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                if (message.channel === "setState") {
                    update(message.data && message.data.demo);
                }
            };
            ws.onclose = () => {
                statusDot.classList.remove("connected");
                statusText.textContent = "Reconnecting…";
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
            <a href="{peer_href}">← Eight quadcopters</a>
            <span>Firefly hexacopter</span>
        </nav>
        """
    dashboard_body = dashboard_body.replace(
        "__VISUALIZER_NAV__",
        visualizer_nav,
        1,
    )
    replacements = {
        "__MASS__": f"{model.mass_kg:.3f}",
        "__JX__": f"{model.inertia_kg_m2[0, 0]:.7f}",
        "__JY__": f"{model.inertia_kg_m2[1, 1]:.7f}",
        "__JZ__": f"{model.inertia_kg_m2[2, 2]:.4f}",
        "__ARM__": f"{model.arm_length_m:.3f}",
        "__ROTOR_RADIUS__": f"{model.rotor_radius_m:.3f}",
        "__MAX_SPEED__": f"{model.max_rotor_speed_rad_s:.0f}",
        "__MAX_THRUST__": f"{model.max_rotor_thrust_n:.3f}",
        "__TWR__": f"{model.thrust_to_weight:.3f}",
        "__SOURCE__": FIREFLY_SOURCE,
        "__ROTOR_ROWS__": rotor_rows,
    }
    for marker_text, replacement in replacements.items():
        dashboard_body = dashboard_body.replace(marker_text, replacement)
    if "</head>" not in original or "</body>" not in original:
        raise RuntimeError("Unexpected ui-server index document")
    return original.replace("</head>", dashboard_css + "</head>", 1).replace(
        "</body>",
        dashboard_body + "</body>",
        1,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the unchanged RAPTOR policy on a RotorS Firefly "
            "hexacopter model"
        )
    )
    parser.add_argument("--port", type=int, default=13337)
    parser.add_argument(
        "--quad-href",
        default=None,
        help="show a navigation link to this quadcopter page URL",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hexacopter"),
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.check:
        metrics, trajectory = await run_episode(realtime=False)
        write_results(args.output_dir, metrics, trajectory)
        print(json.dumps(metrics, indent=2))
        return 0 if metrics["passed"] else 1

    index_transform = (
        _hexacopter_dashboard_html
        if args.quad_href is None
        else lambda original: _hexacopter_dashboard_html(
            original,
            peer_href=args.quad_href,
        )
    )
    runner, _site, actual_port = await start_loopback_server(
        args.port,
        index_transform=index_transform,
    )
    if getattr(args, "announce", True):
        print(
            f"RAPTOR Firefly UI listening on "
            f"http://127.0.0.1:{actual_port}"
        )
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
            namespace = handshake["data"]["namespace"]
            episode = 0
            while True:
                episode += 1
                metrics, trajectory = await run_episode(
                    websocket=websocket,
                    namespace=namespace,
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
