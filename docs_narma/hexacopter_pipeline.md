# Hexacopter Inference and Visualization Pipeline

## Purpose and scope

The released RAPTOR checkpoint is a quadrotor controller with four
motor-command outputs. [`demo/hexacopter.py`](../demo/hexacopter.py) tests
whether that unchanged checkpoint can control a six-motor RotorS AscTec
Firefly model through a virtual-quadrotor control-allocation adapter.

The adapter and this hexacopter experiment are additions made for this
repository. They are not part of the original RAPTOR paper and have not been
validated on Firefly hardware.

## Reference airframe

The simulated plant uses parameters from the ETH Zürich RotorS
[`firefly.xacro`](https://github.com/ethz-asl/rotors_simulator/blob/master/rotors_description/urdf/firefly.xacro):

| Parameter | Value |
|---|---:|
| Nominal mass | 1.500 kg |
| Inertia diagonal | `[0.0347563, 0.0458929, 0.0977]` kg m² |
| Arm length | 0.215 m |
| Rotor radius | 0.100 m |
| Maximum rotor speed | 838 rad/s |
| Motor constant | `8.54858e-6` N/(rad/s)² |
| Moment constant | 0.016 m |
| Motor rise/fall constants | 12.5/25.0 ms |
| Total thrust-to-weight ratio | approximately 2.448 |

[`firefly_model`](../demo/hexacopter.py) defines the six positions, alternating
spin directions, physical limits, inertia, and actuator constants.

## End-to-end signal path

```text
Odometry-like state + previous four virtual commands
                         |
                         v
              unchanged RAPTOR GRU
                         |
                         | a in R^4, nominally [-1, 1]
                         v
              virtual X-quadrotor
                         |
                         | desired [Fz, tau_x, tau_y, tau_z]
                         v
             bounded six-motor allocator
                         |
                         | six rotor-speed setpoints
                         v
        motor lag + six-rotor rigid-body dynamics
```

RAPTOR continues to receive its native 22-value observation and produce its
native four outputs. It never receives the six physical motor speeds or the
Firefly model parameters.

## From virtual commands to a wrench

The policy output is clipped and mapped to a normalized virtual motor-speed
command:

```text
u_i = (clip(a_i, -1, 1) + 1) / 2
```

The adapter interprets the commands through an X-quadrotor with RAPTOR's motor
ordering and converts each command into a virtual thrust:

```text
f_i = f_virtual,max * u_i^2
```

The virtual allocation matrix maps the four thrusts to:

```text
w_des = [Fz, tau_x, tau_y, tau_z]^T
```

The virtual maximum thrust is an adapter choice, currently 1.5 times the
physical Firefly per-rotor maximum. It provides a concrete interpretation of
the original four outputs but is not learned by RAPTOR or specified by the
paper.

## Bounded allocation to six motors

For rotor \(i\), the physical allocation matrix contains:

```text
collective:       1
roll torque:      y_i
pitch torque:    -x_i
yaw torque:       spin_i * moment_constant
```

The allocator seeks six thrusts \(f\) such that:

```text
A6 f approximately equals w_des
0 <= f_i <= f_physical,max
```

Force and torque residuals are normalized before comparison so that newtons do
not numerically dominate newton-metres. The implementation first tries the
unconstrained pseudoinverse solution and, when a bound is violated, evaluates
precomputed active-set cases with motors fixed at zero, fixed at maximum, or
left free. It selects the feasible candidate with the smallest normalized
wrench residual.

The final rotor-speed commands are:

```text
omega_i,cmd = sqrt(f_i / motor_constant)
```

This path is implemented by
[`BoundedAllocator`](../demo/hexacopter.py).

## Six-rotor dynamics

The plant tracks six physical rotor speeds with separate rising and falling
first-order motor dynamics. Rotor thrust and reaction torque are combined into
a body wrench, and the 6-DoF rigid-body state is advanced at 100 Hz with RK4.
The state contains:

- position and global linear velocity;
- quaternion orientation and body angular velocity;
- six physical rotor speeds; and
- the previous four virtual RAPTOR commands.

The four virtual commands—not the six allocated motor commands—are fed back to
RAPTOR because that matches the policy's training interface.

## Robustness episode and measurements

The 16-second episode:

1. starts with position offset and approximately 30 degrees of tilt;
2. applies a `[1.0, -0.7, 0.4]` m/s velocity kick at 4 seconds; and
3. increases both mass and inertia by 25% at 8 seconds without resetting the
   GRU.

The demo records:

- pose and position/attitude errors;
- all four RAPTOR outputs;
- all six commanded and realized rotor speeds;
- normalized wrench-allocation residual;
- motor saturation count; and
- recovery and final-error metrics.

Run the acceptance test with:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m demo.hexacopter --check
```

## Browser visualizer

The visualizer reuses the generic RLtools Three.js page but patches the
parameter and render messages for six rotors. In particular, it:

- builds all six arms and rotors from the Firefly coordinates;
- colors rotors by spin direction;
- animates each rotor from its physical normalized speed;
- shows the four-command-to-wrench-to-six-motor control path;
- labels the recovery, kick, and payload phases;
- displays live position/attitude error, allocation residual, saturation, and
  mass; and
- lists all reference parameters and six rotor coordinates.

The browser receives state through a websocket. Simulation and policy inference
remain on the remote server, so no display server is required there.

Run both the quadcopter and hexacopter pages with:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m demo.visualizers
```

The Firefly page is available at `/hexacopter/` on the same port and browser
origin as the quadcopter page at `/quadcopter/`.

## Interpretation and limitations

This demo establishes that the unchanged policy can be placed upstream of a
model-based virtual-quad-to-hex allocator in this lightweight simulation. It
does not establish that RAPTOR natively understands six-motor geometry.

The current experiment excludes:

- complete RotorS/Gazebo aerodynamics;
- propeller interference and flexible-frame effects;
- sensor and estimator behavior;
- PX4 scheduling and control-allocation integration;
- battery/ESC nonlinearities beyond the motor model; and
- real Firefly or AF70 flight testing.

Hardware deployment therefore requires estimator validation, frame-convention
and motor-order checks, actuator-limit testing, failsafes, and staged tethered
experiments.
