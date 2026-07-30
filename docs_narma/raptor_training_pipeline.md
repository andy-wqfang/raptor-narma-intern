# RAPTOR Training and Adaptation Pipeline

## Overview

[RAPTOR](https://arxiv.org/abs/2509.11481) trains one recurrent quadrotor
controller in two stages:

```text
physically correlated drone distribution
                  |
                  v
        sample 1,000 quadrotors
                  |
                  v
 train 1,000 independent SAC teachers
                  |
                  v
 on-policy Meta-Imitation Learning
                  |
                  v
 one 2,084-parameter recurrent student
                  |
                  v
 zero-shot adaptation through GRU state
```

The expensive part is divided into independent teacher jobs. The resulting
teachers act as control oracles for a smaller recurrent student that is
deployable on constrained flight controllers.

## 1. Sampling physically plausible quadrotors

The generator entry point is
[`sample_dynamics_parameters.cpp`](../rl-tools/src/foundation_policy/pre_training/sample_dynamics_parameters.cpp).
It creates 1,000 fixed parameter JSON files. Each file defines one MDP and is
used to train one teacher.

The central sampling transformations are in
[`10_sample_initial_parameters.h`](../rl-tools/include/rl_tools/rl/environments/l2f/operations_generic/10_sample_initial_parameters.h).
Rather than sampling every physical value independently, the code starts from
a small set of root quantities and derives correlated vehicle parameters.

| Quantity | Training range or construction |
|---|---|
| Mass | 0.02--5 kg, sampled uniformly in cube-root scale |
| Thrust-to-weight | 1.5--5 |
| Thrust curve | Crazyflie baseline constant/linear/quadratic shape, scaled using mass and thrust-to-weight |
| Frame size | Cube-root mass scaling with a mass/size deviation |
| Rotor geometry | Symmetric planar X configuration derived from arm length |
| Torque-to-inertia | 40--1,200 |
| \(J_x,J_y\) | Derived from maximum rotor torque and torque-to-inertia |
| \(J_z\) | Derived from the planar-body relation used by the paper |
| Rotor moment constant | 0.005--0.05 |
| Motor rise time | 0.03--0.10 s |
| Motor fall time | 0.03--0.30 s |
| Disturbance force | Gaussian scale tied to surplus available thrust |

The construction is important. For example, a heavier sampled vehicle changes
the thrust-curve scale and expected frame size; rotor distance and available
thrust determine torque authority; torque authority and sampled
torque-to-inertia determine inertia. This avoids many obviously contradictory
combinations that direct independent parameter sampling would create.

The paper notes that the mass-size deviation implementation accidentally uses a
normal variate where a uniform distribution was intended. The released
implementation and this reproduction retain that behavior for fidelity.

## 2. Training 1,000 specialized teachers

For every fixed dynamics configuration \(\Xi_i\), the pretraining stage learns:

```text
pi_i*(a_t | fully observed state)
```

using Soft Actor-Critic. The teachers are:

- independent and therefore embarrassingly parallel;
- specialized to exactly one dynamics configuration;
- larger feedforward policies with 64-wide hidden layers;
- Markovian;
- given ground-truth motor state that is normally unavailable on hardware; and
- trained for 1,000,000 environment steps.

Each episode is at most 500 steps at 100 Hz. Initial states include large
position, orientation, linear-velocity, and angular-velocity errors. Training
uses a mixture of position holding and smooth trajectories sampled from a
second-order Langevin process. The reward encourages position/orientation
tracking, low linear and angular velocity, smooth motor commands, survival, and
recovery rather than termination.

The implementation is configured in
[`pre_training/config.h`](../rl-tools/src/foundation_policy/pre_training/config.h).
The paper reports approximately 31 minutes for one teacher on one Ryzen 9
7945HX core and approximately 34 hours for all 1,000 teachers on the authors'
16-core laptop.

## 3. Student information restriction

The teacher can observe the true motor state, but the recurrent student sees
only:

```text
o_t = [
    position error,                 3
    orientation rotation matrix,   9
    linear-velocity error,          3
    body angular velocity,          3
    previous motor command          4
]                                      = 22 values
```

The student is never given:

- a teacher/configuration identifier;
- mass or inertia;
- arm length or rotor coordinates;
- thrust-curve or reaction-torque coefficients;
- motor time constants;
- current motor RPM; or
- external disturbance force.

Its architecture, defined in
[`post_training/config.h`](../rl-tools/src/foundation_policy/post_training/config.h),
is:

```text
Dense(22 -> 16, ReLU)
GRU(16 -> 16)
Dense(16 -> 4, identity)
```

Including the trainable initial GRU state, it contains 2,084 parameters.

## 4. Meta-Imitation Learning

Let:

- \(P[i]=\Xi_i\) be drone \(i\)'s fixed environment;
- \(\Pi^*[i]=\theta_i^*\) be its trained teacher weights;
- \(\theta\) be the shared student weights;
- \(\tau\) be a trajectory; and
- \(\tau_a^*\) be teacher action labels for that trajectory.

The released configuration uses 1,000 epochs, 1,000 teachers, 10 episodes per
teacher, 500-step sequences, batch size 64, and 10 warm-up epochs.

### Epochs 1--10: teacher rollout warm-up

For each drone, its own teacher generates the trajectory:

```text
tau ~ sample_trajectory(Xi_i, pi_i*)
```

The trajectory is projected into the student's partial observations, while the
teacher is queried with its privileged observations to produce four target
motor commands at each state.

This gives the randomly initialized student safe demonstrations before it is
allowed to control the simulated vehicles.

### Epochs 11--1,000: student on-policy rollouts

The current student now generates the trajectory:

```text
tau ~ sample_trajectory(Xi_i, pi_theta)
```

Afterward, the corresponding specialized teacher labels every state that the
student visited:

```text
y_t = pi_i*(privileged_observation(state_t))
```

The student input for the same timestep is:

```text
x_t = partial_observation(state_t, previous_student_action)
```

Therefore, "on-policy" describes the state distribution: the student visits the
states. The teacher remains the target oracle.

This distinction addresses compounding behavior-cloning errors:

```text
student error
    -> student reaches an unusual state
    -> teacher labels the recovery action at that state
    -> student learns the recovery
```

Unlike standard DAgger dataset aggregation, the trajectory list is cleared at
the beginning of every epoch. After warm-up, training uses only fresh
student-generated trajectories.

### Sequence loss and optimization

Episodes are shuffled into 500-step sequence batches. Reset masks clear the GRU
state at episode boundaries. The student predicts:

```text
Y_pred = forward(theta, X)
```

and minimizes:

```text
L(theta) = mean(||Y_pred - Y_teacher||^2)
```

with Adam at learning rate `1e-4`. Under the paper's unit-Gaussian action
distribution assumption, this MSE is the practical form of minimizing the KL
divergence between the teacher's approximate optimal-action distribution and
the student's history-conditioned distribution.

The data collection and on-policy switch are implemented in
[`post_training/main.cpp`](../rl-tools/src/foundation_policy/post_training/main.cpp)
and teacher relabeling is implemented in
[`post_training/helper.h`](../rl-tools/src/foundation_policy/post_training/helper.h).

## 5. Why this produces adaptation

The same instantaneous pose can require different commands on two drones.
A memoryless policy that is not given dynamics parameters must compromise.
RAPTOR instead observes the response to its previous actions:

```text
motor command history
        +
observed pose/velocity response
        |
        v
GRU hidden state encodes control-relevant dynamics
        |
        v
configuration-appropriate next motor commands
```

For example, a command followed by weak vertical acceleration provides evidence
of a lower thrust-to-weight ratio; delayed acceleration provides evidence of
motor lag. The student is not trained to regress named parameters. It learns
whatever latent representation best reproduces the correct teachers.

At deployment there is no gradient update, parameter file, or explicit system
identification routine. Adaptation occurs only through recurrent-state updates
at inference time. This is the paper's meaning of zero-shot in-context
adaptation.

## 6. Reproducing the training workflow

The original build and data commands remain in the repository
[`README.md`](../README.md#training). At a high level:

```bash
# Generate 1,000 fixed dynamics configurations.
foundation_policy_pre_training_sample_dynamics_parameters

# Train one teacher for every configuration, normally in parallel.
seq 0 999 | xargs -P "$(nproc)" -I {} \
  foundation_policy_pre_training \
  ./src/foundation_policy/dynamics_parameters/{}.json

# Load the teacher checkpoints and run Meta-Imitation Learning.
foundation_policy_post_training
```

The teacher stage can be skipped when using the authors' released teacher
checkpoints. The paper reports approximately 1.9 hours for Meta-Imitation
Learning on the same reference laptop used for its timing measurements.

## 7. Relationship to the visual demonstrations

It is important to separate three scopes:

| Scope | What it demonstrates |
|---|---|
| Original RAPTOR training | Produces the four-output recurrent foundation policy from 1,000 quadrotor teachers |
| Eight-quadcopter page | Runs the released checkpoint on eight deterministic configurations derived from the training ranges |
| Firefly page | Adds a model-based adapter that converts four virtual quad commands to a six-motor wrench allocation |

The Firefly adapter is not learned in the original pipeline. Native
hexacopter-policy training would require a six-output action space, a
six-rotor dynamics distribution, specialized hexacopter teachers, and a new
student distillation run.

