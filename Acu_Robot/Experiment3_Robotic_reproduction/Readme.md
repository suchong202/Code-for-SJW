# Experiment 3: Robotic Acupuncture Manipulation Reproduction

This folder contains the code and partially available datasets for the robotic acupuncture manipulation reproduction experiments. The experiments include two main parts:

1. **Cross-Domain Generalization Validation** — evaluating the consistency of hand motion features between force-feedback stylus manipulation and natural acupuncture needle manipulation.
2. **Robotic Reproduction and Admittance Control** — performing VPM-based trajectory generation, robotic trajectory execution, adaptive admittance control, and safety monitoring.

## Folder Structure

```text
Experiment_3/
├── 1_Cross-Domain Generalization Validation/
│   ├── cross_domain_validation.py
│   ├── data_hand_5/
│   ├── data_hand_6/
│   ├── Fig_CrossDomain_Waveform.png
│   └── Fig_CrossDomain_Distribution.png
│
└── 2_Reproduction and Admittance Control/
    ├── Robotic_Arm/
    ├── model/
    ├── data/
    ├── 1_train_model.py
    ├── 2_run_robot_inference.py
    ├── 3_run_robot_reproduction.py
    ├── 4_run_adaptive_admittance_control.py
    ├── 5_run_security_admittance_control.py
    ├── plot_tracking_result.py
    ├── plot_admittance_result.py
    └── temp_inference.csv
```

------

# Part 1: Cross-Domain Generalization Validation

This experiment evaluates whether the hand motion features extracted from different manipulation tools maintain consistency between:

- Force-feedback stylus manipulation (source domain)
- Natural acupuncture needle manipulation (target domain)

## Files

- `cross_domain_validation.py`
  Python script for performing cross-domain feature analysis and generating comparison results.

## Data Organization

```text
1_Cross-Domain Generalization Validation/

├── data_hand_5/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
│
└── data_hand_6/
    ├── 1/
    ├── 2/
    ├── 3/
    └── 4/
```

where:

- `data_hand_5` — hand keypoint data and original videos collected while holding the force-feedback stylus.
- `data_hand_6` — hand keypoint data and original videos collected while holding a clinical acupuncture needle.
- Folders `1–4` represent the four acupuncture manipulation types.
- Each manipulation folder contains multiple independent trials.

Due to medical ethics regulations, only a portion of the dataset is currently available on GitHub. The complete dataset contains 10 trials for each manipulation type, while only a subset of the trials is publicly provided.

## Output Results

Running:

```bash
python cross_domain_validation.py
```

generates the following figures:

```text
Fig_CrossDomain_Waveform.png
Fig_CrossDomain_Distribution.png
```

These figures correspond to the temporal waveform comparison and feature distribution analysis between the two manipulation domains.

------

# Part 2: Robotic Reproduction and Admittance Control

This experiment implements the complete robotic acupuncture manipulation reproduction pipeline, including:

- VPM model training
- Reference trajectory generation
- Robotic arm trajectory execution
- Adaptive force-position hybrid admittance control
- Safety monitoring based on needle pose estimation

## Folder Description

### Robotic Arm Library

```text
Robotic_Arm/
```

This folder contains the function library required for controlling the RealMan RM65 robotic arm.

------

## Model Parameters

```text
model/
```

This folder contains the trained model parameters required for inference and execution.

Contents:

```text
LSTM_Lifting_best.pth
LSTM_Twirling_best.pth

scaler_X_Lifting.pkl
scaler_X_Twirling.pkl

scaler_Y_Lifting.pkl
scaler_Y_Twirling.pkl

best.pt
```

where:

- `LSTM_Lifting_best.pth` and `LSTM_Twirling_best.pth` are trained VPM models for lifting-thrusting and twisting-rotating manipulations.
- `scaler_X_*` and `scaler_Y_*` are normalization parameters used for feature transformation.
- `best.pt` is the YOLOv8n-pose model used for needle pose estimation and safety monitoring.

------

# Data Organization

The `data` folder contains the input data and execution results generated during robotic reproduction.

```text
data/

├── data_raw/
├── 1_data_hand/
├── 2_data_execution_trajectory/
├── 3_data_tracking_result/
└── 4_data_admittance_control_result/
```

## 1. Raw Training Data

```text
data_raw/
```

Contains the original multimodal dataset used for VPM model training.

The structure is identical to Experiment 2:

```text
data_raw/

├── data_hand/
└── data_pos/
```

- `data_hand` contains MediaPipe-extracted hand keypoint data and original videos.
- `data_pos` contains robotic end-effector pose and force/torque data.

------

## 2. Original Hand Motion Data

```text
1_data_hand/
```

Contains natural acupuncture needle manipulation videos and corresponding MediaPipe hand keypoints.

Structure:

```text
1_data_hand/

├── 1/
├── 2/
├── 3/
└── 4/
```

Folders `1–4` represent the four acupuncture manipulation types.

Due to medical ethics regulations, only a portion of the dataset is currently available on GitHub.

------

## 3. VPM Generated Reference Trajectory

```text
2_data_execution_trajectory/
```

Contains the target robotic trajectories generated by the VPM model.

These trajectories are generated from the expert hand motion data and serve as the reference trajectory for robot execution.

------

## 4. Direct Robot Execution Results

```text
3_data_tracking_result/
```

Contains the actual robot execution trajectories without admittance compensation.

The recorded data are used to evaluate the trajectory tracking performance of the robotic system.

------

## 5. Admittance Control Execution Results

```text
4_data_admittance_control_result/
```

Contains the robot execution data under adaptive force-position hybrid admittance control.

Recorded information includes:

- Actual end-effector trajectory
- Force/torque measurements
- Dynamic admittance parameters (`K`, `B`)

------

# Execution Pipeline

The recommended workflow is:

```text
1. Train VPM model
        ↓
2. Generate robotic reference trajectory
        ↓
3. Execute trajectory directly
        ↓
4. Execute trajectory with adaptive admittance control
        ↓
5. Enable safety monitoring
        ↓
6. Generate experimental figures
```

------

## 1. Train VPM Model

Run:

```bash
python 1_train_model.py
```

The script reads training data from:

```text
data/data_raw/
```

and generates the trained LSTM VPM model parameters in:

```text
model/
```

------

## 2. Generate Robot Reference Trajectory

Run:

```bash
python 2_run_robot_inference.py
```

The script loads:

- VPM model parameters from `model/`
- Natural hand motion data from `data/1_data_hand/`

and generates:

```text
data/2_data_execution_trajectory/
```

The temporary inference file:

```text
temp_inference.csv
```

is generated during this process.

------

## 3. Direct Robot Reproduction

Run:

```bash
python 3_run_robot_reproduction.py
```

The robot executes the generated reference trajectory.

Execution results are saved to:

```text
data/3_data_tracking_result/
```

------

## 4. Adaptive Admittance Control

Run:

```bash
python 4_run_adaptive_admittance_control.py
```

The robot executes the reference trajectory with force-position hybrid admittance control.

The following data are recorded:

- Actual robot trajectory
- Interaction force
- Dynamic admittance parameters

Results are saved to:

```text
data/4_data_admittance_control_result/
```

------

## 5. Safety-Control Execution

Run:

```bash
python 5_run_security_admittance_control.py
```

This script extends the adaptive admittance control framework by integrating the YOLOv8n-pose model (`best.pt`) for needle pose estimation and safety monitoring.

------

# Visualization

## Robot Tracking Results

Run:

```bash
python plot_tracking_result.py
```

The script generates:

- Reference trajectory vs. actual trajectory comparison
- Spatial trajectory distribution plots

The figures are saved in:

```text
data/3_data_tracking_result/
```

------

## Admittance Control Results

Run:

```bash
python plot_admittance_result.py
```

The script generates:

- Reference trajectory vs. actual trajectory comparison
- Interaction force curves
- Dynamic admittance parameter variation (`K`, `B`)

The figures are saved in:

```text
data/4_data_admittance_control_result/
```

------

The provided code and datasets support the reproducibility of the robotic acupuncture manipulation reproduction experiments, including cross-domain validation, VPM-based trajectory generation, robotic execution, adaptive admittance control, and safety monitoring.