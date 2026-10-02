# Acu_Robot

Code and dataset for  
**"Design and Research of a Teleoperated Acupuncture Robot System for Reproducing the Manual Acupuncture Manipulations of Expert Acupuncturists".**

This repository provides the experimental codes and partial datasets used in the corresponding study. The project focuses on the design, modeling, and validation of a teleoperated acupuncture robotic system for reproducing expert manual acupuncture manipulations (MAMs). The repository includes three main experimental modules:

1. **Force Feedback Effects on MAM Acquisition**
2. **LSTM-based Vision-Pose Mapping Performance**
3. **Robotic Acupuncture Manipulation Reproduction**

Each experimental module is organized in an independent folder, including the corresponding code, data structure descriptions, and instructions for reproduction. Detailed usage information can be found in the README file within each subfolder.

Due to medical ethics regulations, only a portion of the dataset is currently available on GitHub.

---

## Repository Structure

Acu_Robot/
│
├── Experiment_1_Force_feedback/
│
├── Experiment_2_VPM_model/
│
└── Experiment_3_Robotic_reproduction/

---

## 1. Experiment 1: Force Feedback Effects on MAM Acquisition

**Folder:**  
`Experiment1_Force_feedback/`

This experiment evaluates the influence of bilateral force feedback on the acquisition quality of manual acupuncture manipulations.

The provided code includes:

- Data loading and preprocessing
- Comparison between control and experimental conditions
- Visualization of manipulation trajectories

The detailed description of the dataset structure, experimental conditions, and code usage is provided in the corresponding README file inside this folder.

---

## 2. Experiment 2: LSTM-based Vision-Pose Mapping Performance

**Folder:**  
`Experiment2_VPM_model/`

This experiment introduces the training and evaluation process of the LSTM-based Vision-Pose Mapping (VPM) model.

The provided codes include:

- Multimodal dataset processing
- LSTM model training
- Comparative experiments with RNN, SVR, and MLP
- Ablation studies
- Performance visualization

The detailed description of the dataset organization, model training procedure, comparative experiments, and plotting instructions is provided in the corresponding README file inside this folder.

---

## 3. Experiment 3: Robotic Acupuncture Manipulation Reproduction

**Folder:**  
`Experiment3_Robotic_reproduction/`

This folder contains the complete implementation of robotic acupuncture manipulation reproduction experiments.

The experiments include two main parts:

## 3.1 Cross-Domain Generalization Validation

This experiment evaluates whether the hand motion features learned from force-feedback stylus manipulation can generalize to natural acupuncture needle manipulation.

The validation compares:

- Expert manipulation using a Phantom haptic stylus
- Expert manipulation using a clinical acupuncture needle

The corresponding code and data processing scripts are provided in:

3_Robotic_reproduction/
└── 1_Cross-Domain_Generalization_Validation/

---

## 3.2 Robotic Reproduction and Admittance Control

This folder implements the complete robotic reproduction pipeline, including:

- VPM model inference
- Reference trajectory generation
- Robotic arm trajectory execution
- Adaptive admittance control
- Safety monitoring

The implementation includes:

- RealMan RM65 robotic arm control interface
- Trained VPM model parameters
- YOLOv8-pose based needle safety monitoring model
- Experimental data recording and visualization scripts

The detailed workflow and file descriptions are provided in:

3_Robotic_reproduction/
└── 2_Reproduction_and_Admittance_Control/

The detailed description of the system workflow, hardware requirements, dataset structure, model files, and code execution procedures is provided in the corresponding README file inside this folder.

---

## Requirements

The code is mainly developed using Python and requires the following environments:

- Python 3.x
- PyTorch
- NumPy
- OpenCV
- MediaPipe
- Scikit-learn
- Matplotlib

Additional dependencies may be required for specific experiments. Please refer to the README file in each experimental folder for detailed installation instructions.

---

## Dataset Availability

The datasets provided in this repository are partially released for research reproducibility.

Due to medical ethics regulations, only a portion of the dataset is currently available on GitHub.

The complete dataset contains expert acupuncture manipulation recordings collected under approved experimental protocols. Access to the full dataset may require additional ethical approval and authorization.

---

## Citation

If you use this code or dataset in your research, please cite:

Design and Research of a Teleoperated Acupuncture Robot System for Reproducing the Manual Acupuncture Manipulations of Expert Acupuncturists

---

## Contact

For questions regarding the code, dataset, or experimental implementation, please contact the authors.