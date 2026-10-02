# Experiment 1: Force Feedback Effects on MAM Acquisition

This folder contains the code and data organization for the force-feedback validation experiment described in the manuscript. The experiment compares acupuncture manipulation performance under two conditions: **without force feedback** and **with bilateral force feedback**.

Due to medical ethics regulations, only a portion of the dataset is currently available on GitHub.

## Files

- `data_analysis.py` — Python script for loading the experimental data, processing the trajectories, and generating three comparison figures.

## Data Organization

The two experimental conditions are organized as follows:

```text
Experiment_1/
├── data_analysis.py
├── data_pos_1/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
└── data_pos_2/
    ├── 1/
    ├── 2/
    ├── 3/
    └── 4/
```

where:

- `data_pos_1` — **Control Condition (Without Force Feedback)**
- `data_pos_2` — **Experimental Condition (With Force Feedback)**
- Folders `1–4` represent the four acupuncture manipulation types.
- Each manipulation folder contains multiple independent trials.
- Due to medical ethics regulations, only a portion of the dataset is currently available on GitHub. The complete dataset contains 10 trials for each manipulation type, while only a subset of the trials is publicly provided.

## Usage

Place the `data_pos_1` and `data_pos_2` folders in the same directory as `data_analysis.py`, then run:

```bash
python data_analysis.py
```

The script automatically reads the two groups of experimental data, performs the corresponding data processing, and generates three figures for comparing the manipulation characteristics between the control and experimental conditions.

## Experimental Conditions

| Condition              | Description            |
| ---------------------- | ---------------------- |
| Control Condition      | Without Force Feedback |
| Experimental Condition | With Force Feedback    |

The analysis is intended to evaluate the effect of bilateral force feedback on the acquisition and consistency of acupuncture manipulation data.

