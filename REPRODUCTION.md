# OptIForest Reproduction Project

COMP8240 - Applications of Data Science
Macquarie University

## 1. Original Research

Paper: OptIForest: Optimal Isolation Forest for Anomaly Detection

Publication: IJCAI 2023

Original implementation:
https://github.com/xiagll/OptIForest

Original source commit:
a8be90d3ba767ead91bfc054cdea562327c11a31

This project reproduces OptIForest experiments and
evaluates the algorithm on additional real-world
and newly generated synthetic data.

## 2. Experimental Environment

- Operating system: macOS 26.6.2
- Python: 3.10.21
- NumPy: 1.23.5
- pandas: 1.5.3
- SciPy: 1.15.3
- scikit-learn: 1.2.2

Full dependencies: requirements-reproduction.txt

## 3. Experimental Configuration

- Algorithm: OptIForest (L2OPT)
- Ensemble size: 100 trees
- Threshold: 403
- Repetitions: 15 per configuration
- Branching configurations: 0 and 2
- Metrics: ROC-AUC, Average Precision
- Additional measurements: Training and testing time

The supplied demo.py fits the detector on all feature
rows and evaluates anomaly scores on those same rows.

Branch 0 randomly selects binary or ternary
hierarchical merging. Branch 2 uses binary merging.

## 4. Datasets

### AD
Original benchmark dataset from the OptIForest repository.

- Observations: 3279
- Features: 1555
- Normal: 2820
- Anomalies: 459

### Mammography
Additional real-world anomaly detection dataset.

- Observations: 11183
- Features: 6
- Normal: 10923
- Anomalies: 260

Preparation script: scripts/prepare_mammography.py

Raw source data is not included in this repository.
Its download source and redistribution terms
must be documented before publication.

### Synthetic v1
Generated using scripts/generate_synthetic.py.

- 2800 normal observations
- 450 anomalies
- 50 numerical features
- Normal features sampled from N(0,1)
- Anomaly features sampled from N(3,1.5^2)

### Synthetic v2
Generated using scripts/generate_synthetic_v2.py.

- 2800 normal observations
- 450 anomalies
- 50 numerical features
- Base features sampled from N(0,1)
- First 10 anomaly features receive additive
  Gaussian shifts sampled from N(1.5,0.5^2)
- Remaining 40 features are unshifted
- Dataset generation seed: 42

Both synthetic datasets use binary labels:
0 = normal; 1 = anomaly.

## 5. Example Commands

Generate synthetic data:

python scripts/generate_synthetic.py
python scripts/generate_synthetic_v2.py

Run original benchmark:

python demo.py --dataset=ad --threshold=403 --branch=0
python demo.py --dataset=ad --threshold=403 --branch=2

Run new datasets:

python demo.py --dataset=mammography --threshold=403 --branch=0
python demo.py --dataset=mammography --threshold=403 --branch=2

python demo.py --dataset=synthetic_v2 --threshold=403 --branch=0
python demo.py --dataset=synthetic_v2 --threshold=403 --branch=2

## 6. Results

Experimental outputs are stored in results/.

The original implementation uses 15 repeated fits.
Detector randomness is not explicitly seeded in demo.py,
so individual repeated results may vary.

## 7. Attribution

The OptIForest implementation originates from
Xiang et al. (IJCAI 2023).

The additional preprocessing scripts, synthetic
data generators, experiments and documentation
are part of this COMP8240 reproduction project.

## 8. Reproducing the Mammography Experiment

Dataset: Mammography, OpenML dataset 310.

Public CSV source:
https://datahub.io/core/openml-datasets/data/mammography

The downloaded public CSV was compared against
the original local dataset. Its columns, values,
and row order matched exactly.

From the project root, install the dependencies
listed in requirements-reproduction.txt.

Download and prepare the dataset:

    PYTHON=venv/bin/python bash scripts/setup_mammography.sh

Run the experiments:

    venv/bin/python demo.py --dataset=mammography --threshold=403 --branch=0
    venv/bin/python demo.py --dataset=mammography --threshold=403 --branch=2

The setup script downloads the dataset if the
raw file is missing and converts its class labels
to the format required by OptIForest.

The raw Mammography dataset is not included
in the GitHub repository.
