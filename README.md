# VIRAGA.Q_p2

## Hybrid Quantum-Classical Early Disease Detection Prototype

VIRAGA.Q_p2 is the second prototype of the VIRAGA healthcare platform, developed as part of the Smart India Hackathon (SIH).

The prototype explores the use of Hybrid Quantum Machine Learning (QML) and Classical Machine Learning (CML) for early disease detection while providing a fair benchmarking framework for comparing both approaches under identical experimental conditions.

Unlike many quantum healthcare projects that evaluate quantum models in isolation, VIRAGA.Q_p2 is designed to determine whether quantum machine learning provides measurable value when compared directly against established classical methods.

---

## Current Disease Modules

### Breast Cancer Detection

Dataset:
- Wisconsin Breast Cancer Diagnostic Dataset
- 569 patient samples
- 30 diagnostic features

Models:
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- Quantum Support Vector Classifier (QSVC)
- Variational Quantum Classifier (VQC) *(under development)*

---

### Parkinson's Disease Detection

Dataset:
- Oxford Parkinson's Disease Dataset
- 197 voice recordings
- 22 biomedical voice features

Models:
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- Quantum Support Vector Classifier (QSVC)
- Variational Quantum Classifier (VQC) *(under development)*

---

## Core Features

### Hybrid Quantum-Classical Architecture

The platform trains and evaluates both classical and quantum models within the same workflow.

### Fair Benchmarking Framework

All model families are evaluated using:

- Identical datasets
- Identical train/test splits
- Identical preprocessing pipelines
- Identical selected feature subsets for quantum comparisons

This ensures that performance differences arise from the models themselves rather than differences in data preparation.

### Automated Evaluation

Generated metrics include:

- Accuracy
- Precision
- Sensitivity (Recall)
- Specificity
- F1 Score
- ROC-AUC
- PR-AUC
- Training Time

### Explainable AI

SHAP-based explainability is used to provide feature-level interpretation of model predictions.

### Modular Design

Each disease module is implemented independently, allowing future diseases to be added without redesigning the entire platform.

---

## Current Experimental Results

### Breast Cancer Module

Best Classical ROC-AUC:

- Random Forest: **0.997**

Best Quantum ROC-AUC:

- QSVC (6 qubits): **0.932**

### Parkinson's Module

Best Classical ROC-AUC:

- Random Forest: **0.941**

Best Quantum ROC-AUC:

- QSVC (4 qubits): **0.855**

These results demonstrate the platform's ability to benchmark classical and quantum approaches under controlled experimental conditions.

---

## System Workflow

```text
Data Ingestion
        ↓
Preprocessing
        ↓
Feature Selection
        ↓
 ┌──────────────┬──────────────┐
 │              │              │
 ▼              ▼
Classical ML    Quantum ML
 │              │
 ▼              ▼
Benchmarking Engine
        ↓
Explainability (SHAP)
        ↓
Results Dashboard
```

---

## Repository Structure

```text
VIRAGA.Q_p2/
│
├── modules/
│   ├── cancer/
│   └── parkinsons/
│
├── scripts/
│   ├── train_cancer.py
│   └── train_parkinsons.py
│
├── data.py
├── training.py
├── quantum.py
├── settings.py
├── app.py
│
├── artifacts/
└── data/
```

---

## Technology Stack

### Data Processing

- NumPy
- Pandas
- Scikit-Learn

### Classical Machine Learning

- Random Forest
- Support Vector Machines
- Logistic Regression
- XGBoost

### Quantum Machine Learning

- IBM Qiskit
- Qiskit Machine Learning
- QSVC
- VQC

### Explainability

- SHAP

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Python
- GitHub
- Visual Studio Code

---

## Running the Prototype

### Clone Repository

```bash
git clone https://github.com/arav1nd34/VIRAGA.Q_p2.git
cd VIRAGA.Q_p2
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Cancer Module

```bash
python scripts/train_cancer.py
```

### Train Parkinson's Module

```bash
python scripts/train_parkinsons.py
```

---

## Future Development

Planned additions include:

- Sepsis Prediction Module
- Cardiovascular Disease Module
- Alzheimer's Disease Module
- Diabetes Risk Prediction Module
- Complete VQC Integration
- Enhanced Benchmark Visualizations
- Cloud Deployment
- Clinical Decision Support Features

---

## Project Status

Current Version:
**Prototype 2 (VIRAGA.Q_p2)**

Development Status:
**Active**

This repository represents the second-generation research prototype and benchmarking framework developed for the VIRAGA project.

---

## Authors

Team VIRAGA

Smart India Hackathon (SIH)

Hybrid Quantum Machine Learning for Early Disease Detection
