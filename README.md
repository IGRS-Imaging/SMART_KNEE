# SMART-KNEE: Sparse-Morphology based Anatomical Reconstruction using Topology-Aware Networks for Imageless Total Knee Arthroplasty

A framework for femur and tibia bone synthesis, training, testing, and CSV-based result export.

## Repository

https://github.com/IGRS-medical-imaging/BONE_SYNTHESIS_MICCAI_2026.git

---

# Installation

Clone the repository:

```bash
git clone https://github.com/IGRS-medical-imaging/BONE_SYNTHESIS_MICCAI_2026.git
cd BONE_SYNTHESIS_MICCAI_2026
```

Install dependencies:

```bash
conda env create -f environment.yml
```

---

# Dataset Preparation

Before training or testing, calculate the mean shape for the target bone.

## Calculate Mean Shape

### Femur

```bash
python util/calculate_mean_shape.py femur
```

### Tibia

```bash
python util/calculate_mean_shape.py tibia
```

---

# Training

## Femur Model

```bash
python main.py --bone femur --mode train
```

## Tibia Model

```bash
python main.py --bone tibia --mode train
```

---

# Evaluation

## Test Femur Model

```bash
python main.py --bone femur --mode test
```

## Test Tibia Model

```bash
python main.py --bone tibia --mode test
```

---

# Saving Results as CSV

To save inference results in CSV format:

## Femur CSV Export

```bash
python main.py --bone femur --mode test --save_csv
```

---

# Project Structure

```text
.
├── config.py
├── engine/
│   ├── evaluator.py
│   └── trainer.py
├── losses/
│   └── composite_loss.py
├── models/
│   ├── decoder.py
│   ├── egnn_processor.py
│   ├── encoder.py
│   └── landmark_completion_model.py
├── utils/
│   ├── calculate_mean_shape.py
│   ├── visualization.py
│   └── visualization_stages.py
├── environment.yml
├── main.py
├── test.py
└── README.md
```

---

# Workflow

1. Calculate the mean shape for the target bone.
2. Train the model.
3. Run testing/inference.
4. Optionally export predictions/results to CSV.

