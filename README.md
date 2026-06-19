<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=72&duration=1&pause=1000&color=000000&center=true&vCenter=true&repeat=false&width=700&height=110&lines=SMART-KNEE" alt="SMART-KNEE" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=40&duration=1&pause=1000&color=000000&center=true&vCenter=true&repeat=false&width=750&height=60&lines=Sparse-Morphology+based+Anatomical+Reconstruction+using+Topology Aware+Networks" alt="subtitle line 1" />
  <br>
  <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=40&duration=1&pause=1000&color=000000&center=true&vCenter=true&repeat=false&width=600&height=50&lines=for+Imageless+Total+Knee+Arthroplasty" alt="subtitle line 2" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/status-research-orange">
</p>

<p align="center">
A two-stage, cascaded deep learning framework that reconstructs patient-specific femur and tibia morphology from a sparse set of intraoperative landmarks and a partial surface point cloud — <b>without any preoperative CT or MRI</b>.
</p>

<p align="center">
  <i>Landmark localisation error: 2.28 ± 1.68 mm (Femur), 1.65 ± 0.83 mm (Tibia) — both within the 3 mm clinically accepted ITKA threshold.</i>
</p>

<p align="center">
  <img width="970" height="407" alt="results-finial methodology drawio (1)" src="https://github.com/user-attachments/assets/faa7757d-806c-49ed-a441-f19eadd4d3ef" />
  <br>
  <sub>Fig. 1 — Intraoperative anchor landmarks are completed by the Topology-Aware EGNN, then fused with a sparse surface point cloud by the Anatomy-Aware GAN to reconstruct complete, patient-specific bone geometry.</sub>
</p>

---

## Table of Contents
- [Overview](#overview)
- [Highlights](#highlights)
- [Method](#method)
  - [Stage 1 — Topology-Aware EGNN](#stage-1--topology-aware-egnn-landmark-completion)
  - [Stage 2 — Anatomy-Aware GAN](#stage-2--anatomy-aware-gan-surface-reconstruction)
- [Results](#results)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Imageless Total Knee Arthroplasty (ITKA) plans implant positioning in real time from sparse, intraoperatively digitised anatomical landmarks, avoiding the cost and radiation exposure of CT/MRI-based workflows. The catch is that sparse-landmark reconstruction has historically relied on statistical shape models and parametric assumptions, which cap how much patient-specific anatomical detail can be recovered — leading to inaccurate axis alignment, suboptimal implant sizing, and soft-tissue imbalance.

**SMART-KNEE** closes that gap with a cascaded pipeline:

1. A **Topology-Aware Equivariant Graph Neural Network (EGNN)** predicts the complete anatomical landmark set from just 4 intraoperative anchor landmarks per bone, enforcing E(3)-equivariance (rotation / translation / reflection invariance) and anatomical plausibility.
2. An **Anatomy-Aware GAN** fuses those predicted landmarks with a sparse intraoperative surface point cloud to reconstruct complete, watertight, patient-specific femur and tibia surfaces.

The result is a fully imageless pipeline that produces patient-specific bone morphology accurate enough for intraoperative ITKA planning.

## Highlights

- **No preoperative imaging required** — the full pipeline runs from intraoperative landmarks and a sparse probe-acquired point cloud only.
- **E(3)-equivariant landmark completion** — robust to how the patient/probe is oriented in the operating theatre.
- **Up to 6× more accurate landmark localisation** than comparable equivariant GNN baselines (VN-EGNN, EquiPPIS).
- **Clinically validated on a 3D-printed, CT-derived phantom** under optical tracking, in addition to a held-out digital test cohort.
- **361 annotated femur + tibia models** with a 12-landmark (femur) / 11-landmark (tibia) anatomical schema, derived from a publicly available lower-limb CT dataset.

## Method

<p align="center">
  <img width="785" height="284" alt="results-EGNN_FINAL drawio" src="https://github.com/user-attachments/assets/ab0e82eb-c2d5-49ac-b258-2cbde8fb20a8" />

  <br>
  <sub>Fig. 2 — Topology-Aware EGNN mapping sparse known anchors (red) to the complete predicted landmark set (green).</sub>
</p>

### Stage 1 — Topology-Aware EGNN (Landmark Completion)

Knee anatomy is modelled as a graph: landmarks are nodes, anatomical relationships are edges. Of the 12 femoral and 11 tibial landmarks, 4 per bone are known intraoperative anchors and the rest (8 femoral, 7 tibial) are unknown targets.

- The mean canonical shape template is aligned to the known anchors via a **Kabsch–Umeyama transform**, giving an anatomically plausible initialisation.
- Each node is encoded with its known/unknown status, knee chirality (left/right), and geometric descriptors.
- **4 equivariant message-passing layers** iteratively refine node features and 3D coordinates via learnt gated aggregation.
- A decoder MLP applies a clipped coordinate correction, denormalised back to millimetre space.
- Training minimises a composite loss over positional error, inter-landmark distance deviation, and anatomical fidelity.

Anchor selection matters: distally clustered anchors (e.g. nodes {3,4,5,6}) push femoral error up to 10 mm, while a geometrically diverse anchor set ({0,1,2,3} for F, {3,4,9,10} for T) achieves the reported 2.28 mm / 1.65 mm accuracy — a 4.4× and 4.7× improvement respectively.

### Stage 2 — Anatomy-Aware GAN (Surface Reconstruction)

<p align="center">
  <img width="643" height="501" alt="results-finial_GAN drawio" src="https://github.com/user-attachments/assets/2cb7e4da-0789-4779-91ac-7e5ee46265c1" />

  <br>
  <sub>Fig. 3 — Anatomy-Aware GAN for patient-specific femur and tibia surface reconstruction.</sub>
</p>

Training happens in two phases:

1. **Autoencoder pretraining** — a PointNet autoencoder learns a structured latent geometry manifold from complete knee surfaces, optimised with symmetric Chamfer Distance. Weights are frozen after this phase.
2. **Conditional adversarial training** — a generator fuses a Patch Encoder (local surface topology from the partial point cloud) with a Landmark Encoder (patient-specific proportions from the Stage 1 predictions) into a single latent code, decoded by the frozen decoder into a complete point cloud. A lightweight latent-space discriminator drives the generator's output onto the pretrained geometry manifold under an LS-GAN objective, jointly supervised by adversarial realism, patch coverage, and landmark fidelity losses.

At inference, the output point cloud is rescaled to millimetre space and converted into a watertight mesh via Poisson surface reconstruction.

## Results

### Landmark localisation (per-bone mean error, mm)

| Bone | VN-EGNN | E(3)-EGNN | **SMART-KNEE (Ours)** |
|---|---|---|---|
| Femur | 7.50 | 7.91 | **1.25 ± 0.75** |
| Tibia | 7.81 | 12.12 | **1.69 ± 0.09** |

> Full per-node breakdown (12 femoral / 11 tibial landmarks) is in Table 2 of the paper. Largest gains appear at landmarks farthest from the known anchors — exactly where topology-aware equivariant message passing matters most.

### Surface reconstruction quality (39 held-out test subjects)

| Metric | Femur | Tibia |
|---|---|---|
| Dice Score (%) | 87.0 | 92.48 |
| Hausdorff Distance, HD95 (mm) | 5.1 | 4.3 |
| Average Surface Distance, ASD (mm) | 2.4 | 1.6 |
| Normal Surface Distance, NSD@2.0mm | 0.45 | 0.70 |
| Ground-truth ↔ generated surface distance (mm) | 1.89–1.97 | 2.87 |

Mean ASD across both bones (1.69 mm) falls within the clinically accepted 3 mm ITKA accuracy criterion.

### Clinical (phantom) evaluation

End-to-end validation on a 3D-printed, CT-derived phantom under optical tracking:

| Metric | Femur | Tibia |
|---|---|---|
| Landmark RMSE (mm) | 0.25 | 0.36 |
| Surface distance vs. CT ground truth (mm) | 0.38 | 0.26 |

Both metrics are substantially within the 3 mm ITKA threshold, confirming the full pipeline meets clinical accuracy requirements without any preoperative imaging.

## Dataset

SMART-KNEE is trained and evaluated on **361 annotated femur and tibia model pairs**, derived from a publicly available lower-limb cadaver CT dataset (Fischer, *Scientific Data*, 2023), split 70% / 15% / 15% into train / validation / test with balanced left/right representation.

Each bone is annotated with a fixed anatomical landmark schema:

| Femur ID | Landmark | | Tibia ID | Landmark |
|---|---|---|---|---|
| 0 | Hip Center (FHC) | | 0 | Knee Center (TKC) |
| 1 | Knee Center (FKC) | | 1 | Medial Plateau (TMP) |
| 2 | Medial Epicondyle (FME) | | 2 | Lateral Plateau (TLP) |
| 3 | Lateral Epicondyle (FLE) | | 3 | Medial Condyle (TMC) |
| 4 | Medial Distal Condyle (FMDC) | | 4 | Lateral Condyle (TLC) |
| 5 | Lateral Distal Condyle (FLDC) | | 5 | Tuberosity (TT) |
| 6 | Medial Post. Condyle (FMPC) | | 6 | Post. Cruciate Lig. (TPCL) |
| 7 | Lateral Post. Condyle (FLPC) | | 7 | Ant. Cruciate Lig. (TACL) |
| 8 | Medial Ant. Cortex (FMAC) | | 8 | Medial Malleolus (MM) |
| 9 | Lateral Ant. Cortex (FLAC) | | 9 | Lateral Malleolus (LM) |
| 10 | Med. Cond. Prox. Post. (FMCPP) | | 10 | Ankle Center (AC) |
| 11 | Lat. Cond. Prox. Post. (FLCPP) | | | |

4 landmarks per bone are treated as known intraoperative anchors; the remainder (8 femoral, 7 tibial) are predicted by the EGNN. Sparse surface point clouds are sampled from the femoral anterior cortex / distal & posterior condyles and the tibial proximal plateaus / tuberosity, simulating real intraoperative probe acquisition.

> *Download / preprocessing instructions for the dataset go here once finalised.*

## Installation

Requires **Python 3.9+** and (recommended) a CUDA-capable GPU for training.

```bash
git clone https://github.com/IGRS-medical-imaging/BONE_SYNTHESIS_MICCAI_2026.git
cd BONE_SYNTHESIS_MICCAI_2026
conda env create -f environment.yml
conda activate smart-knee
```

## Quick Start

End-to-end run for a single bone, from mean-shape computation to CSV export:

```bash
# 1. Compute the mean shape template
python util/calculate_mean_shape.py femur

# 2. Train
python main.py --bone femur --mode train

# 3. Evaluate on the held-out test set
python main.py --bone femur --mode test

# 4. Export predictions to CSV
python main.py --bone femur --mode test --save_csv
```

Repeat with `tibia` in place of `femur` for the tibial model.

## Usage

### Calculate Mean Shape
```bash
python util/calculate_mean_shape.py femur
python util/calculate_mean_shape.py tibia
```

### Training
```bash
python main.py --bone femur --mode train
python main.py --bone tibia --mode train
```

### Evaluation
```bash
python main.py --bone femur --mode test
python main.py --bone tibia --mode test
```

### Saving Results as CSV
```bash
python main.py --bone femur --mode test --save_csv
python main.py --bone tibia --mode test --save_csv
```

## Project Structure

```text
.
├── config.py                      # Training / model / data hyperparameters
├── engine/
│   ├── evaluator.py                # Test-time inference & metric computation
│   └── trainer.py                  # Training loop, optimisation, checkpointing
├── losses/
│   └── composite_loss.py           # Positional, inter-landmark, and adversarial losses
├── models/
│   ├── decoder.py                   # PointNet / GAN decoder
│   ├── egnn_processor.py            # Equivariant message-passing layers (Stage 1)
│   ├── encoder.py                   # Patch / landmark encoders (Stage 2)
│   └── landmark_completion_model.py # Full Topology-Aware EGNN model
├── utils/
│   ├── calculate_mean_shape.py      # Mean shape template via Kabsch–Umeyama alignment
│   ├── visualization.py             # Result plotting utilities
│   └── visualization_stages.py      # Intermediate-stage visualisation
├── environment.yml
├── main.py
├── test.py
└── README.md
```

## Limitations & Future Work

- Validation to date covers a held-out digital test cohort and a single 3D-printed, CT-derived phantom under optical tracking. Cadaveric specimens and live intraoperative, patient-specific TKA settings are the natural next step and are planned as future work.
- Reconstruction quality (particularly femoral HD95) is currently bounded more by intraoperative surface acquisition coverage than by the generative architecture itself — wider articular surface sampling is expected to narrow this further.

## Citation

If you use SMART-KNEE in your research, please cite:

```bibtex
@inproceedings{rajasekar2026smartknee,
  title     = {SMART-KNEE: Sparse-Morphology based Anatomical Reconstruction using
               Topology-Aware Networks for Imageless Total Knee Arthroplasty},
  author    = {Rajasekar, Durga and Lakshmi S, Swetha and M R, Vishnu and
               Maik, Vivek and Lakshmanan, Manojkumar and Sivaprakasam, Mohanasankar},
  booktitle = {TBD},
  year      = {2026}
}
```

*(Update the `booktitle`/`year` fields once the venue and proceedings details are finalised.)*

## Acknowledgments

This work was carried out at the **Department of Electrical Engineering, IIT Madras** and the **Healthcare Technology Innovation Centre (HTIC), IIT Madras**.

## License

This project is released under the [MIT License](LICENSE).
