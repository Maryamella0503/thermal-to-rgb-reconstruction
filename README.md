# Thermal-to-RGB Scene Reconstruction Using Depth-Aware ControlNet

This repository contains the full implementation for the dissertation:

**"Thermal-to-RGB Scene Reconstruction Using Depth-Aware ControlNet"**

## Project Overview

This project investigates whether structural consistency in thermal-to-RGB image reconstruction can be improved through conditioning-level refinement, without retraining large-scale diffusion models. A Thermal-Aware Depth Normalisation Adapter (TADN) is proposed and evaluated within a Stable Diffusion 1.5 and ControlNet framework. The focus of this work is on improving structural consistency rather than photorealistic accuracy in cross-modal reconstruction.

## Setup Instructions

To run this project locally:

### 1. Clone the repository
```bash
git clone https://github.com/Maryamella0503/thermal-to-rgb-reconstruction.git
cd thermal-to-rgb-reconstruction
```
### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run notebooks
Open the notebooks using Jupyter:
```bash
jupyter notebook
```

### Run in order:
1. Stage_1 → thermal to depth
2. Stage_2 → baseline pipeline
3. Stage_3 → TADN pipeline
4. evaluation → metrics and qualitative analysis

Note: Running the full pipeline requires downloading pretrained models and may take significant time depending on hardware. GPU acceleration is recommended.

## Repository Structure

### 1. Thermal_Depth_Sandbox/
Main experimental pipeline, organised into three stages:

- **Stage_1/**
  → Thermal to depth conversion  
  → Includes:
  - `01_input_thermal/` (input thermal images)
  - `01_depth_outputs/` (generated depth maps)
  - `01_original RGB/` (reference RGB images)
  - `01_thermal_to_depth_restart.ipynb` (Notebook implementing the thermal-to-depth conversion stage, generating depth maps from input thermal images using the ControlNet depth preprocessor)

- **Stage_2/**
  → Baseline RGB reconstruction (no TADN)
  - `02_outputs_baseline/` (RGB outputs generated using the baseline pipeline without any TADN refinement)
  - `02_controlnet_depth_baseline.ipynb` (Notebook implementing the baseline depth-conditioned reconstruction pipeline)

- **Stage_3/**
  → TADN-enhanced RGB reconstruction
  - `03_outputs_tadn/` (TADN-refined depth maps produced from the initial thermal-derived depth outputs)
  - `03_tadn_rgb_generated_outputs/` (Final RGB outputs generated using the pipeline with TADN-refined depth conditioning)
  - `03_controlnet_depth_TADN.ipynb` (Notebook implementing the TADN-enhanced reconstruction pipeline)

- **Previous_attempt/**
  → Earlier experimental iterations retained for completeness

---

### 2. evaluation/
→ Quantitative and qualitative evaluation
Evaluation outputs include edge-based precision, recall, and F1 scores used in the dissertation.
- `evaluation_outputs/`
- `edge_debug_evaluation/`
- `qualitative.ipynb`
- `final_table.csv`

---

### 3. figures/
→ Figures used in the dissertation

---

### 4. requirements.txt
→ Python dependencies required to run the project

---

### 5. README.md
→ Project overview and instructions

---

## Notes
- This project uses pretrained Stable Diffusion and ControlNet models.
- GPU acceleration is recommended for faster inference.
- Some folders (e.g., Previous_attempt) are retained to show development progression.

## Method Summary

The pipeline takes a thermal image as input and generates a structurally plausible RGB reconstruction through four stages:

1. Thermal image input
2. Depth map extraction via ControlNet preprocessor
3. Optional TADN refinement — min-max normalisation, CLAHE contrast enhancement, Gaussian smoothing
4. RGB generation via Stable Diffusion 1.5 with ControlNet depth conditioning

## Reproducibility

All experiments were conducted under fixed random seeds and identical generation parameters across baseline and TADN configurations. Results are logged in structured CSV format and can be reproduced by running the notebooks in order. No model weights were modified during experimentation.

## Models Used

- Stable Diffusion 1.5: runwayml/stable-diffusion-v1-5
- ControlNet Depth: lllyasviel/control_v11f1p_sd15_depth

All models are loaded via Hugging Face Diffusers and used in inference-only mode. No fine-tuning was performed.

## Dataset

Thermal images were sourced from the FLIR dataset, a publicly available collection of forward-facing automotive thermal imagery. Nine images were selected for controlled evaluation.

## Disclaimer

This repository is provided for academic assessment purposes. Generated RGB outputs are model-based reconstructions and should not be interpreted as ground-truth representations.

## Author
Maryam Yasser Ellathy,
University of Leeds, School of Computing
