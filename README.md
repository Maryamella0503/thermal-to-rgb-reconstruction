# Thermal-to-RGB Scene Reconstruction Using Depth-Aware ControlNet

This repository contains the full implementation for the dissertation:

**"Thermal-to-RGB Scene Reconstruction Using Depth-Aware ControlNet"**

## Project Overview

This project investigates whether structural consistency in thermal-to-RGB image reconstruction can be improved through conditioning-level refinement, without retraining large-scale diffusion models. A Thermal-Aware Depth Normalisation Adapter (TADN) is proposed and evaluated within a Stable Diffusion 1.5 and ControlNet framework.

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


## Repository Structure

### 1. Thermal_Depth_Sandbox/
Main experimental pipeline, organised into three stages:

- **Stage_1/**
  → Thermal to depth conversion  
  → Includes:
  - `01_input_thermal/` (input thermal images)
  - `01_depth_outputs/` (generated depth maps)
  - `01_original RGB/` (reference RGB images)
  - `01_thermal_to_depth_restart.ipynb`

- **Stage_2/**
  → Baseline RGB reconstruction (no TADN)
  - `02_outputs_baseline/`
  - `02_controlnet_depth_baseline.ipynb`

- **Stage_3/**
  → TADN-enhanced RGB reconstruction
  - `03_outputs_tadn/`
  - `03_tadn_rgb_generated_outputs/`
  - `03_controlnet_depth_TADN.ipynb`

- **Previous_attempt/**
  → Earlier experimental iterations retained for completeness

---

### 2. evaluation/
→ Quantitative and qualitative evaluation

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

All experiments were conducted under fixed random seeds and identical generation parameters across baseline and TADN configurations. Results are logged in structured CSV format and can be reproduced by running the notebooks in order.

## Requirements

Install dependencies with:

pip install -r requirements.txt

Key dependencies: PyTorch, Hugging Face Diffusers, ControlNet, OpenCV, NumPy, Matplotlib, Pandas

## Models Used

- Stable Diffusion 1.5: runwayml/stable-diffusion-v1-5
- ControlNet Depth: lllyasviel/control_v11f1p_sd15_depth

All models are loaded via Hugging Face Diffusers and used in inference-only mode. No fine-tuning was performed.

## Dataset

Thermal images were sourced from the FLIR dataset, a publicly available collection of forward-facing automotive thermal imagery. Nine images were selected for controlled evaluation.

## Author
Maryam Yasser Ellathy
University of Leeds - School of Computing
