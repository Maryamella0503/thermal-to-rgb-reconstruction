# Thermal-to-RGB Scene Reconstruction Using Depth-Aware ControlNet

This repository contains the full implementation for the dissertation:

**"Thermal-to-RGB Scene Reconstruction Using Depth-Aware ControlNet"**

## Project Overview

This project investigates whether structural consistency in thermal-to-RGB image reconstruction can be improved through conditioning-level refinement, without retraining large-scale diffusion models. A Thermal-Aware Depth Normalisation Adapter (TADN) is proposed and evaluated within a Stable Diffusion 1.5 and ControlNet framework.

## Repository Structure

1. Thermal_Depth_Sandbox/  
   → Main experimental pipeline (depth generation, baseline outputs, TADN outputs)  
   → Includes:
      - 01_input_thermal/
      - 01_depth_outputs/
      - 02_outputs_baseline/
      - 03_outputs_tadn/
      - notebooks for each stage of the pipeline

2. evaluation/  
   → Edge-based evaluation scripts and outputs  
   → Includes:
      - evaluation_outputs/
      - edge_debug_evaluation/
      - qualitative.ipynb
      - table_3_1.csv

3. figures/  
   → Figures used in the dissertation

4. OLD_* folders  
   → Previous experimental attempts (not used in final evaluation, retained for completeness)

5. README.md  
   → Project overview and instructions

6. requirements.txt  
   → Python dependencies required to run the project

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
