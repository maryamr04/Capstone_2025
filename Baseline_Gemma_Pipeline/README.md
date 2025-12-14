# Sensor Recommendation with Attention Extraction (Gemma-3)

This notebook implements a batch inference pipeline using the
`google/gemma-3-4b-it` language model to recommend landmine-detection sensors
and extract attention matrices for downstream analysis.

The workflow is designed for **Google Colab with GPU support** and saves all
outputs directly to **Google Drive**.

## Overview

The pipeline performs the following steps:

1. Load structured input data (`mines.csv`) from Google Drive  
2. Select numerical precision based on available GPU  
3. Run batched text generation using Gemma-3  
4. Save sensor recommendations to CSV  
5. Re-run inference at the model level to extract attention matrices  
6. Save and archive attention weights for analysis  

## Environment

Tested in Google Colab with:

- PyTorch (CUDA-enabled)
- Transformers
- Datasets
- Accelerate
- Pandas

Recommended hardware:
- **T4 GPU** (Colab Free)  
- **A100 GPU** (Colab Pro, preferred for bf16)
