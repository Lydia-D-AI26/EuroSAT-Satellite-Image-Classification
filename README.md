#  Satellite Image Classification with EuroSAT

**Authors:** Lydia, Soledad, Andre  
**Date:** April 2026  

## Overview
This project implements a full image classification pipeline on the 
**EuroSAT** dataset — 27,000 Sentinel-2 satellite images across 
10 land-use/land-cover classes (Forest, Highway, Industrial, 
Residential, etc.).

## Project Structure
The notebook is organized in 4 sections:

| Section | Description |
|---------|-------------|
| **0 - EDA** | Data loading, class distribution, outlier detection |
| **1 - Baseline CNN** | Custom CNN trained from scratch (~89% accuracy) |
| **2 - Transfer Learning** | ResNet-50 (97%) and ConvNeXT (~95%) |
| **3 - Model Comparison** | Performance, efficiency & final model selection |
| **4 - Web App** | Interactive Flask app for inference |

##  Results Summary

| Model | Test Accuracy |
|-------|--------------|
| Baseline CNN (scratch) | ~89% |
| ResNet-50 | ~97% |
| ConvNeXT  | ~95% |

**ConvNeXT** was selected as the final model for its best 
balance of accuracy and computational efficiency.

## Tech Stack
- Python, PyTorch, torchvision
- HuggingFace Transformers
- scikit-learn, pandas, matplotlib, seaborn
- Flask (web app)

##  How to Run
1. Open the notebook in Google Colab
2. Enable GPU: `Runtime > Change runtime type > GPU`
3. Run all cells in order

## Dataset
[EuroSAT](https://github.com/phelber/EuroSAT) — loaded automatically 
via `torchvision.datasets.EuroSAT`
