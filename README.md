# Deepfake-Detection-CNN-ViT-Integration
Detecting Deepfake Videos with Convolutional Neural Network (CNN) and Addressing Catastrophic Forgetting of CNN with Vision Transformer (ViT) Integration

# Continual Deepfake Detection (CDD) with Vision Transformer (ViT) and CNN

## Overview
This project focuses on building a Continual Deepfake Detection (CDD) system that addresses the challenge of catastrophic forgetting in deep learning models. Deepfake technology, which uses deep learning to create highly realistic fake media, poses threats to various sectors like politics, journalism, and personal privacy. Our approach combines a Vision Transformer (ViT) with a Convolutional Neural Network (CNN) to improve detection accuracy and adaptability over time.

## Project Aim
The aim of this project is to design and implement a deepfake detection system that retains its performance across evolving datasets without succumbing to catastrophic forgetting. This is achieved through a novel integration of CNN and Vision Transformer (ViT) architectures for continual learning.

## Objectives
1. Investigate the impact of catastrophic forgetting on CNN models in deepfake detection.
2. Identify CNN features contributing to catastrophic forgetting and potential improvements for continual learning.
3. Develop and implement a Continual Deepfake Detection (CDD) framework to improve robustness and adaptability.

## Methodology

### Datasets
- **FaceForensics++**: 1,000 original videos altered by Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
    - **Train/Test Split**: Part A - 10,440 train images, 2,596 test images. Part B - 10,237 train images, 2,590 test images.
- **DeepFake Detection Challenge (DFDC)**: Over 100,000 videos for deepfake detection.
    - **Train/Test Split**: Part A - 69,580 train images, 13,231 test images. Part B - 61,026 train images, 15,620 test images.

### Data Preprocessing
- **Video Frame Selection**: Middle 15 frames from each video.
- **Face Detection**: Using Haar cascade to detect and crop faces to a 299x299 resolution.
- **Data Organization**: Images labeled and stored in "Real" and "Fake" folders.

### Model Architectures
- **XceptionNet CNN**: Modified for deepfake detection using depthwise separable convolutions, batch normalization, and ReLU activation.
- **Vision Transformer (ViT)**: Processes images as patches, embeds them, and models relationships using a Transformer encoder.

### Training Process
1. **Baseline Training**: Train on Part A of Dataset 1 (FaceForensics++) and test on the same dataset to establish baseline accuracy.
2. **Cross-Dataset Testing**: Test the model on Dataset 2 (DFDC) Parts A and B to assess generalization.
3. **Incremental Training**: Retrain on Dataset 2 Part A, then test on Dataset 1 and Dataset 2 to observe knowledge transfer.
4. **Iterative Re-Training**: Continue the process on subsequent dataset parts to reinforce learning while preventing performance degradation.

## Results
- **Baseline Accuracy**: 
  - FaceForensics++ Part A (train) on Part A (test): 81.92%
  - FaceForensics++ Part A on Part B: 81.31%
- **Generalization to DFDC**:
  - The model trained on Part A of FaceForensics++ achieved competitive accuracy when tested on DFDC.

## Requirements
- **Python 3.8+**
- **Libraries**: `tensorflow`, `keras`, `numpy`, `opencv-python`, `torch`, `transformers`

Install the required libraries with:
```bash
pip install -r requirements.txt
