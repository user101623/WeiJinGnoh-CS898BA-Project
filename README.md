# Proposal of Project Concept
Robust Fruit and Nut Classification via Restoration and Feature Analysis.

The goal of this project concept is to develop a computer vision system that takes "noisy" (degraded) imagery of fruits, nuts, and seeds, restores the image quality, and performs classification while simultaneously extracting physical descriptors (texture variance) to determine surface quality (e.g., fresh vs. damaged).

## Target Dataset
[Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) (Kaggle)

## Core Methodology

### Phase 1: Synthetic Degradation and Restoration 
* **Degradation:** Programatically inject realistic sensor noise such as Gaussian, salt and pepper and motion blur into the clean dataset to create a non-ideal input environment.
* **Restoration:** Implement image restoration model using Convolutional Autoencoder to invert simulated noise and reconstruct high-fidelity inputs.

### Phase 2: Surface Analysis
* Apply Homogeneity Criteria to the restored surfaces by calculating local texture variance using Gabor filters.

### Phase 3: Classification 
* Classifies the fruit identity using the restored image features concatenated with the texture variance metrics.