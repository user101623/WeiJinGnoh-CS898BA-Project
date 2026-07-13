# Project Proposal & Implementation Report

## 1. Project Concept
**Title:** Robust Fruit and Nut Classification via Restoration and Feature Analysis

The goal of this project is to develop an advanced, end-to-end computer vision pipeline capable of handling degraded, "noisy" imagery of fruits and nuts. The system explicitly restores image quality, extracts critical physical descriptors (surface texture variance and entropy) to analyze surface homogeneity, and performs robust multi-class classification. This design shifts away from standard "black-box" models by introducing an engineered fusion of learned deep features and explicit mathematical descriptors.

---

## 2. Dataset Specification
* **Source Dataset:** Fruits-360 Dataset (Kaggle)
* **Target Resolution:** 100x100 pixels (for computational efficiency and standardized matrix processing)
* **Target Classes:**
  * **Fruits:** Apple, Banana, Strawberry
  * **Nuts & Seeds:** Almonds, Caju (Cashew) seed, Chestnut, Hazelnut, Pistachio, Walnut

---

## 3. Core Methodology Flow

* **Phase 1: Synthetic Degradation & Supervised Restoration**
  * **Degradation:** Programmatically injects realistic sensor noise (e.g., salt-and-pepper artifacts) into clean image arrays to simulate non-ideal capture environments.
  * **Restoration:** Deploys a deep Convolutional Autoencoder to invert the noise distribution and reconstruct high-fidelity structural inputs.

* **Phase 2: Surface Homogeneity Analysis**
  * Segments object topologies using the Watershed algorithm directly on the surface area, then applies mathematical Gabor filter kernels to isolate local texture variations.

* **Phase 3: Multi-Modal Feature Fusion & Classification**
  * Extracts structural metrics (variance and entropy) and concatenates them directly with the flattened deep spatial feature maps from a standard CNN branch to output class identities via a 6-class softmax prediction.

---

## 4. Current Architecture Implementation

### 📂 main.py
Acts as the main runtime hub coordinating data loading, model lifecycles, and evaluation steps.
* Loads and processes the Fruits-360 (100x100) dataset, scaling pixel channels to a normalized range.
* Maps clean target images to synthetic pairs modified by a standalone noise injection block.
* Manages the complete training setup to fit both the restoration and classification networks sequentially.
* Houses the full pipeline function to drive single-thread inference including noise injection, autoencoder denoising, watershed surface evaluation, Gabor extraction, and final prediction.

### 📂 utils.py
Isolates explicit algorithmic computer vision operations from the neural networks.
* **Watershed Segmentation Function**: Drops marker seeds across the object surface map to segment local structural boundaries.
* **Texture Feature Extraction Function**: Contains a critical safety check to safely unwrap incoming TensorFlow graph tensor objects into standard array formats before passing them to OpenCV processing functions. It applies a Gabor filter kernel to calculate explicit structural descriptors consisting of texture variance and Shannon entropy.

### 📂 models.py
Defines network layers using the inbuilt Keras Functional API from Tensorflow library.
* **Convolutional Autoencoder Structure**: A symmetrical structural restoration network. Compresses image grids with downsampling and spatial pools to discard sparse noise vectors, expanding them back cleanly via transposed convolutions and a final boundary activation layer.
* **Fused Classifier Structure**: Builds a multi-input classification pipeline. It processes the autoencoder's output image through a deep CNN layer sequence, flattens the result, and merges it with the two explicit Gabor engineering elements using a feature concatenation block before passing the fused array to a final classification layer.
* **Baseline CNN Structure**: Assembles a standard spatial architecture that classifies images directly without feature engineering, providing a performance control benchmark.