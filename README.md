# 🧠 Brain-Computer Interface (BCI): Error Potential Detection

**Biomedical Signal Processing & Deep Learning Pipeline**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Signal Processing](https://img.shields.io/badge/DSP-Butterworth%20Filters-green)

## 📌 Overview
This project implements a complete Machine Learning pipeline to detect **Error Potentials (ErrP)** in EEG (Electroencephalography) data. The system analyzes brain waves to determine if a user perceived a mistake made by a BCI speller interface.

This implementation focuses on a **Deep Learning approach (MLP)** combined with rigorous **Signal Processing** (Epoching, Filtering, Artifact Removal) to handle high-dimensional, noisy biomedical time-series data.

## 🛠️ System Architecture

### 1. Preprocessing Pipeline (`src/preprocess.py`)
* **Temporal Filtering:** Applied 4th-order **Butterworth Bandpass Filter (1Hz - 40Hz)** to remove DC drift and high-frequency muscle noise (EMG).
* **Epoching:** Extracted **1.3-second time windows** time-locked to the feedback event stimulus.
* **Downsampling:** Reduced sampling rate to optimize feature density and memory usage.
* **Normalization:** Standard Scalar normalization to zero-mean and unit variance for neural network stability.

### 2. Model Architecture (`src/model.py`)
* **Type:** Multi-Layer Perceptron (MLP) Classifier.
* **Structure:**
    * Input Layer: Flattened EEG Epoch Vector (~2800 features)
    * Hidden Layer 1: 100 Neurons (ReLU activation)
    * Hidden Layer 2: 50 Neurons (ReLU activation)
    * Output Layer: Sigmoid (Probability of Error)
* **Optimization:** Trained using **Adam Optimizer** with **Early Stopping** to prevent overfitting on the small dataset.

## 📂 Project Structure
```bash
BCI_Error_Detection/
├── data/               # Dataset (Not included in repo)
├── src/
│   ├── config.py       # Hyperparameters & Paths
│   ├── preprocess.py   # Signal Processing Logic
│   ├── model.py        # Neural Network Architecture
│   └── train.py        # Training Loop
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

🚀 Getting Started
1. Prerequisites
Install the required dependencies:
pip install -r requirements.txt

2. Dataset Setup
This project uses the INRIA BCI Challenge dataset.

 1. Download the data from Kaggle.

 2. Unzip the train.zip and test.zip files.

 3. Place them in the data/ directory so the structure looks like this:

      •data/train/Subject01...csv

      •data/test/Subject16...csv

      •data/TrainLabels.csv
    
3. Training
Run the training script to process signals and train the MLP:
python -m src.train

📊 Results

 •Preprocessing: Successfully attenuated 50Hz power-line noise and ocular artifacts.

 •Classification: The MLP model achieves robust detection of P300/ErrP patterns, validated via AUC metrics on unseen subjects.
