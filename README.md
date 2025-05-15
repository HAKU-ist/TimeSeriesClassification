# Time series Classification using KNN and LSTM

This project implements time series classification for datesets of four difficulty levels (level1 to level4), using two methods:

- **DTW + KNN (Dynamic Time Warping + k-Nearest Neighbour)**
- **LSTM (Long Short-Term Memory networks)**

---

## Project Structure

```
TimeSeriesClassification/
├── codes/ # All source code
│ ├── KNNl1.py # DTW + KNN for level1
│ ├── KNNl2.py # DTW + KNN for level2
│ ├── LSTMl3.py # LSTM for level3 (variable-length)
│ ├── LSTMl4.py # LSTM for level4 (EEG)
│ └── runs/
│ ├── lstm_level3/
│ └── lstm_level4/
├── dataset/
│ ├── level1/ # Level 1 data
│ ├── level2/ # Level 2 data
│ ├── level3/ # Level 3 data
│ └── level4/ # Level 4 data
├── model/
│ ├── lstm_level3.pth # Trained model for level3
│ └── lstm_level4.pth # Trained model for level4
└── README.md # Project description
```
---
## Dataset Description

Each `level` folder contains:

- `reference/`: labeled training samples
- `test/`: unlabeled test samples

---

## Methods

### DTW + KNN

Implemented in:

- `codes/KNNl1.py`
- `codes/KNNl2.py`

Distance-based nearest neighbor classification using DTW.

---

### LSTM Classification (Pytorch)

Implemented in:

- `codes/LSTMl3.py`
- `codes/LSTMl4.py`

Features:

- Built with **PyTorch**
- Accelerated on Apple M1/M2/M3 via **MPS backend**
- Loss tracking via **TensorBoard**
- Model checkpoints saved in `.pth` format

---

## Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib tensorboard

# Run training script (example: level4)
cd codes
python LSTMl4.py
```
Launch TensorBoard:

```bash
tensorboard --logdir=runs
```

Open your browser and visit:  
👉 http://localhost:6006

---
