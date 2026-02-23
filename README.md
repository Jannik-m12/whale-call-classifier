# 🐋 Whale Call Classifier

## Overview
The Whale Call Classifier is a deep learning project that classifies marine mammal calls from audio recordings. It uses the [Watkins Marine Mammal Sound Database](https://huggingface.co/datasets/confit/wmms-parquet) and converts raw audio into mel-spectrograms, which are then used to train a convolutional neural network (CNN) to identify up to 27 whale and marine mammal species.

---

## Project Structure
```
whale-call-classifier/
├── data/
│   ├── watkins_dataset/        # Raw HuggingFace dataset (downloaded locally)
│   └── processed/              # Preprocessed spectrograms and labels
│       ├── X_train_variable.pkl
│       ├── X_test_variable.pkl
│       ├── y_train.npy
│       ├── y_test.npy
│       └── species_mapping.pkl
├── models/                     # Saved Keras model checkpoints
│   ├── best_model.keras
│   ├── best_baseline_model.keras
│   ├── best_custom_model.keras
│   ├── best_enhanced_model.keras
│   └── best_improved_model.keras
├── notebooks/                  # Jupyter notebooks for exploration & training
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_cnn_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── download.py         # Downloads dataset from HuggingFace
│   │   └── preprocess.py       # Audio → mel-spectrogram pipeline
│   ├── features/
│   │   └── extract_features.py # MFCC, chroma, spectrogram feature extraction
│   ├── models/
│   │   ├── train.py            # Model training & saving
│   │   ├── evaluate.py         # Accuracy, precision, recall, F1 evaluation
│   │   └── predict.py          # Load model and run inference
│   ├── visualization/
│   │   └── plot_spectrograms.py # Spectrogram plotting utilities
│   └── utils/
│       └── audio_utils.py      # Audio loading, saving, MFCC, mel-spectrogram
├── tests/
│   ├── test_preprocess.py      # Unit tests for preprocessing pipeline
│   └── test_model.py           # Unit tests for model training & evaluation
├── configs/
│   └── config.yaml             # Project configuration
├── results/                    # Evaluation results and plots
├── requirements.txt
├── setup.py
└── README.md
```

---

## Pipeline

```
Raw Audio (.flac/.wav)
        │
        ▼
  Audio Decoding          (soundfile)
        │
        ▼
  Mono Conversion &       (librosa)
  Resampling → 16 kHz
        │
        ▼
  Mel-Spectrogram         (128 mel bands, variable length)
        │
        ▼
  Filter by Duration      (2s – 60s)
        │
        ▼
  CNN Classification      (TensorFlow/Keras)
        │
        ▼
  Species Prediction      (27 classes)
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd whale-call-classifier
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Download the Dataset
```bash
python src/data/download.py
```
This downloads the Watkins Marine Mammal Sound Database from HuggingFace and saves it to `./data/watkins_dataset/`.

### 2. Preprocess the Data
```bash
python src/data/preprocess.py
```
Converts raw audio to variable-length mel-spectrograms and saves them to `./data/processed/`. Only species with **≥ 21 recordings** are included (27 species total).

### 3. Train the Model
Use the Jupyter notebooks in `notebooks/` for interactive training:
```bash
jupyter notebook notebooks/03_cnn_training.ipynb
```
Or use the training module directly:
```python
from src.models.train import train_model, save_model
history = train_model(X_train, y_train, model, epochs=30)
save_model(model, "./models/best_model.keras")
```

### 4. Evaluate the Model
```python
from src.models.evaluate import evaluate_model
metrics = evaluate_model(model, X_test, y_test)
print(metrics)  # accuracy, precision, recall, f1_score
```

### 5. Run Inference
```bash
python src/models/predict.py <model_path> <features>
```

---

## Configuration

Key parameters in [`src/data/preprocess.py`](src/data/preprocess.py):

| Parameter | Value | Description |
|---|---|---|
| `SAMPLE_RATE` | 16000 Hz | Target sample rate |
| `N_MELS` | 128 | Number of mel frequency bands |
| `HOP_LENGTH` | 512 | STFT hop length |
| `MIN_SAMPLES` | 21 | Minimum recordings per species |
| `min_duration` | 2.0 s | Minimum clip duration |
| `max_duration` | 60.0 s | Maximum clip duration |

---

## Running Tests

```bash
pytest tests/
```

Tests cover:
- **Mel-spectrogram output shape and validity** (`tests/test_preprocess.py`)
- **Processed data integrity** (shape, species mapping with 27 classes)
- **Model training and evaluation** (`tests/test_model.py`)

---

## Species

The classifier covers **27 marine mammal species** from the Watkins dataset (all species with ≥ 21 recordings). The full mapping is saved in `./data/processed/species_mapping.pkl`.

---

## Dependencies

Key libraries (see [`requirements.txt`](requirements.txt) for full list):

- `tensorflow` — CNN model training & inference
- `librosa` — Audio processing & mel-spectrogram generation
- `soundfile` — Audio decoding
- `datasets` (HuggingFace) — Dataset loading
- `numpy`, `scikit-learn` — Data handling & metrics
- `matplotlib` — Visualization
- `pytest` — Testing

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.