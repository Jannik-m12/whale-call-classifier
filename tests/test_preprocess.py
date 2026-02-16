"""Tests for the preprocessing pipeline (src/data/preprocess.py)."""
import numpy as np
import pytest
import os
import pickle

from src.data.preprocess import (
    audio_to_melspectrogram,
    SAMPLE_RATE,
    DURATION,
)

@pytest.fixture
def sine_wave():
    """5-second 440 Hz sine wave at 16 kHz."""
    t = np.linspace(0, DURATION, DURATION * SAMPLE_RATE, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)

class TestAudioToMelspectrogram:
    def test_output_shape(self, sine_wave):
        mel = audio_to_melspectrogram(sine_wave, SAMPLE_RATE)
        assert mel.shape == (128, 157), f"Expected (128, 157), got {mel.shape}"

    def test_output_is_finite(self, sine_wave):
        mel = audio_to_melspectrogram(sine_wave, SAMPLE_RATE)
        assert np.all(np.isfinite(mel)), "Contains NaN or Inf"

class TestProcessedData:
    PROCESSED_DIR = "./data/processed"

    @pytest.fixture(autouse=True)
    def _check_data_exists(self):
        if not os.path.exists(os.path.join(self.PROCESSED_DIR, "X_train.npy")):
            pytest.skip("Processed data not found — run preprocess.py first")

    def test_x_train_shape(self):
        X = np.load(os.path.join(self.PROCESSED_DIR, "X_train.npy"))
        assert X.shape[1:] == (128, 157)

    def test_species_mapping_27_classes(self):
        with open(os.path.join(self.PROCESSED_DIR, "species_mapping.pkl"), "rb") as f:
            mapping = pickle.load(f)
        assert len(mapping) == 27
