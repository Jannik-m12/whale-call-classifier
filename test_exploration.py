"""Test script for data exploration notebook cells"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Cell 1: Load the dataset
print("=== Cell 1: Loading dataset ===")
from datasets import load_from_disk
import pandas as pd
import numpy as np

dataset = load_from_disk("./data/watkins_dataset")
print(f"Total samples: {len(dataset['train'])}")

df = dataset['train'].to_pandas()
print(f"Columns: {df.columns.tolist()}")
print(df.head(2))

# Cell 2: Check species distribution
print("\n=== Cell 2: Species distribution ===")
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script
import matplotlib.pyplot as plt
import seaborn as sns

species_counts = df['species'].value_counts()
print(f"Number of species: {len(species_counts)}")
print("\nTop 10 species:")
print(species_counts.head(10))

plt.figure(figsize=(12, 6))
species_counts.head(15).plot(kind='bar')
plt.title('Top 15 Species by Number of Audio Samples')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('./results/species_distribution.png', dpi=300)
print("Saved species_distribution.png")
plt.close()

# Cell 3: Audio visualization
print("\n=== Cell 3: Audio visualization ===")
import librosa
import librosa.display

sample = dataset['train'][0]
print(f"Sample keys: {list(sample.keys())}")
print(f"Audio keys: {list(sample['audio'].keys())}")

audio_array = np.array(sample['audio']['array'], dtype=np.float32)
sample_rate = sample['audio']['sampling_rate']

print(f"Species: {sample['species']}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio_array)/sample_rate:.2f} seconds")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
librosa.display.waveshow(audio_array, sr=sample_rate)
plt.title(f"Waveform - {sample['species']}")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(1, 2, 2)
mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
librosa.display.specshow(mel_spec_db, sr=sample_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f"Mel Spectrogram - {sample['species']}")

plt.tight_layout()
plt.savefig('./results/sample_visualization.png', dpi=300)
print("Saved sample_visualization.png")
plt.close()

print("\n✅ All cells ran successfully!")
