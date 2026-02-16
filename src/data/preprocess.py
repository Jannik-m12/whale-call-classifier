import librosa
import numpy as np
from datasets import load_from_disk, Audio
from tqdm import tqdm
import pickle
import os
import io
import soundfile as sf
from collections import Counter

"""
Preprocess whale audio data into mel-spectrograms for selected species with >20 samples.
Uses soundfile to decode audio, bypassing torchcodec entirely.
"""

# Configuration
SAMPLE_RATE = 16000  # Match dataset's native sample rate (all 16kHz)
DURATION = 5  # seconds (standardize all clips)
N_MELS = 128  # Number of mel bands
HOP_LENGTH = 512
MIN_SAMPLES = 21  # Minimum samples per species (>20, matching notebook)

def get_species_with_min_samples(dataset, min_samples=MIN_SAMPLES):
    """Get species that have at least min_samples recordings"""
    species_list = dataset['train']['species']
    species_counts = Counter(species_list)
    
    selected_species = sorted([species for species, count in species_counts.items() 
                       if count >= min_samples])
    
    print(f"Found {len(selected_species)} species with ≥{min_samples} samples:")
    for species in selected_species:
        print(f"  - {species}: {species_counts[species]} samples")
    
    return selected_species

def audio_to_melspectrogram(audio_array, sr):
    """Convert audio to mel-spectrogram"""
    # Resample if needed
    if sr != SAMPLE_RATE:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Pad or trim to fixed length
    target_length = DURATION * SAMPLE_RATE
    if len(audio_array) < target_length:
        audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
    else:
        audio_array = audio_array[:target_length]
    
    # Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def process_split(dataset_split, selected_species, species_to_int, split_name="train"):
    """Process a single dataset split.
    Uses Audio(decode=False) + soundfile to bypass torchcodec."""
    X = []
    y = []
    skipped = 0
    errors = 0
    
    # Disable HuggingFace audio decoding — read raw bytes instead
    dataset_split = dataset_split.cast_column("audio", Audio(decode=False))
    
    for idx in tqdm(range(len(dataset_split)), desc=f"  {split_name}"):
        try:
            sample = dataset_split[idx]
            species = sample['species']
            
            # Filter by selected species
            if species not in species_to_int:
                skipped += 1
                continue
            
            # Decode audio manually with soundfile (bypasses torchcodec)
            audio_bytes = sample['audio']['bytes']
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            audio_array = audio_array.astype(np.float32)
            
            # Handle stereo → mono
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Convert to spectrogram
            mel_spec = audio_to_melspectrogram(audio_array, sr)
            
            X.append(mel_spec)
            y.append(species_to_int[species])
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Error sample {idx}: {e}")
            continue
    
    print(f"    ✅ Processed: {len(X)}, Skipped: {skipped}, Errors: {errors}")
    return np.array(X), np.array(y)

def main():
    print("🐋 Whale Call Classifier - Preprocessing Pipeline")
    print("=" * 60)
    
    print("\n🎵 Loading dataset...")
    dataset = load_from_disk("./data/watkins_dataset")
    print(f"   Train: {len(dataset['train'])} samples")
    print(f"   Test:  {len(dataset['test'])} samples")
    
    # Get species with enough samples
    print(f"\n🔍 Filtering species with ≥{MIN_SAMPLES} samples...")
    selected_species = get_species_with_min_samples(dataset, MIN_SAMPLES)
    
    # Create species to integer mapping
    species_to_int = {species: idx for idx, species in enumerate(selected_species)}
    
    print(f"\n🎵 Converting audio to mel-spectrograms...")
    print(f"   Sample rate: {SAMPLE_RATE} Hz, Duration: {DURATION}s, Mel bins: {N_MELS}")
    
    # Process train split
    print(f"\n📦 Processing train split...")
    X_train, y_train = process_split(dataset['train'], selected_species, species_to_int, "train")
    
    # Process test split
    print(f"\n📦 Processing test split...")
    X_test, y_test = process_split(dataset['test'], selected_species, species_to_int, "test")
    
    print(f"\n✅ Results:")
    print(f"   Train — X: {X_train.shape}, y: {y_train.shape}")
    print(f"   Test  — X: {X_test.shape}, y: {y_test.shape}")
    
    if len(X_train) == 0:
        print("❌ No samples processed. Aborting.")
        return
    
    # Save processed data
    os.makedirs('./data/processed', exist_ok=True)
    
    np.save('./data/processed/X_train.npy', X_train)
    np.save('./data/processed/y_train.npy', y_train)
    np.save('./data/processed/X_test.npy', X_test)
    np.save('./data/processed/y_test.npy', y_test)
    
    # Save species mapping
    with open('./data/processed/species_mapping.pkl', 'wb') as f:
        pickle.dump(species_to_int, f)
    
    print(f"\n💾 Saved to ./data/processed/")
    print(f"   - X_train.npy: {X_train.nbytes / 1e6:.1f} MB")
    print(f"   - y_train.npy: {y_train.nbytes / 1e6:.1f} MB")
    print(f"   - X_test.npy:  {X_test.nbytes / 1e6:.1f} MB")
    print(f"   - y_test.npy:  {y_test.nbytes / 1e6:.1f} MB")
    print(f"   - species_mapping.pkl")
    
    # Print species mapping
    print(f"\n🏷️  Species mapping ({len(species_to_int)} classes):")
    for species, label in sorted(species_to_int.items(), key=lambda x: x[1]):
        print(f"   {label:2d}: {species}")

if __name__ == "__main__":
    main()