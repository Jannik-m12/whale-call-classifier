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

def audio_to_melspectrogram(audio_array, sr, min_duration=2.0, max_duration=60.0):
    """Convert audio to mel-spectrogram with variable length (no padding to fixed duration)"""
    # Check original duration before processing
    original_duration = len(audio_array) / sr
    
    # Skip clips that are too short or too long
    if original_duration < min_duration:
        return None  # Too short, insufficient data
    if original_duration > max_duration:
        return None  # Too long, likely outlier
    
    # Resample if needed
    if sr != SAMPLE_RATE:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # NO PADDING - Keep natural length!
    # Remove the fixed-length padding/trimming entirely
    
    # Generate mel-spectrogram with natural time dimension
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db  # Shape: (128, variable_time_frames)


def process_split_variable_length(dataset_split, selected_species, species_to_int, split_name="train"):
    """Process split with variable-length spectrograms"""
    spectrograms = []
    labels = []
    skipped = 0
    errors = 0
    filtered_short = 0
    filtered_long = 0
    duration_stats = []
    
    # Disable HuggingFace audio decoding
    dataset_split = dataset_split.cast_column("audio", Audio(decode=False))
    
    for idx in tqdm(range(len(dataset_split)), desc=f"  {split_name}"):
        try:
            sample = dataset_split[idx]
            species = sample['species']
            
            # Filter by selected species
            if species not in species_to_int:
                skipped += 1
                continue
            
            # Decode audio manually with soundfile
            audio_bytes = sample['audio']['bytes']
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
            audio_array = audio_array.astype(np.float32)
            
            # Handle stereo → mono
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Track original duration
            original_duration = len(audio_array) / sr
            duration_stats.append(original_duration)
            
            # Convert to spectrogram (NO FIXED LENGTH!)
            mel_spec = audio_to_melspectrogram(audio_array, sr)
            
            if mel_spec is None:
                if original_duration < 2.0:  # Updated threshold
                    filtered_short += 1
                else:
                    filtered_long += 1
                continue
            
            spectrograms.append(mel_spec)
            labels.append(species_to_int[species])
            
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Error sample {idx}: {e}")
            continue
    
    # Print statistics
    print(f"    ✅ Processed: {len(spectrograms)}")
    print(f"    📊 Skipped (wrong species): {skipped}")
    print(f"    📉 Filtered short (<2.0s): {filtered_short}")
    print(f"    📉 Filtered long (>60s): {filtered_long}")
    print(f"    ❌ Errors: {errors}")
    if duration_stats:
        print(f"    📏 Duration range: {min(duration_stats):.2f}s - {max(duration_stats):.2f}s")
    
    # Return as list (not numpy array) since shapes vary
    return spectrograms, np.array(labels)

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
    print(f"   Sample rate: {SAMPLE_RATE} Hz, Variable duration, Mel bins: {N_MELS}")
    
    # Process train split
    print(f"\n📦 Processing train split...")
    X_train, y_train = process_split_variable_length(dataset['train'], selected_species, species_to_int, "train")

    # Process test split
    print(f"\n📦 Processing test split...")
    X_test, y_test = process_split_variable_length(dataset['test'], selected_species, species_to_int, "test")
    
    print(f"\n✅ Results:")
    print(f"   Train — X: {len(X_train)} spectrograms, y: {y_train.shape}")
    print(f"   Test  — X: {len(X_test)} spectrograms, y: {y_test.shape}")
    
    if len(X_train) == 0:
        print("❌ No samples processed. Aborting.")
        return
    
    # Save processed data (as pickle since shapes vary)
    os.makedirs('./data/processed', exist_ok=True)

    with open('./data/processed/X_train_variable.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('./data/processed/X_test_variable.pkl', 'wb') as f:
        pickle.dump(X_test, f)

    np.save('./data/processed/y_train.npy', y_train)
    np.save('./data/processed/y_test.npy', y_test)

    # Save species mapping
    with open('./data/processed/species_mapping.pkl', 'wb') as f:
        pickle.dump(species_to_int, f)

    print(f"\n💾 Saved to ./data/processed/")
    print(f"   - X_train_variable.pkl: Variable-length spectrograms")
    print(f"   - X_test_variable.pkl: Variable-length spectrograms")
    print(f"   - y_train.npy: {y_train.nbytes / 1e6:.1f} MB")
    print(f"   - y_test.npy: {y_test.nbytes / 1e6:.1f} MB")
    print(f"   - species_mapping.pkl")
    
    # Print species mapping
    print(f"\n🏷️  Species mapping ({len(species_to_int)} classes):")
    for species, label in sorted(species_to_int.items(), key=lambda x: x[1]):
        print(f"   {label:2d}: {species}")

if __name__ == "__main__":
    main()