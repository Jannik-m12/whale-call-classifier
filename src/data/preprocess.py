"""
Preprocessing pipeline for Whale Call Classifier.

Loads audio from HuggingFace dataset, filters to species with >20 samples,
converts to fixed-length mel spectrograms, performs stratified train/val/test
split, computes class weights, and saves processed arrays to data/processed/.
"""

import os
import numpy as np
import librosa
import yaml
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_selected_species(dataset, min_samples=21):
    """Return list of species with >= min_samples in the training set."""
    species_list = dataset["train"]["species"]
    from collections import Counter
    counts = Counter(species_list)
    selected = sorted([s for s, c in counts.items() if c >= min_samples])
    print(f"Selected {len(selected)} species with >= {min_samples} samples each.")
    print(f"Excluded {len(counts) - len(selected)} species.")
    return selected


def pad_or_crop(audio, target_length):
    """Pad (with zeros) or crop audio array to exact target_length samples."""
    if len(audio) >= target_length:
        return audio[:target_length]
    else:
        return np.pad(audio, (0, target_length - len(audio)), mode="constant")


def audio_to_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512):
    """Convert audio array to a log-scaled mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def preprocess_dataset(config_path="configs/config.yaml"):
    """
    Full preprocessing pipeline:
    1. Load HuggingFace dataset
    2. Filter to species with >20 samples
    3. Pad/crop audio to fixed duration
    4. Convert to mel spectrograms
    5. Stratified train/val/test split
    6. Compute class weights
    7. Save to data/processed/
    """
    config = load_config(config_path)

    # Extract config values
    dataset_path = config["data"]["dataset_path"]
    output_path = config["data"]["processed_data_path"]
    min_samples = config["data"]["min_samples_per_species"]
    target_sr = config["data"]["target_sample_rate"]
    target_duration = config["data"]["target_duration_sec"]
    n_mels = config["features"]["mel_spectrogram"]["n_mels"]
    n_fft = config["features"]["mel_spectrogram"]["n_fft"]
    hop_length = config["features"]["mel_spectrogram"]["hop_length"]
    val_split = config["training"]["validation_split"]
    test_split = config["training"]["test_split"]

    target_length = int(target_sr * target_duration)  # samples

    # --- Step 1: Load dataset ---
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"  Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

    # --- Step 2: Get selected species ---
    selected_species = get_selected_species(dataset, min_samples)
    label_map = {species: idx for idx, species in enumerate(selected_species)}
    num_classes = len(label_map)
    print(f"  {num_classes} classes, label map: {label_map}")

    # --- Step 3 & 4: Process all splits ---
    # Combine train + test from HuggingFace, then do our own stratified split
    all_spectrograms = []
    all_labels = []
    skipped = 0

    splits = ["train", "test"]
    total = sum(len(dataset[s]) for s in splits)

    print(f"\nProcessing {total} samples...")
    for split in splits:
        for i in tqdm(range(len(dataset[split])), desc=f"  {split}"):
            sample = dataset[split][i]
            species = sample["species"]

            # Filter by species
            if species not in label_map:
                skipped += 1
                continue

            audio = sample["audio"]["array"].astype(np.float32)
            sr = sample["audio"]["sampling_rate"]

            # Resample if needed
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Pad or crop to fixed length
            audio = pad_or_crop(audio, target_length)

            # Convert to mel spectrogram
            mel_spec = audio_to_mel_spectrogram(
                audio, target_sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
            )

            all_spectrograms.append(mel_spec)
            all_labels.append(label_map[species])

    print(f"\n  Processed: {len(all_spectrograms)}, Skipped (species filter): {skipped}")

    X = np.array(all_spectrograms, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # Add channel dimension for CNN: (N, n_mels, time_frames, 1)
    X = X[..., np.newaxis]
    print(f"  Spectrogram shape: {X.shape[1:]}")  # e.g. (128, 157, 1)

    # --- Step 5: Stratified train/val/test split ---
    # First split off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    # Then split train into train + val
    val_fraction = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=42, stratify=y_trainval
    )

    print(f"\n  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # --- Step 6: Compute class weights ---
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(num_classes), y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"  Class weights computed for {num_classes} classes.")

    # --- Step 7: Save to data/processed/ ---
    os.makedirs(output_path, exist_ok=True)

    np.save(os.path.join(output_path, "X_train.npy"), X_train)
    np.save(os.path.join(output_path, "X_val.npy"), X_val)
    np.save(os.path.join(output_path, "X_test.npy"), X_test)
    np.save(os.path.join(output_path, "y_train.npy"), y_train)
    np.save(os.path.join(output_path, "y_val.npy"), y_val)
    np.save(os.path.join(output_path, "y_test.npy"), y_test)
    np.save(os.path.join(output_path, "class_weights.npy"), class_weights)

    # Save label map and metadata
    import json
    metadata = {
        "label_map": label_map,
        "num_classes": num_classes,
        "spectrogram_shape": list(X.shape[1:]),
        "target_sr": target_sr,
        "target_duration_sec": target_duration,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
        "class_weights": class_weight_dict,
    }
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Saved processed data to {output_path}/")
    print(f"   Files: X_train.npy, X_val.npy, X_test.npy, y_train.npy, y_val.npy, y_test.npy")
    print(f"   Files: class_weights.npy, metadata.json")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weight_dict, metadata


if __name__ == "__main__":
    preprocess_dataset()
