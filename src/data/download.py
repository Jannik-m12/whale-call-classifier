"""
Download Watkins Marine Mammal Sound Database from HuggingFace
"""

from datasets import load_dataset
import os


def download_dataset(save_path="./data/watkins_dataset"):
    print("🐋 Downloading Watkins Marine Mammal Sound Database...")
    print("This may take 10-20 minutes depending on your internet speed.")

    # Download the dataset
    dataset = load_dataset("confit/wmms-parquet")

    # Save it locally
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset.save_to_disk(save_path)

    print(f"✅ Dataset downloaded and saved to: {save_path}")
    print(f"Total samples: {len(dataset['train'])}")

    # Show first example
    print("\n📊 First sample:")
    print(dataset['train'][0])

    return dataset


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    save_path = os.path.join(project_root, "data", "watkins_dataset")
    download_dataset(save_path)