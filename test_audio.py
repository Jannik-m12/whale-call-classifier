from datasets import load_from_disk
import numpy as np

ds = load_from_disk("./data/watkins_dataset")

# Test 1: Direct access from dataset (not pandas)
print("=== Test 1: Direct dataset access ===")
try:
    s = ds["train"][0]
    print(f"Species: {s['species']}")
    print(f"Audio type: {type(s['audio'])}")
    print(f"Audio keys: {list(s['audio'].keys())}")
    print(f"Array len: {len(s['audio']['array'])}")
    print(f"SR: {s['audio']['sampling_rate']}")
    arr = np.array(s["audio"]["array"], dtype=np.float32)
    print(f"Numpy array shape: {arr.shape}")
    print("SUCCESS: Direct access works!")
except Exception as e:
    print(f"FAILED: {e}")

# Test 2: Pandas access
print("\n=== Test 2: Pandas access ===")
try:
    df = ds["train"].to_pandas()
    row = df.iloc[0]
    print(f"Species: {row['species']}")
    print(f"Audio type: {type(row['audio'])}")
    arr = np.array(row["audio"]["array"], dtype=np.float32)
    print(f"Numpy array shape: {arr.shape}")
    print("SUCCESS: Pandas access works!")
except Exception as e:
    print(f"FAILED: {e}")
