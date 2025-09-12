import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import shutil
import sys
import rasterio
import argparse


output_dir = "data-256"

# Helper function to load TIF files and transpose to (H, W, C)
def load_tif_as_hwc(path):
    with rasterio.open(path) as src:
        # read() returns (bands, height, width)
        img = src.read()
        # transpose to (height, width, bands)
        return np.transpose(img, (1, 2, 0))

data_dir = "ROIs2017_winter"

# Get all S1 and S2 files by walking through subdirectories
s1_files = []
s2_files = []

print(f"Searching for .tif files in '{data_dir}'...")
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.tif'):
            full_path = os.path.join(root, file)
            if '_s1_' in file:
                s1_files.append(full_path)
            elif '_s2_' in file:
                s2_files.append(full_path)

# Sort to ensure matching pairs based on filename
s1_files.sort(key=lambda f: os.path.basename(f))
s2_files.sort(key=lambda f: os.path.basename(f))

if not s1_files or not s2_files:
    print(f"ERROR: No .tif files found in '{data_dir}'. Please check the directory path and file extensions.")
    sys.exit(1)

if len(s1_files) != len(s2_files):
    print(f"WARNING: Mismatch in S1 ({len(s1_files)}) and S2 ({len(s2_files)}) file counts. The script will only use pairs.")
    # This might need a more robust pairing logic if filenames are not perfectly matched
    
print(f"Found {len(s1_files)} paired files")

# Load first file to get shape
first_s1 = load_tif_as_hwc(s1_files[0])
first_s2 = load_tif_as_hwc(s2_files[0])

# Calculate total size as early as possible
# Calculate size for 100% of data (70% train + 10% val + 20% test)
total_samples = len(s1_files)
data_samples = total_samples

# Calculate size in bytes for float32 arrays
X_size_per_sample = first_s1.nbytes
y_size_per_sample = first_s2.nbytes

# Total size for train + validation data (normalized arrays will be same size as original)
total_size_bytes = data_samples * (X_size_per_sample + y_size_per_sample)
total_size_gb = total_size_bytes / (1024 * 1024 * 1024)

# Get available disk space
available_space = shutil.disk_usage(".").free / (1024 * 1024 * 1024)  # Convert to GB

print(f"Estimated total size of arrays to save: {total_size_gb:.2f} GB")
print(f"Available disk space: {available_space:.2f} GB")

if total_size_gb >= available_space:
    print("ERROR: Not enough disk space available to save the arrays")
    print("Required space:", f"{total_size_gb:.2f} GB")
    print("Available space:", f"{available_space:.2f} GB")
    sys.exit(1)

print("Loading and stacking arrays...")

# Initialize arrays
X = np.zeros((len(s1_files), *first_s1.shape), dtype=np.float32)
y = np.zeros((len(s2_files), *first_s2.shape), dtype=np.float32)

# Load all files
for i, (s1_file, s2_file) in enumerate(tqdm(zip(s1_files, s2_files), total=len(s1_files), desc="Loading files")):
    X[i] = load_tif_as_hwc(s1_file).astype(np.float32)
    y[i] = load_tif_as_hwc(s2_file).astype(np.float32)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Create train/validation/test split on the original 256x256 images
print("Splitting original images into train, validation, and test sets...")
total_samples = len(X)
train_size = int(0.7 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size

# Shuffle indices
indices = np.random.permutation(total_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Split the full-size images
X_train_full, y_train_full = X[train_indices], y[train_indices]
X_val_full, y_val_full = X[val_indices], y[val_indices]
X_test_full, y_test_full = X[test_indices], y[test_indices]

print(f"Full-size train images: {len(X_train_full)}")
print(f"Full-size validation images: {len(X_val_full)}")
print(f"Full-size test images: {len(X_test_full)}")

# --- Function to create 256x256 crops from images ---
def create_crops(X_in, y_in, crop_size=256):
    num_original_samples, img_height, img_width, _ = X_in.shape
    
    if img_height % crop_size != 0 or img_width % crop_size != 0:
        print(f"ERROR: Image dimensions ({img_height}x{img_width}) are not divisible by CROP_SIZE ({crop_size})")
        sys.exit(1)
        
    num_crops_h = img_height // crop_size
    num_crops_w = img_width // crop_size
    num_crops_per_img = num_crops_h * num_crops_w
    num_cropped_samples = num_original_samples * num_crops_per_img

    X_cropped = np.zeros((num_cropped_samples, crop_size, crop_size, X_in.shape[-1]), dtype=np.float32)
    y_cropped = np.zeros((num_cropped_samples, crop_size, crop_size, y_in.shape[-1]), dtype=np.float32)

    current_idx = 0
    for i in tqdm(range(num_original_samples), desc="Cropping images"):
        for h in range(num_crops_h):
            for w in range(num_crops_w):
                h_start, h_end = h * crop_size, (h + 1) * crop_size
                w_start, w_end = w * crop_size, (w + 1) * crop_size
                
                X_cropped[current_idx] = X_in[i, h_start:h_end, w_start:w_end, :]
                y_cropped[current_idx] = y_in[i, h_start:h_end, w_start:w_end, :]
                
                current_idx += 1
                
    return X_cropped, y_cropped

# --- Create 256x256 crops for each set ---
print("Creating 256x256 crops for each data split...")
X_train, y_train = create_crops(X_train_full, y_train_full)
X_val, y_val = create_crops(X_val_full, y_val_full)
X_test, y_test = create_crops(X_test_full, y_test_full)

print("Cropping complete.")
print(f"Train samples (cropped): {len(X_train)}")
print(f"Validation samples (cropped): {len(X_val)}")
print(f"Test samples (cropped): {len(X_test)}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Normalize SAR data (X) - Min-Max scaling
print("Normalizing SAR data (X)...")
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])  # (N*H*W, C)
X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])  # (N*H*W, C)
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1]) # (N*H*W, C)

# Fit scaler on training data only (no data leakage)
X_scaler = MinMaxScaler()
X_train_normalized = X_scaler.fit_transform(X_train_reshaped)
X_val_normalized = X_scaler.transform(X_val_reshaped)
X_test_normalized = X_scaler.transform(X_test_reshaped)

# Reshape back
X_train_normalized = X_train_normalized.reshape(X_train.shape)
X_val_normalized = X_val_normalized.reshape(X_val.shape)
X_test_normalized = X_test_normalized.reshape(X_test.shape)

# Normalize Optical data (y) - Standard scaling per channel
print("Normalizing Optical data (y)...")
y_scalers = []
y_train_normalized = np.zeros_like(y_train)
y_val_normalized = np.zeros_like(y_val)
y_test_normalized = np.zeros_like(y_test)

for channel in range(y_train.shape[-1]):
    # Fit scaler on training data only
    scaler = StandardScaler()
    y_train_channel = y_train[:, :, :, channel].reshape(-1, 1)
    y_val_channel = y_val[:, :, :, channel].reshape(-1, 1)
    y_test_channel = y_test[:, :, :, channel].reshape(-1, 1)
    
    y_train_normalized[:, :, :, channel] = scaler.fit_transform(y_train_channel).reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2])
    y_val_normalized[:, :, :, channel] = scaler.transform(y_val_channel).reshape(y_val.shape[0], y_val.shape[1], y_val.shape[2])
    y_test_normalized[:, :, :, channel] = scaler.transform(y_test_channel).reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
    
    y_scalers.append(scaler)

# Save normalized data
print(f"Saving normalized data to '{output_dir}/'...")
np.save(f"{output_dir}/X_train.npy", X_train_normalized)
np.save(f"{output_dir}/y_train.npy", y_train_normalized)
np.save(f"{output_dir}/X_val.npy", X_val_normalized)
np.save(f"{output_dir}/y_val.npy", y_val_normalized)
np.save(f"{output_dir}/X_test.npy", X_test_normalized)
np.save(f"{output_dir}/y_test.npy", y_test_normalized)

# Save scalers
print("Saving scalers...")
with open(f"{output_dir}/X_scaler.pkl", "wb") as f:
    pickle.dump(X_scaler, f)

with open(f"{output_dir}/y_scalers.pkl", "wb") as f:
    pickle.dump(y_scalers, f)

print(f"Done! All data and scalers saved in '{output_dir}/' directory")
print(f"X_train shape: {X_train_normalized.shape}")
print(f"y_train shape: {y_train_normalized.shape}")
print(f"X_val shape: {X_val_normalized.shape}")
print(f"y_val shape: {y_val_normalized.shape}")
print(f"X_test shape: {X_test_normalized.shape}")
print(f"y_test shape: {y_test_normalized.shape}")
    

