import os
import torch
import numpy as np
import pickle
import logging
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from skimage.metrics import structural_similarity
import sys
from pathlib import Path


gpu_id = 0
input_channel_select = None # Can be 0, 1, or None (for all channels)
models_to_run = ["UMamba", "pix2pix", "SwinUnet", "SiMaVP"]
DATA_DIR = "data-256"
batch_size = 64

# --- Model Selection ---
# Add U-Mamba project to the Python path
umamba_repo_path = Path(__file__).parent / 'mymodels/U-Mamba'
umamba_pkg_path = umamba_repo_path / 'umamba'
if umamba_repo_path.is_dir() and umamba_pkg_path.is_dir():
    sys.path.insert(0, str(umamba_repo_path))
    sys.path.insert(0, str(umamba_pkg_path))
else:
    print(f"U-Mamba repository not found or is incomplete.")
    # sys.exit(1) # Don't exit, just print warning in case we are not evaluating it

# Add pix2pix project to the Python path
gan_repo_path = Path(__file__).parent / 'mymodels/pytorch-CycleGAN-and-pix2pix'
if gan_repo_path.is_dir():
    sys.path.insert(0, str(gan_repo_path))
else:
    print(f"pytorch-CycleGAN-and-pix2pix repository not found at {gan_repo_path}")
    # sys.exit(1)

# Add Swin-Unet project to the Python path
swin_repo_path = Path(__file__).parent / 'mymodels/Swin-Unet'
if swin_repo_path.is_dir():
    sys.path.insert(0, str(swin_repo_path))
else:
    print(f"Swin-Unet repository not found at {swin_repo_path}")
    # sys.exit(1)


# --- Model Imports ---
from mymodels.UMambaWrapper import UMambaWrapper
from mymodels.cp2pwrapper import Pix2PixWrapper
from mymodels.swinunetwrapper import SwinUnetWrapper
from mymodels.SiMaVP import SiMaVP


# --- Helper functions for metrics ---
def ssim(y_true, y_pred):
    """
    Computes the mean Structural Similarity Index (SSIM) over a batch of images.
    """
    ssim_scores = []
    # Calculate data_range on the true data distribution
    data_range = np.max(y_true) - np.min(y_true)
    
    for i in range(y_true.shape[0]):
        # For each image in the batch
        score = structural_similarity(y_true[i], y_pred[i], channel_axis=-1, data_range=data_range)
        ssim_scores.append(score)
    return np.mean(ssim_scores)

def mean_absolute_scaled_error(y_true, y_pred):
    """
    Computes the Mean Absolute Scaled Error (MASE).
    The naive forecast is the previous value. This metric is more common for time series.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    # Naive forecast (persistence model) error for reference
    mae_naive = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    if mae_naive == 0:
        return np.inf  # Avoid division by zero
    return mae / mae_naive

def peak_signal_to_noise_ratio(y_true, y_pred):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR).
    """
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return float('inf')
    data_range = np.max(y_true) - np.min(y_true)
    if data_range == 0:
        return -np.inf
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Computes the Symmetric Mean Absolute Percentage Error (sMAPE).
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Handle the case where both y_true and y_pred are zero
    # In this case, the error is 0, so we can replace NaNs with 0
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

def inverse_transform_y(y_scaled, scalers):
    """
    Applies inverse scaling to the optical data (y) channel by channel.
    """
    y_unscaled = np.zeros_like(y_scaled)
    num_channels = y_scaled.shape[-1]
    
    if len(scalers) != num_channels:
        raise ValueError(f"Number of scalers ({len(scalers)}) does not match number of channels ({num_channels})")

    for i in range(num_channels):
        scaler = scalers[i]
        # Reshape channel to (n_samples, 1) for scaler
        channel_data = y_scaled[..., i].reshape(-1, 1)
        # Inverse transform
        unscaled_channel_data = scaler.inverse_transform(channel_data)
        # Reshape back to original image shape
        y_unscaled[..., i] = unscaled_channel_data.reshape(y_scaled.shape[:-1])
        
    return y_unscaled


# --- Configuration ---

if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)  
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device("cpu")
print(f"Using device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"Current CUDA device index: {torch.cuda.current_device()}")


RESULTS_DIR = f"trained-{input_channel_select}/evaluation_results-{input_channel_select}"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Logging Setup ---
log_file = os.path.join(RESULTS_DIR, f'evaluation_metrics_channel_{input_channel_select}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'), # Append to log file
        logging.StreamHandler()
    ]
)

def evaluate_model(model, dataloader, device):
    """Evaluates the model on the test set."""
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_X = batch_X.to(device)
            outputs = model(batch_X) # No squeeze needed for image-to-image
            all_preds.append(outputs.cpu().numpy())
            all_true.append(batch_y.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_true)

def main():
    """Main evaluation script."""
    logging.info(f"Starting evaluation...")
    logging.info(f"Using device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'}")
    
    # --- Load Data and Scaler ---
    try:
        logging.info(f"Loading test data from {DATA_DIR}")
        X_test_scaled = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
        y_test_scaled = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        with open(os.path.join(DATA_DIR, 'y_scalers.pkl'), 'rb') as f:
            y_scalers = pickle.load(f)

        if input_channel_select is not None:
            logging.info(f"Selecting input channel {input_channel_select}.")
            X_test_scaled = X_test_scaled[..., input_channel_select:input_channel_select+1]
            
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Make sure the data exists in '{DATA_DIR}'. Run set_data.py first.")
        return
        
    if X_test_scaled.shape[0] == 0:
        logging.info("No test samples found. Skipping evaluation.")
        return

    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --- Get data dimensions ---
    height = X_test_scaled.shape[1]
    width = X_test_scaled.shape[2]
    channels_in = X_test_scaled.shape[3]
    channels_out = y_test_scaled.shape[3]
    logging.info(f"Input shape: ({height}, {width}, {channels_in})")
    logging.info(f"Output shape: ({height}, {width}, {channels_out})")

    all_models = [
        {
            'model_class': UMambaWrapper,
            'model_name': 'UMamba',
            'params': {'in_channels': channels_in, 'out_channels': channels_out, 'patch_size':(height, width)},
            'model_file': 'UMamba_best_model.pth'
        },
        {
            'model_class': Pix2PixWrapper,
            'model_name': 'pix2pix',
            'params': {'in_channels': channels_in, 'out_channels': channels_out, 'img_size': height},
            'model_file': 'pix2pix_best_netG.pth'
        },
        {
            'model_class': SwinUnetWrapper,
            'model_name': 'SwinUnet',
            'params': {'in_channels': channels_in, 'out_channels': channels_out, 'img_size': height},
            'model_file': 'SwinUnet_best_model.pth'
        },
        {
            'model_class': SiMaVP,
            'model_name': 'SiMaVP',
            'params': {'in_channels': channels_in, 'out_channels': channels_out},
            'model_file': 'SiMaVP_best_model.pth'
        }
    ]

    models_to_evaluate = [m for m in all_models if m['model_name'] in models_to_run]
    logging.info(f"Models to evaluate: {[m['model_name'] for m in models_to_evaluate]}")

    for model_config in models_to_evaluate:
        model_name = model_config['model_name']
        model_class = model_config['model_class']
        params = model_config['params']
        
        trained_model_dir = os.path.join(f"trained-{input_channel_select}", model_name)
        model_path = os.path.join(trained_model_dir, model_config['model_file'])
    
        if not os.path.exists(model_path):
            logging.warning(f"Model '{model_name}' not found at {model_path}. Skipping.")
            continue

        # --- Load Model ---
        logging.info(f"Loading model: {model_name} from {model_path}")
        model = model_class(**params)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        
        logging.info(f"--- Evaluating {model_name} ---")
        
        # --- Get Predictions ---
        preds_scaled, true_scaled = evaluate_model(model, test_loader, device)

        # --- Inverse Transform ---
        logging.info("Inverse transforming predictions and ground truth...")
        preds_unscaled = inverse_transform_y(preds_scaled, y_scalers)
        true_unscaled = inverse_transform_y(true_scaled, y_scalers)
        
        # --- Calculate Metrics ---
        logging.info("Calculating metrics...")
        mae = mean_absolute_error(true_unscaled.reshape(-1, channels_out), preds_unscaled.reshape(-1, channels_out))
        rmse = np.sqrt(mean_squared_error(true_unscaled.reshape(-1, channels_out), preds_unscaled.reshape(-1, channels_out)))
        psnr = peak_signal_to_noise_ratio(true_unscaled, preds_unscaled)
        ssim_score = ssim(true_unscaled, preds_unscaled)

        logging.info(f"Results for {model_name}:")
        logging.info(f"  MAE:   {mae:.6f}")
        logging.info(f"  RMSE:  {rmse:.6f}")
        logging.info(f"  PSNR:  {psnr:.6f} dB")
        logging.info(f"  SSIM:  {ssim_score:.6f}")

        # --- Save Predictions ---
        # results_file = os.path.join(RESULTS_DIR, f"{model_name}_predictions.npz")
        # np.savez(results_file, predictions=preds_unscaled, ground_truth=true_unscaled)
        # logging.info(f"Saved predictions and ground truth to {results_file}")
        logging.info("-" * 50)


if __name__ == "__main__":
    main() 