import os
import sys
import json
import csv
import torch
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from skimage.metrics import structural_similarity
from pathlib import Path

# ----------------------
# Config
# ----------------------
gpu_id = 0
input_channel_select = None  # 0, 1, or None (all)
models_to_run = ["pix2pix"]
DATA_DIR = "/home/dll0505/AdditionalStorage/thesis/sen12-ms/sub_dataset/data-256-_0.02_10000"
MODEL_DIR = "trained-lpips_0.02_10000"
batch_size = 16

# ----------------------
# Repo paths (optional)
# ----------------------
umamba_repo_path = Path(__file__).parent / 'mymodels/U-Mamba'
umamba_pkg_path = umamba_repo_path / 'umamba'
if umamba_repo_path.is_dir() and umamba_pkg_path.is_dir():
    sys.path.insert(0, str(umamba_repo_path))
    sys.path.insert(0, str(umamba_pkg_path))
else:
    print("U-Mamba repository not found or is incomplete.")

gan_repo_path = Path(__file__).parent / 'mymodels/pytorch-CycleGAN-and-pix2pix'
if gan_repo_path.is_dir():
    sys.path.insert(0, str(gan_repo_path))
else:
    print(f"pytorch-CycleGAN-and-pix2pix repository not found at {gan_repo_path}")

swin_repo_path = Path(__file__).parent / 'mymodels/Swin-Unet'
if swin_repo_path.is_dir():
    sys.path.insert(0, str(swin_repo_path))
else:
    print(f"Swin-Unet repository not found at {swin_repo_path}")

# ----------------------
# Model imports (pix2pix)
# ----------------------
from mymodels.cp2pwrapper import Pix2PixWrapper  # must exist

# ----------------------
# Optional LPIPS
# ----------------------
_lpips_available = True
try:
    import lpips  # pip install lpips
except Exception as e:
    _lpips_available = False

# ----------------------
# Helpers
# ----------------------
def ssim_batch_color(y_true, y_pred):
    """
    SSIM for multi-channel images (batch). Uses global data_range from y_true.
    """
    ssim_scores = []
    data_range = np.max(y_true) - np.min(y_true)
    for i in range(y_true.shape[0]):
        score = structural_similarity(
            y_true[i], y_pred[i],
            channel_axis=-1, data_range=data_range
        )
        ssim_scores.append(score)
    return np.mean(ssim_scores)

def ssim_per_band(y_true, y_pred):
    """
    SSIM per band (treat each band as grayscale). Returns list length C.
    """
    N, H, W, C = y_true.shape
    scores = []
    for c in range(C):
        yt = y_true[..., c]
        yp = y_pred[..., c]
        dr = float(np.max(yt) - np.min(yt)) if np.max(yt) > np.min(yt) else 1.0
        band_scores = []
        for i in range(N):
            band_scores.append(structural_similarity(yt[i], yp[i], data_range=dr))
        scores.append(float(np.mean(band_scores)))
    return scores

def psnr(y_true, y_pred):
    """PSNR for color/multichannel images (batch)."""
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0: return float('inf')
    data_range = np.max(y_true) - np.min(y_true)
    if data_range == 0: return float('-inf')
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

def psnr_per_band(y_true, y_pred):
    """PSNR per band."""
    C = y_true.shape[-1]
    vals = []
    for c in range(C):
        yt = y_true[..., c]
        yp = y_pred[..., c]
        mse = np.mean((yt - yp) ** 2)
        if mse == 0:
            vals.append(float('inf'))
            continue
        dr = np.max(yt) - np.min(yt)
        if dr == 0:
            vals.append(float('-inf'))
            continue
        vals.append(20 * np.log10(dr) - 10 * np.log10(mse))
    return vals

# Inverse transform using S2_clip.json (no sklearn scalers)
def load_s2_clip_json(path):
    with open(path, "r") as f:
        clip = json.load(f)
    clip_int = {}
    for k, v in clip.items():
        ki = int(k) if isinstance(k, str) else k
        clip_int[ki] = tuple(v)
    return clip_int

def inverse_from_tanh_using_clip(y_scaled, s2_clip):
    """
    y_scaled: (N,H,W,C) in [-1,1]
    s2_clip: dict {1:(L1,H1), ..., 13:(L13,H13)}
    returns original S2 units
    """
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    y01 = (y_scaled + 1.0) * 0.5  # [-1,1] -> [0,1]
    y_unscaled = np.empty_like(y01, dtype=np.float32)
    C = y01.shape[-1]
    assert C == 13, f"Expected 13 channels for S2, got {C}"
    for c in range(C):
        lo, hi = s2_clip[c + 1]  # bands are 1-indexed
        y_unscaled[..., c] = lo + y01[..., c] * (hi - lo)
    return y_unscaled

# Sentinel-2 band names (order B1..B12 with B8A as band 9 in typical 13-band stacks)
BAND_NAMES = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]

# Spectral groups for summaries
GROUPS = {
    "Coastal": ["B1"],
    "VIS": ["B2","B3","B4"],
    "RedEdge": ["B5","B6","B7"],
    "NIR": ["B8","B8A"],
    "WV": ["B9"],
    "Cirrus": ["B10"],
    "SWIR": ["B11","B12"],
    "10m": ["B2","B3","B4","B8"],
    "20m": ["B5","B6","B7","B8A","B11","B12"],
    "60m": ["B1","B9","B10"],
}

def indices_for(names):
    return [BAND_NAMES.index(n) for n in names]

def group_metrics_from_perband(per_band_vals):
    """
    per_band_vals: dict {metric_name: [C-length list]}
    returns dict {metric_name__group: value}
    """
    out = {}
    for gname, gmembers in GROUPS.items():
        idxs = indices_for(gmembers)
        for metric, arr in per_band_vals.items():
            vals = [arr[i] for i in idxs]
            out[f"{metric}__{gname}"] = float(np.mean(vals))
    return out

# ----------------------
# LPIPS prep & metric (RGB from B4,B3,B2)
# ----------------------
def _extract_rgb_01(truth_orig, pred_orig, s2_clip):
    """
    Build RGB in [0,1] using bands B4,B3,B2 scaled with their band clip ranges.
    truth_orig, pred_orig: (N,H,W,13) in ORIGINAL units.
    Returns two arrays in shape (N,H,W,3) in [0,1].
    """
    # Band indices
    iR = BAND_NAMES.index("B4")
    iG = BAND_NAMES.index("B3")
    iB = BAND_NAMES.index("B2")

    def scale01(x, lo, hi):
        return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)

    loR, hiR = s2_clip[iR + 1]
    loG, hiG = s2_clip[iG + 1]
    loB, hiB = s2_clip[iB + 1]

    true_rgb = np.stack([
        scale01(truth_orig[..., iR], loR, hiR),
        scale01(truth_orig[..., iG], loG, hiG),
        scale01(truth_orig[..., iB], loB, hiB),
    ], axis=-1).astype(np.float32)

    pred_rgb = np.stack([
        scale01(pred_orig[..., iR], loR, hiR),
        scale01(pred_orig[..., iG], loG, hiG),
        scale01(pred_orig[..., iB], loB, hiB),
    ], axis=-1).astype(np.float32)

    return true_rgb, pred_rgb

def compute_lpips_rgb(truth_orig, pred_orig, s2_clip, device):
    """
    Computes LPIPS on RGB (B4,B3,B2). Returns mean LPIPS.
    truth_orig, pred_orig: (N,H,W,13) in ORIGINAL units.
    """
    if not _lpips_available:
        return None

    try:
        loss_fn = lpips.LPIPS(net='alex').to(device)
        loss_fn.eval()
    except Exception as e:
        logging.warning(f"Could not initialize LPIPS: {e}")
        return None

    true_rgb_01, pred_rgb_01 = _extract_rgb_01(truth_orig, pred_orig, s2_clip)

    # Map [0,1] -> [-1,1] and to NCHW torch tensors
    def to_lpips_tensor(x01):
        x = (x01 * 2.0 - 1.0).transpose(0, 3, 1, 2)  # N,H,W,C -> N,C,H,W
        return torch.from_numpy(x).float().to(device)

    t = to_lpips_tensor(true_rgb_01)
    p = to_lpips_tensor(pred_rgb_01)

    # Batch in reasonable chunks to avoid OOM
    bs = 8
    scores = []
    with torch.no_grad():
        for i in range(0, t.shape[0], bs):
            s = loss_fn(p[i:i+bs], t[i:i+bs])  # lpips expects inputs in [-1,1]
            scores.append(s.squeeze().detach().cpu().numpy())
    scores = np.concatenate([np.atleast_1d(s) for s in scores]).astype(np.float32)
    return float(np.mean(scores))

# ----------------------
# SAM metric (Spectral Angle Mapper) in degrees
# ----------------------
def compute_sam_deg(truth_orig, pred_orig, eps=1e-8):
    """
    truth_orig, pred_orig: (N,H,W,C)
    Returns dict with mean, median, p25, p75 of SAM (degrees).
    """
    # Flatten spatial dims
    T = truth_orig.reshape(-1, truth_orig.shape[-1]).astype(np.float64)
    P = pred_orig.reshape(-1, pred_orig.shape[-1]).astype(np.float64)

    # Compute per-pixel spectral angles
    dot = np.sum(T * P, axis=1)
    nT = np.linalg.norm(T, axis=1)
    nP = np.linalg.norm(P, axis=1)
    denom = np.maximum(nT * nP, eps)
    cosang = np.clip(dot / denom, -1.0, 1.0)
    ang_rad = np.arccos(cosang)
    ang_deg = np.degrees(ang_rad)

    # Robust summaries
    mean = float(np.mean(ang_deg))
    median = float(np.median(ang_deg))
    p25 = float(np.percentile(ang_deg, 25))
    p75 = float(np.percentile(ang_deg, 75))
    return {
        "SAM_mean_deg": mean,
        "SAM_median_deg": median,
        "SAM_p25_deg": p25,
        "SAM_p75_deg": p75,
    }

# ----------------------
# Device
# ----------------------
if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    device_name = torch.cuda.get_device_name(device)
else:
    device = torch.device("cpu")
    device_name = "cpu"
print(f"Using device: {device_name}")
if torch.cuda.is_available():
    print(f"Current CUDA device index: {torch.cuda.current_device()}")

# RESULTS_DIR = f"trained-{input_channel_select}/evaluation_results-{input_channel_select}"
RESULTS_DIR = f"{MODEL_DIR}/evaluation_results_per_band_all_losses"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------
# Logging
# ----------------------
log_file = os.path.join(RESULTS_DIR, f'evaluation_metrics_channel_{input_channel_select}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)

# ----------------------
# Evaluation
# ----------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            all_preds.append(outputs.cpu().numpy())
            all_true.append(batch_y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_true)

def main():
    logging.info("Starting evaluation...")
    logging.info(f"Using device: {device_name}")

    # --- Load data and S2 clip table ---
    try:
        logging.info(f"Loading test data from {DATA_DIR}")
        X_test_scaled = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
        y_test_scaled = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        s2_clip_path = os.path.join(DATA_DIR, 'S2_clip.json')
        s2_clip = load_s2_clip_json(s2_clip_path)

        if input_channel_select is not None:
            logging.info(f"Selecting input channel {input_channel_select}.")
            X_test_scaled = X_test_scaled[..., input_channel_select:input_channel_select+1]

    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Make sure the data exists in '{DATA_DIR}'.")
        return

    if X_test_scaled.shape[0] == 0:
        logging.info("No test samples found. Skipping evaluation.")
        return

    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Shapes
    H, W = X_test_scaled.shape[1], X_test_scaled.shape[2]
    C_in = X_test_scaled.shape[3]
    C_out = y_test_scaled.shape[3]
    logging.info(f"Input shape:  ({H}, {W}, {C_in})")
    logging.info(f"Output shape: ({H}, {W}, {C_out})")

    # --- Models (pix2pix) ---
    all_models = [
        {
            'model_class': Pix2PixWrapper,
            'model_name': 'pix2pix',
            'params': {'in_channels': C_in, 'out_channels': C_out, 'img_size': H},
            'model_file': 'pix2pix_best_netG.pth'
        }
    ]
    models_to_evaluate = [m for m in all_models if m['model_name'] in models_to_run]
    logging.info(f"Models to evaluate: {[m['model_name'] for m in models_to_evaluate]}")

    # CSV outputs
    overall_csv = os.path.join(RESULTS_DIR, "metrics_overall.csv")
    perband_csv = os.path.join(RESULTS_DIR, "metrics_per_band.csv")

    for model_config in models_to_evaluate:
        model_name = model_config['model_name']
        model_class = model_config['model_class']
        params = model_config['params']
        # trained_model_dir = os.path.join(f"trained-{input_channel_select}", model_name)
        trained_model_dir = os.path.join(MODEL_DIR, model_name)
        model_path = os.path.join(trained_model_dir, model_config['model_file'])

        if not os.path.exists(model_path):
            logging.warning(f"Model '{model_name}' not found at {model_path}. Skipping.")
            continue

        # Load model (safer)
        logging.info(f"Loading model: {model_name} from {model_path}")
        model = model_class(**params)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)

        logging.info(f"--- Evaluating {model_name} ---")

        # Predict
        preds_scaled, true_scaled = evaluate_model(model, test_loader, device)

        # Inverse to original units
        logging.info("Inverse transforming predictions and ground truth using S2_clip.json ...")
        preds = inverse_from_tanh_using_clip(preds_scaled, s2_clip)
        truth = inverse_from_tanh_using_clip(true_scaled,  s2_clip)

        # ---------------- Overall metrics (ORIGINAL units) ----------------
        logging.info("Calculating overall metrics (original units)...")
        mae_all = mean_absolute_error(truth.reshape(-1, C_out), preds.reshape(-1, C_out))
        rmse_all = np.sqrt(mean_squared_error(truth.reshape(-1, C_out), preds.reshape(-1, C_out)))
        psnr_all = psnr(truth, preds)
        ssim_all = ssim_batch_color(truth, preds)

        logging.info(f"  MAE:   {mae_all:.6f}")
        logging.info(f"  RMSE:  {rmse_all:.6f}")
        logging.info(f"  PSNR:  {psnr_all:.6f} dB")
        logging.info(f"  SSIM:  {ssim_all:.6f}")

        # ---------------- Per-band metrics (ORIGINAL units) ----------------
        logging.info("Calculating per-band metrics (original units)...")
        mae_per = []
        rmse_per = []
        for c in range(C_out):
            yt = truth[..., c].reshape(-1)
            yp = preds[..., c].reshape(-1)
            mae_per.append(mean_absolute_error(yt, yp))
            rmse_per.append(np.sqrt(mean_squared_error(yt, yp)))
        psnr_per = psnr_per_band(truth, preds)
        ssim_per = ssim_per_band(truth, preds)

        # ---------------- Normalized-space metrics ([-1,1]) ----------------
        logging.info("Calculating normalized-space metrics ([-1,1])...")
        mae_norm = mean_absolute_error(true_scaled.reshape(-1, C_out), preds_scaled.reshape(-1, C_out))
        rmse_norm = np.sqrt(mean_squared_error(true_scaled.reshape(-1, C_out), preds_scaled.reshape(-1, C_out)))

        mae_norm_per = []
        rmse_norm_per = []
        for c in range(C_out):
            yt = true_scaled[..., c].reshape(-1)
            yp = preds_scaled[..., c].reshape(-1)
            mae_norm_per.append(mean_absolute_error(yt, yp))
            rmse_norm_per.append(np.sqrt(mean_squared_error(yt, yp)))

        # ---------------- LPIPS (RGB from B4,B3,B2) ----------------
        if _lpips_available:
            logging.info("Computing LPIPS (RGB B4,B3,B2)...")
            lpips_rgb_mean = compute_lpips_rgb(truth, preds, s2_clip, device)
            if lpips_rgb_mean is None:
                logging.warning("LPIPS not available at runtime; skipping.")
        else:
            lpips_rgb_mean = None
            logging.warning("lpips package not found; install with `pip install lpips` to enable LPIPS metric.")

        if lpips_rgb_mean is not None:
            logging.info(f"  LPIPS_RGB: {lpips_rgb_mean:.6f}")

        # ---------------- SAM (Spectral Angle Mapper) ----------------
        logging.info("Computing SAM (degrees) in original 13-band space...")
        sam_stats = compute_sam_deg(truth, preds)
        logging.info(f"  SAM_mean_deg:   {sam_stats['SAM_mean_deg']:.4f}")
        logging.info(f"  SAM_median_deg: {sam_stats['SAM_median_deg']:.4f}")
        logging.info(f"  SAM_p25_deg:    {sam_stats['SAM_p25_deg']:.4f}")
        logging.info(f"  SAM_p75_deg:    {sam_stats['SAM_p75_deg']:.4f}")

        # ---------------- Group summaries (from per-band ORIGINAL) ---------
        perband_dict = {
            "MAE": mae_per,
            "RMSE": rmse_per,
            "PSNR": psnr_per,
            "SSIM": ssim_per
        }
        group_summary = group_metrics_from_perband(perband_dict)

        # ---------------- Log summaries ------------------------------------
        logging.info("Per-band (original units):")
        for c, name in enumerate(BAND_NAMES):
            logging.info(f"  {name:>4} | MAE {mae_per[c]:8.3f} | RMSE {rmse_per[c]:8.3f} | "
                         f"PSNR {psnr_per[c]:7.3f} | SSIM {ssim_per[c]:.4f}")

        logging.info("Group summaries (original units):")
        for k, v in group_summary.items():
            logging.info(f"  {k}: {v:.4f}")

        logging.info("Normalized-space overview ([-1,1]):")
        logging.info(f"  MAE_norm:  {mae_norm:.6f}")
        logging.info(f"  RMSE_norm: {rmse_norm:.6f}")

        # ---------------- Save CSV -----------------------------------------
        # Overall CSV (append)
        overall_headers = [
            "model","MAE","RMSE","PSNR","SSIM",
            "MAE_norm","RMSE_norm",
            # New metrics:
            "LPIPS_RGB",
            "SAM_mean_deg","SAM_median_deg","SAM_p25_deg","SAM_p75_deg"
        ] + sorted(group_summary.keys())

        overall_row = [
            model_name, mae_all, rmse_all, psnr_all, ssim_all,
            mae_norm, rmse_norm,
            (lpips_rgb_mean if lpips_rgb_mean is not None else ""),
            sam_stats["SAM_mean_deg"], sam_stats["SAM_median_deg"], sam_stats["SAM_p25_deg"], sam_stats["SAM_p75_deg"]
        ] + [group_summary[k] for k in sorted(group_summary.keys())]

        write_header = not os.path.exists(overall_csv)
        with open(overall_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(overall_headers)
            w.writerow(overall_row)

        # Per-band CSV (write once per run/model)
        # (LPIPS and SAM are not per-band; keep per-band file as-is)
        perband_headers = ["model","band","MAE","RMSE","PSNR","SSIM","MAE_norm","RMSE_norm"]
        write_header_band = not os.path.exists(perband_csv)
        with open(perband_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header_band:
                w.writerow(perband_headers)
            for c, name in enumerate(BAND_NAMES):
                w.writerow([
                    model_name, name,
                    mae_per[c], rmse_per[c], psnr_per[c], ssim_per[c],
                    mae_norm_per[c], rmse_norm_per[c]
                ])

        logging.info("-" * 50)

if __name__ == "__main__":
    main()
