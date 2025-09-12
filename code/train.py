import os, time, logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import sys
from pathlib import Path


gpu_id = 1
if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)  
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device("cpu")
print(f"Using device: {torch.cuda.get_device_name(device)}")
print(f"Current CUDA device index: {torch.cuda.current_device()}")


# --- Model Selection ---
# Choose between 'umamba', 'pix2pix', 'swinunet', and 'simavp'
model_choice = "simavp" 
input_channel_select = 1 # Can be 0, 1, or None (for all channels)

epochs = 150
batch_size = 64
lr = 1e-4
patience = 15
data_dir = f"data-256"


# --- Path Setup ---
if model_choice == "umamba":
    # Add U-Mamba project to the Python path
    umamba_repo_path = Path(__file__).parent / 'mymodels/U-Mamba'
    umamba_pkg_path = umamba_repo_path / 'umamba'
    if umamba_repo_path.is_dir() and umamba_pkg_path.is_dir():
        sys.path.insert(0, str(umamba_repo_path))
        sys.path.insert(0, str(umamba_pkg_path))
    else:
        print(f"U-Mamba repository not found or is incomplete.")
        sys.exit(1)
elif model_choice == "pix2pix":
    # Add pix2pix project to the Python path
    gan_repo_path = Path(__file__).parent / 'mymodels/pytorch-CycleGAN-and-pix2pix'
    if gan_repo_path.is_dir():
        sys.path.insert(0, str(gan_repo_path))
    else:
        print(f"pytorch-CycleGAN-and-pix2pix repository not found at {gan_repo_path}")
        sys.exit(1)
elif model_choice == "swinunet":
    # Add Swin-Unet project to the Python path
    swin_repo_path = Path(__file__).parent / 'mymodels/Swin-Unet'
    if swin_repo_path.is_dir():
        sys.path.insert(0, str(swin_repo_path))
    else:
        print(f"Swin-Unet repository not found at {swin_repo_path}")
        sys.exit(1)


from myutils import get_train_val_loaders, save_training_plots




print("Loading data")
train_loader, val_loader = get_train_val_loaders(
    train_data_folder=data_dir,
    val_data_folder=data_dir,
    batch_size=batch_size,
    channel_select=input_channel_select
)
data_shape = next(iter(train_loader))[0].shape
print(f"Data shape from loader: {data_shape}")
height = data_shape[1]
width = data_shape[2]
channels_in = data_shape[3]
channels_out = next(iter(train_loader))[1].shape[3]

print(f"Input dimensions: {height}x{width}")
print(f"Input channels: {channels_in}")
print(f"Output channels: {channels_out}")


# --- Model, Loss, Optimizer Setup ---
if model_choice == "umamba":
    from mymodels.UMambaWrapper import UMambaWrapper
    model = UMambaWrapper(
                in_channels=channels_in,
                out_channels=channels_out,
                patch_size=(height, width)
    )
    model_name = f"UMamba"
    model = model.to(device)
    
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {param_count:,}")

elif model_choice == "pix2pix":
    from mymodels.cp2pwrapper import Pix2PixWrapper
    from models.networks import define_D, GANLoss

    # Create Generator and Discriminator
    netG = Pix2PixWrapper(in_channels=channels_in, out_channels=channels_out, img_size=height).to(device)
    netD = define_D(input_nc=channels_in + channels_out, ndf=64, netD='basic').to(device)
    
    model_name = f"pix2pix"

    # Define Loss Functions
    criterionGAN = GANLoss(gan_mode='vanilla').to(device)
    criterionL1 = torch.nn.L1Loss()
    lambda_L1 = 100.0

    # Define Optimizers (betas from pix2pix paper)
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    # Schedulers
    scheduler_G = ReduceLROnPlateau(optimizer_G, 'min', patience=patience//2, factor=0.5)
    scheduler_D = ReduceLROnPlateau(optimizer_D, 'min', patience=patience//2, factor=0.5)

    param_count_G = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    param_count_D = sum(p.numel() for p in netD.parameters() if p.requires_grad)
    print(f"Generator parameter count: {param_count_G:,}")
    print(f"Discriminator parameter count: {param_count_D:,}")

elif model_choice == "swinunet":
    from mymodels.swinunetwrapper import SwinUnetWrapper
    model = SwinUnetWrapper(
                in_channels=channels_in,
                out_channels=channels_out,
                img_size=height
    )
    model_name = f"SwinUnet"
    model = model.to(device)
    
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {param_count:,}")

elif model_choice == "simavp":
    from mymodels.SiMaVP import SiMaVP
    model = SiMaVP(
                in_channels=channels_in,
                out_channels=channels_out
    )
    model_name = f"SiMaVP"
    model = model.to(device)
    
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter count: {param_count:,}")


if input_channel_select is not None:
    model_name = f"{model_name}"


# Training state
best_val_loss = float('inf')
wait = 0
history = {'loss': [],'val_loss': [], 'lr': []}
if model_choice == "pix2pix":
    history['D_loss'] = []


# Create save directory
save_dir = f"trained-{input_channel_select}/{model_name}"
os.makedirs(save_dir, exist_ok=True)

log_file = os.path.join(save_dir, f'{model_name}_training.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()


# --- Training loop ---
for epoch in tqdm(range(epochs), desc="Training", unit="epoch", position=0):
    epoch_start = time.time()
    
    if model_choice == "umamba" or model_choice == "swinunet" or model_choice == "simavp":
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False, position=1) as batch_pbar:
            for batch_X, batch_y in batch_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                with torch.no_grad():
                    train_loss += loss.item() * batch_X.size(0)

                batch_pbar.set_postfix({'loss': f"{loss.item():.6f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        train_loss /= len(train_loader.dataset)
        history['loss'].append(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
    
    elif model_choice == "pix2pix":
        netG.train()
        netD.train()
        train_G_loss = 0.0
        train_D_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False, position=1) as batch_pbar:
            for real_A, real_B in batch_pbar:
                real_A, real_B = real_A.to(device), real_B.to(device)
                fake_B = netG(real_A)

                # --- Update Discriminator ---
                optimizer_D.zero_grad()
                # Real
                pred_real = netD(torch.cat((real_A, real_B), -1).permute(0, 3, 1, 2))
                loss_D_real = criterionGAN(pred_real, True)
                # Fake
                pred_fake = netD(torch.cat((real_A, fake_B), -1).permute(0, 3, 1, 2).detach())
                loss_D_fake = criterionGAN(pred_fake, False)
                # Combined loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()

                # --- Update Generator ---
                optimizer_G.zero_grad()
                # GAN loss
                pred_fake = netD(torch.cat((real_A, fake_B), -1).permute(0, 3, 1, 2))
                loss_G_GAN = criterionGAN(pred_fake, True)
                # L1 loss
                loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
                # Combined loss
                loss_G = loss_G_GAN + loss_G_L1
                loss_G.backward()
                optimizer_G.step()

                train_G_loss += loss_G.item() * real_A.size(0)
                train_D_loss += loss_D.item() * real_A.size(0)
                
                batch_pbar.set_postfix({'G_loss': f"{loss_G.item():.4f}", 'D_loss': f"{loss_D.item():.4f}", 'lr': f"{optimizer_G.param_groups[0]['lr']:.2e}"})
        
        train_G_loss /= len(train_loader.dataset)
        train_D_loss /= len(train_loader.dataset)
        history['loss'].append(train_G_loss)
        history['D_loss'].append(train_D_loss)
        current_lr = optimizer_G.param_groups[0]['lr']

    # --- Validation phase ---
    val_loss = 0.0
    val_mae = 0.0
    
    eval_model = model if model_choice in ["umamba", "swinunet", "simavp"] else netG
    eval_model.eval()

    with tqdm(val_loader, desc="Validating", unit="batch", leave=False, position=2) as val_pbar:
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = eval_model(batch_X)
                loss = criterionL1(outputs, batch_y) if model_choice == "pix2pix" else criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                val_pbar.set_postfix({'val_loss': f"{loss.item():.6f}"})

    val_loss /= len(val_loader.dataset)
    history['val_loss'].append(val_loss)
    history['lr'].append(current_lr)
    
    # Update learning rate
    if model_choice in ["umamba", "swinunet", "simavp"]:
        scheduler.step(val_loss)
    elif model_choice == "pix2pix":
        scheduler_G.step(val_loss)
        scheduler_D.step(val_loss)
    
    # Print and log key epoch metrics
    tqdm.write(f"\nEpoch {epoch+1}/{epochs} - {time.time()-epoch_start:.1f}s")
    if model_choice == "pix2pix":
        tqdm.write(f"Train G_Loss: {train_G_loss:.6f} | Train D_Loss: {train_D_loss:.6f} | Val Loss: {val_loss:.6f}")
        logger.info(f"Train G_Loss: {train_G_loss:.6f} | Train D_Loss: {train_D_loss:.6f} | Val Loss: {val_loss:.6f}")
    else:
        tqdm.write(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        logger.info(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    tqdm.write(f"Learning Rate: {current_lr:.2e}")
    logger.info(f"Learning Rate: {current_lr:.2e}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        if model_choice in ["umamba", "swinunet", "simavp"]:
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_best_model.pth'))
        elif model_choice == "pix2pix":
            torch.save(netG.state_dict(), os.path.join(save_dir, f'{model_name}_best_netG.pth'))
            torch.save(netD.state_dict(), os.path.join(save_dir, f'{model_name}_best_netD.pth'))
    else:
        wait += 1
        if wait >= patience:
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch+1}")
            logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

save_training_plots(history, save_dir, model_name)

