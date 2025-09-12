import os, numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def get_train_val_loaders(train_data_folder, val_data_folder, batch_size, channel_select=None):
    X_train_scaled = np.load(os.path.join(train_data_folder, 'X_train.npy'))
    y_train_scaled = np.load(os.path.join(train_data_folder, 'y_train.npy'))
    X_val_scaled = np.load(os.path.join(val_data_folder, 'X_val.npy'))
    y_val_scaled = np.load(os.path.join(val_data_folder, 'y_val.npy'))
    print("\nData validation:")
    print(f"y_train range: [{y_train_scaled.min()}, {y_train_scaled.max()}]")
    print(f"y_val range: [{y_val_scaled.min()}, {y_val_scaled.max()}]") 
    
    if channel_select is not None:
        print(f"Selecting channel {channel_select} from input data.")
        X_train_scaled = X_train_scaled[..., channel_select:channel_select+1]
        X_val_scaled = X_val_scaled[..., channel_select:channel_select+1]
        
    print("X_train_scaled.shape:", X_train_scaled.shape)
    # Convert to PyTorch datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled),torch.FloatTensor(y_val_scaled))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)

    return train_loader, val_loader


def save_training_plots(history, save_dir, model_name):
    """Save loss curves and learning rate schedule to PNG files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Loss curves
    ax1.plot(history['loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training History')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Learning rate schedule
    ax2.plot(history['lr'], color='green')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    
    plt.tight_layout()
    
    # Save and optionally display
    plot_path = os.path.join(save_dir, f'{model_name}_training_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    plt.close()









