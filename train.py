# train.py
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import TUEGDataset
from src.models.PatchTST import Model as PatchTSTModel
from src.utils.metrics import compute_metrics


class ConfigArgs:
    """Wrapper to pass dictionary config values as object attributes to PatchTST."""
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        # x shape: [batch, channels, seq_len]
        x, y = x.to(device), y.to(device)
        
        # PatchTST expects [batch, seq_len, channels]
        x_in = x.permute(0, 2, 1)
        
        optimizer.zero_grad()
        output = model(x_in)  # output: [batch, pred_len, channels]
        
        # Permute target to match output: [batch, pred_len, channels]
        y_target = y.permute(0, 2, 1)
        loss = criterion(output, y_target)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, fs=250):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            x_in = x.permute(0, 2, 1)
            
            output = model(x_in) # [batch, pred_len, channels]
            
            # Permute back to [batch, channels, pred_len] for metric calculations
            all_preds.append(output.permute(0, 2, 1).cpu())
            all_targets.append(y)
            
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return compute_metrics(all_targets, all_preds, fs=fs)


def main():
    cfg = load_config()
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    
    fs = cfg["data"]["fs"]
    seq_len = int(cfg["data"]["input_sec"] * fs)
    pred_len = int(cfg["data"]["target_sec"] * fs)
    
    # 1. Dataset & DataLoader Setup
    dataset = TUEGDataset(
        data_files=[cfg["data"]["data_path"]], 
        fs=fs, 
        input_sec=cfg["data"]["input_sec"], 
        target_sec=cfg["data"]["target_sec"],
        stride_sec=cfg["data"]["stride_sec"]
    )
    dataloader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    
    # 2. Configure PatchTST Model Arguments
    model_args = ConfigArgs(
        enc_in=cfg["model"]["c_in"],        # Changed c_in -> enc_in to match PatchTST.py
        seq_len=seq_len,
        pred_len=pred_len,
        patch_len=cfg["model"]["patch_len"],
        stride=cfg["model"]["stride"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        e_layers=cfg["model"]["e_layers"],
        d_ff=cfg["model"]["d_model"] * 4,
        dropout=cfg["model"]["dropout"],
        fc_dropout=0.0,                    # Added missing fc_dropout
        head_dropout=0.0,
        individual=0,
        padding_patch='end',               # Added missing padding_patch
        revin=1,
        affine=0,
        subtract_last=0,
        decomposition=0,                   # Added missing decomposition
        kernel_size=25                      # Added missing kernel_size
    )
    
    # 3. Model, Loss, and Optimizer Initialization
    model = PatchTSTModel(model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = nn.MSELoss()
    
    print(f"Starting Training | Input: {cfg['data']['input_sec']}s ({seq_len} steps) -> Target: {cfg['data']['target_sec']}s ({pred_len} steps)")
    
    # 4. Main Epoch Loop
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        metrics = evaluate(model, dataloader, device, fs=fs)
        
        print(f"Epoch {epoch:02d} | Train Loss: {loss:.6f} | MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | Spectral Sim: {metrics['Spectral_Similarity']:.4f}")


if __name__ == "__main__":
    main()