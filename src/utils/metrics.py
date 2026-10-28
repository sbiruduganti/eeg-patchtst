# src/utils/metrics.py
import numpy as np
import torch
import scipy.signal as signal

def compute_metrics(y_true, y_pred, fs=250):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Welch's PSD spectral similarity
    _, psd_true = signal.welch(y_true, fs=fs, axis=-1)
    _, psd_pred = signal.welch(y_pred, fs=fs, axis=-1)
    
    psd_true_flat = psd_true.reshape(psd_true.shape[0], -1)
    psd_pred_flat = psd_pred.reshape(psd_pred.shape[0], -1)
    
    dot_product = np.sum(psd_true_flat * psd_pred_flat, axis=-1)
    norm_true = np.linalg.norm(psd_true_flat, axis=-1)
    norm_pred = np.linalg.norm(psd_pred_flat, axis=-1)
    
    spectral_sim = np.mean(dot_product / (norm_true * norm_pred + 1e-8))
    
    return {"MAE": mae, "RMSE": rmse, "Spectral_Similarity": spectral_sim}