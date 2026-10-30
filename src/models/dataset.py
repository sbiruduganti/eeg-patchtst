import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.signal as signal

class TUEGDataset(Dataset):
    """
    Dataset loader for TUEG preprocessed 30-second EEG segments.
    
    Parameters:
    -----------
    data_files : list of str or np.ndarray
        List of paths to preprocessed files or a pre-loaded array of shape (N, Channels, TimeSteps).
        Each 30-second segment is assumed to have shape (Channels, 30 * sampling_rate).
    fs : Sampling frequency in Hz (default: 250 Hz).
    input_sec : Input window duration in seconds (default: 5.0 s).
    target_sec : Target window duration in seconds (default: 0.5 s).
    stride_sec : Step size between input windows in seconds. If None, defaults to input_sec (non-overlapping inputs).
    """
    def __init__(self, data_files, fs=250, input_sec=5.0, target_sec=0.5, stride_sec=None):
        super().__init__()
        self.fs = fs
        self.seq_len = int(input_sec * fs)      # gives number of samples inputted
        self.pred_len = int(target_sec * fs)    # number of samples to predict
        self.stride = int((stride_sec if stride_sec is not None else input_sec) * fs) # overlap between samples
        
        # Total duration required for a single input + target pair
        self.window_len = self.seq_len + self.pred_len
        
        self.samples = []
        self._process_segments(data_files)

    def _process_segments(self, data_files):
        """
        Extracts valid (input, target) slices strictly within each 30-second segment boundary.
        """
        for seg in data_files:
            # Load numpy array if file path is provided
            if isinstance(seg, str):
                seg = np.load(seg)  # Expected shape: (Channels, Total_Time_Steps)
            
            # Ensure time dimension is last
            if seg.shape[0] > seg.shape[1] and seg.shape[0] < 100:
                seg = seg.T  # Transpose to (Channels, TimeSteps) if needed
                
            num_time_steps = seg.shape[-1]
            
            # Slice non-overlapping 5s windows (0-5s -> 5-5.5s, 5-10s -> 10-10.5s, etc.)
            for start_idx in range(0, num_time_steps - self.window_len + 1, self.stride):
                end_in = start_idx + self.seq_len
                end_tgt = end_in + self.pred_len
                
                input_window = seg[:, start_idx:end_in]
                target_window = seg[:, end_in:end_tgt]
                
                self.samples.append((input_window, target_window))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # Shape: (Channels, TimeSteps) converted to FloatTensor
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


#