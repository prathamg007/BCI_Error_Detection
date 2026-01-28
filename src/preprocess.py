import pandas as pd
import numpy as np
import os
from scipy.signal import butter, lfilter
from src import config

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Creates Butterworth bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, fs=200):
    """Applies Bandpass filter to EEG signal."""
    b, a = butter_bandpass(config.LOW_CUT, config.HIGH_CUT, fs, order=4)
    return lfilter(b, a, data, axis=0)

def extract_epochs(df, filename, labels_map=None, mode='train'):
    """
    Slices the continuous EEG signal into 1.3s epochs based on Feedback Events.
    """
    # Exclude non-signal columns
    ignore_cols = ['Time', 'FeedBackEvent', 'EOG', 'source_file']
    sig_cols = [c for c in df.columns if c not in ignore_cols]
    
    # Preprocessing
    raw_signal = df[sig_cols].values
    clean_signal = apply_filter(raw_signal, fs=config.FS)
    
    # Find Events (FeedBackEvent == 1)
    event_indices = df.index[df['FeedBackEvent'] == 1].tolist()
    
    X, y, ids = [], [], []
    win_samples = int(config.WINDOW_SEC * config.FS)
    
    for i, idx in enumerate(event_indices):
        start = idx
        end = idx + win_samples
        
        if end < len(df):
            # Extraction & Downsampling
            epoch = clean_signal[start:end, :]
            epoch_down = epoch[::config.DOWNSAMPLE_RATE, :]
            
            # Generate ID for Submission
            base_name = os.path.basename(filename).replace('.csv', '')
            fb_id = f"{base_name}_feedback_{i+1}"
            
            X.append(epoch_down.flatten())
            ids.append(fb_id)
            
            # Label Handling
            if mode == 'train':
                label = 0
                if labels_map:
                    label = labels_map.get(fb_id, 0)
                # Fallback: if label not in map, assume 0 or check dataframe
                elif 'FeedBackEvent' in df.columns:
                     label = df.iloc[idx]['FeedBackEvent']
                y.append(label)

    return np.array(X), np.array(y), np.array(ids)