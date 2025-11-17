import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import logging
import os
import json

logger = logging.getLogger(__name__)

def load_curve_data(data_path: str) -> np.ndarray:
    """
    Loads 1D curve data (X, Y) from .csv, .txt, or .npy files.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    try:
        if data_path.endswith(('.csv', '.txt')):
            # Assume comma delimiter, skip 1 header row by default
            data = np.loadtxt(data_path, delimiter=',', skiprows=1)
        elif data_path.endswith('.npy'):
            data = np.load(data_path)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")
            
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"Data must be a 2-column array (X, Y). Got shape {data.shape}")
        
        logger.info(f"Loaded curve data from {data_path}, shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading curve data from {data_path}: {e}")
        # Try again without skipping row for simple text files
        if data_path.endswith(('.csv', '.txt')):
            try:
                data = np.loadtxt(data_path, delimiter=',')
                if data.ndim != 2 or data.shape[1] != 2:
                    raise ValueError(f"Data must be a 2-column array (X, Y). Got shape {data.shape}")
                logger.info("Loaded curve data after fallback (no skipped row).")
                return data
            except Exception as e2:
                logger.error(f"Fallback loading also failed: {e2}")
                raise ValueError(f"Unsupported file format or invalid data structure in {data_path}.")
        raise

def plot_curve_to_bytes(curve_data: np.ndarray, system_info: dict, title_suffix: str = "") -> bytes:
    """
    Plots a 1D curve and returns the image as bytes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve_data[:, 0], curve_data[:, 1], 'b.', markersize=4)
    
    plot_title = system_info.get("title", "Data")
    ax.set_title(plot_title + title_suffix)
    
    xlabel_text = system_info.get("xlabel", "X-axis")
    ax.set_xlabel(xlabel_text)
    
    ylabel_text = system_info.get("ylabel", "Y-axis")
    ax.set_ylabel(ylabel_text)
    
    ax.grid(True, linestyle='--')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='jpeg', dpi=150)
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close(fig)
    return image_bytes