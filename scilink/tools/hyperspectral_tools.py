import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from atomai.stat import SpectralUnmixer
from .image_processor import create_multi_abundance_overlays
import cv2

def run_spectral_unmixing(
    hspy_data: np.ndarray,
    n_components: int,
    settings: dict,
    logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Runs the atomai.stat.SpectralUnmixer tool.
    
    Returns:
        tuple: (components, abundance_maps, reconstruction_error)
    """
    try:
        logger.info(f"  (Tool Info: Running SpectralUnmixer with n_components={n_components})")
        
        tool_kwargs = settings.copy()
        tool_kwargs.pop('n_components', None)
        tool_kwargs.pop('method', None)
        tool_kwargs.pop('normalize', None)

        unmixer = SpectralUnmixer(
            method=tool_kwargs.pop('method', 'nmf'),
            n_components=n_components,
            normalize=tool_kwargs.pop('normalize', True),
            random_state=42 if 'random_state' not in tool_kwargs else tool_kwargs['random_state']
            **tool_kwargs
        )
        
        components, abundance_maps = unmixer.fit(hspy_data)
        error = getattr(unmixer.model, 'reconstruction_err_', 0.0) # Get error if available
        
        return components, abundance_maps, float(error)
        
    except Exception as e:
        logger.error(f"  (Tool Error: Spectral unmixing failed: {e})", exc_info=True)
        raise # Re-raise for the controller to catch

def create_energy_axis(n_channels: int, system_info: dict = None) -> tuple[np.ndarray, str, bool]:
    """
    Create energy axis from system_info if available, otherwise use channel indices.
    """
    if system_info and "energy_range" in system_info:
        energy_info = system_info["energy_range"]
        
        if "start" in energy_info and "end" in energy_info:
            start = energy_info["start"]
            end = energy_info["end"]
            units = energy_info.get("units", "eV")
            
            energy_axis = np.linspace(start, end, n_channels)
            xlabel = f"Energy ({units})"
            has_energy_info = True
            return energy_axis, xlabel, has_energy_info
            
    # Fallback: channel indices
    energy_axis = np.arange(n_channels)
    xlabel = "Channel"
    has_energy_info = False
    return energy_axis, xlabel, has_energy_info

def create_nmf_summary_plot(
    components: np.ndarray, 
    abundance_maps: np.ndarray, 
    n_comp: int, 
    system_info: dict,
    logger: logging.Logger
) -> bytes:
    """
    Create a single summary plot showing all components and abundance maps.
    """
    try:
        n_channels = components.shape[1]
        energy_axis, xlabel, has_energy_info = create_energy_axis(n_channels, system_info)
        
        fig, axes = plt.subplots(2, n_comp, figsize=(n_comp * 3, 6))
        
        if n_comp == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_comp):
            # Top row: Component spectra
            axes[0, i].plot(energy_axis, components[i, :], 'b-', linewidth=1.5)
            axes[0, i].set_title(f'NMF Component {i+1}', fontsize=10)
            axes[0, i].set_xlabel(xlabel)
            if i == 0:
                axes[0, i].set_ylabel('Intensity')
            axes[0, i].grid(True, alpha=0.3)
            
            # Bottom row: Abundance maps
            im = axes[1, i].imshow(abundance_maps[..., i], cmap='seismic', aspect='auto')
            axes[1, i].set_title(f'Abundance Map {i+1}', fontsize=10)
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        title = f'NMF Analysis: {n_comp} Components'
        if has_energy_info:
            title += " (Energy Calibrated)"
        plt.suptitle(title, fontsize=14, y=0.95)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close()
        
        return image_bytes
        
    except Exception as e:
        logger.error(f"  (Tool Error: Failed to create summary plot for {n_comp} components: {e})")
        return None

def create_elbow_plot(component_range: list[int], errors: list[float], logger: logging.Logger) -> bytes | None:
    """Create an elbow plot of reconstruction error vs. number of components."""
    if not component_range or not errors or len(component_range) != len(errors):
        logger.warning("  (Tool Info: Invalid input for creating elbow plot.)")
        return None
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(component_range, errors, 'bo-', markersize=6)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('NMF Reconstruction Error (Frobenius Norm)')
        ax.set_title('NMF Reconstruction Error vs. Number of Components (Elbow Plot)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xticks(component_range)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)
        logger.info("  (Tool Info: Successfully created NMF elbow plot.)")
        return image_bytes
    except Exception as e:
        logger.error(f"  (Tool Error: Failed to create elbow plot: {e})", exc_info=True)
        return None

def create_component_abundance_pairs(
    components: np.ndarray, 
    abundance_maps: np.ndarray, 
    system_info: dict,
    logger: logging.Logger
) -> list[dict]:
    """
    Create individual component-abundance pair images with consistent y-scaling.
    Returns a list of dictionaries: [{'label': str, 'bytes': bytes}, ...]
    """
    pair_images_list = []
    n_components = components.shape[0]
    
    try:
        n_channels = components.shape[1]
        energy_axis, xlabel, has_energy_info = create_energy_axis(n_channels, system_info)
        
        global_min = np.min(components)
        global_max = np.max(components)
        y_margin = (global_max - global_min) * 0.05
        y_limits = (global_min - y_margin, global_max + y_margin)
        
        logger.info(f"  (Tool Info: Creating {n_components} component-abundance pairs with y-scale: {y_limits})")
        
        for i in range(n_components):
            fig, (ax_spectrum, ax_abundance) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Left plot: Component spectrum
            ax_spectrum.plot(energy_axis, components[i, :], 'b-', linewidth=2)
            ax_spectrum.set_ylim(y_limits)
            ax_spectrum.set_xlabel(xlabel)
            ax_spectrum.set_ylabel('Intensity')
            ax_spectrum.set_title(f'Component {i+1} Spectrum')
            ax_spectrum.grid(True, alpha=0.3)
            
            # Right plot: Abundance map
            im = ax_abundance.imshow(abundance_maps[..., i], cmap='viridis', aspect='equal')
            ax_abundance.set_title(f'Component {i+1} Abundance Map')
            ax_abundance.axis('off')
            plt.colorbar(im, ax=ax_abundance, fraction=0.046, pad=0.04, label='Abundance')
            
            fig.suptitle(f'Component {i+1} Analysis', fontsize=12, y=0.98)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
            buf.seek(0)
            pair_images_list.append({
                "label": f"Component {i+1} Pair (Spectrum + Abundance Map)",
                "bytes": buf.getvalue()
            })
            plt.close()
            
        return pair_images_list
        
    except Exception as e:
        logger.error(f"  (Tool Error: Failed to create component-abundance pairs: {e})")
        return []

def create_structure_overlays(
    structure_img_gray: np.ndarray,
    abundance_maps: np.ndarray,
    logger: logging.Logger
) -> bytes | None:
    """
    Wrapper for create_multi_abundance_overlays.
    """
    try:
        logger.info(f"  (Tool Info: Creating abundance overlays for {abundance_maps.shape[2]} components)")
        
        overlay_bytes = create_multi_abundance_overlays(
            structure_image=structure_img_gray,
            abundance_maps=abundance_maps,
            threshold_percentile=85.0, # Show top 15%
            alpha=0.5,
            use_simple_colors=True
        )
        return overlay_bytes
    except Exception as e:
        logger.warning(f"  (Tool Warning: Failed to create abundance overlays: {e})")
        return None


def apply_spatial_mask(
    current_hspy_data: np.ndarray, 
    abundance_maps: np.ndarray, 
    component_index: int, 
    percentile: float = 85.0
) -> np.ndarray:
    """
    Masks hyperspectral data based on an abundance map.
    Uses the *current* iteration's data as the base.
    """
    if abundance_maps is None:
        raise ValueError("Abundance maps are None, cannot apply spatial mask.")
    
    mask_map = abundance_maps[..., component_index]
    
    if mask_map.ndim != 2:
        raise ValueError(f"Abundance map must be 2D, but got shape {mask_map.shape}")
        
    # Resize mask map to match data if needed
    if mask_map.shape != current_hspy_data.shape[:2]:
        # Use cv2.resize, ensuring cv2 is imported
        mask_map = cv2.resize(mask_map, (current_hspy_data.shape[1], current_hspy_data.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # Threshold non-zero pixels to find the mask
    positive_pixels = mask_map[mask_map > 1e-6]
    if positive_pixels.size == 0:
        # No positive pixels found, return original data
        return current_hspy_data 

    threshold_val = np.percentile(positive_pixels, percentile)
    mask_2d = mask_map >= threshold_val
    
    if np.sum(mask_2d) == 0:
        # Mask is empty, return original data
        return current_hspy_data 

    # Apply mask
    masked_data = current_hspy_data.copy()
    masked_data[~mask_2d] = 0 # Zero out pixels *not* in the mask
    return masked_data

def apply_spectral_slice(
    original_hspy_data: np.ndarray, 
    system_info: dict, 
    energy_range: list
) -> tuple[np.ndarray, dict]:
    """
    Slices hyperspectral data based on an energy range.
    Uses the *original* data as the base and returns an updated system_info.
    """
    energy_axis, _, has_info = create_energy_axis(original_hspy_data.shape[2], system_info)
    if not has_info:
        raise ValueError("Cannot apply spectral slice: No energy axis information found in metadata.")
        
    if energy_range is None or len(energy_range) != 2:
        raise ValueError(f"Invalid energy_range: {energy_range}")
        
    start_e, end_e = min(energy_range), max(energy_range)
    
    slice_indices = np.where((energy_axis >= start_e) & (energy_axis <= end_e))[0]
    
    if len(slice_indices) == 0:
        raise ValueError(f"No data found in energy range {energy_range}.")
        
    sliced_data = original_hspy_data[..., slice_indices]
    
    # We must also update the system_info to reflect this slice
    new_system_info = system_info.copy()
    new_system_info["energy_range"] = {
        "start": float(energy_axis[slice_indices[0]]),
        "end": float(energy_axis[slice_indices[-1]]),
        "units": system_info.get("energy_range", {}).get("units", "unknown")
    }
    
    return sliced_data, new_system_info


def compare_component_with_weighted_raw(
    hspy_data: np.ndarray, 
    component_spectrum: np.ndarray, 
    abundance_map: np.ndarray, 
    component_idx: int,
    logger
) -> bytes:
    """
    Calculates the Abundance-Weighted Average Spectrum of the raw data
    and plots it against the NMF component for validation.
    """
    import matplotlib.pyplot as plt
    from io import BytesIO
    
    try:
        h, w, e = hspy_data.shape
        
        # Flatten
        flat_data = hspy_data.reshape(-1, e) 
        flat_abundance = abundance_map.ravel() 
        
        # Weighted Average
        total_weight = np.sum(flat_abundance)
        if total_weight < 1e-10:
            return None

        weighted_raw_spectrum = np.dot(flat_abundance, flat_data) / total_weight
        
        # Scale NMF to match raw data max
        scale_factor = np.max(weighted_raw_spectrum) / (np.max(component_spectrum) + 1e-6)
        scaled_nmf = component_spectrum * scale_factor

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(weighted_raw_spectrum, color='black', linewidth=2, alpha=0.8, label='Raw Data (Weighted Avg)')
        ax.plot(scaled_nmf, color='red', linestyle='--', linewidth=1.5, label=f'NMF Comp {component_idx+1} (Model)')
        
        residual = weighted_raw_spectrum - scaled_nmf
        ax.fill_between(range(len(residual)), residual, 0, color='gray', alpha=0.2, label='Residual')

        ax.set_title(f"Validation: Component {component_idx+1} vs. Raw Data")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Failed to create weighted comparison plot: {e}")
        return None
    

def create_validated_component_pair(
    hspy_data: np.ndarray, 
    component_spectrum: np.ndarray, 
    abundance_map: np.ndarray, 
    component_idx: int,
    logger
) -> bytes:
    """
    Generates a combined visualization for Refinement Steps:
    [Left]: Abundance Map (Spatial distribution)
    [Right]: Validation Plot (Abundance-Weighted Raw Data vs. NMF Model)
    """
    try:
        h, w, e = hspy_data.shape
        
        # --- 1. Calculate Weighted Average (Validation Logic) ---
        # Reshape for dot product: (N_pixels, n_channels)
        flat_data = hspy_data.reshape(-1, e)
        # Weights: (N_pixels,)
        flat_abundance = abundance_map.ravel()
        
        total_weight = np.sum(flat_abundance)
        
        # Safety check: if component is empty/dead
        if total_weight < 1e-10:
            logger.warning(f"Component {component_idx}: Abundance map empty, skipping plot.")
            return None

        # Weighted Mean: (Weights @ Data) / Sum(Weights)
        weighted_raw_spectrum = np.dot(flat_abundance, flat_data) / total_weight
        
        # --- 2. Scaling for Visual Comparison ---
        # NMF intensity is arbitrary. We scale the NMF curve (Model) to match 
        # the max intensity of the Raw Data (Ground Truth).
        scale_factor = np.max(weighted_raw_spectrum) / (np.max(component_spectrum) + 1e-6)
        scaled_nmf = component_spectrum * scale_factor

        # --- 3. Setup Plot (1 Row, 2 Cols) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- Left Plot: Spatial Abundance Map ---
        im = ax1.imshow(abundance_map, cmap='viridis')
        ax1.set_title(f"Component {component_idx+1} Distribution", fontsize=12, fontweight='bold')
        ax1.axis('off')
        # Add colorbar specifically to this axis
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # --- Right Plot: Spectral Validation ---
        # Ground Truth (Black)
        ax2.plot(weighted_raw_spectrum, color='black', linewidth=2, alpha=0.8, label='Raw Data (Weighted Avg)')
        # Model (Red Dashed)
        ax2.plot(scaled_nmf, color='red', linestyle='--', linewidth=1.5, label='NMF Model')
        
        # Residual (Gray Fill) - Shows artifacts
        residual = weighted_raw_spectrum - scaled_nmf
        ax2.fill_between(range(len(residual)), residual, 0, color='gray', alpha=0.2, label='Residual')

        ax2.set_title("Validation: Model vs Ground Truth", fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Spectral Channel")
        ax2.set_ylabel("Intensity")
        
        plt.tight_layout()
        
        # Save to memory
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Failed to create validated pair for component {component_idx}: {e}")
        return None