import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from atomai.stat import SpectralUnmixer
from .image_processor import create_multi_abundance_overlays

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
        
        unmixer = SpectralUnmixer(
            method=settings.get('method', 'nmf'),
            n_components=n_components,
            normalize=settings.get('normalize', True),
            random_state=42, # for reproducibility
            **{k: v for k, v in settings.items()
               if k not in ['method', 'n_components', 'normalize', 'enabled', 
                            'auto_components', 'min_auto_components', 'max_auto_components',
                            'run_preprocessing', 'output_dir']}
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