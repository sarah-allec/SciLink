import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from atomai.stat import SlidingFFTNMF

def run_fft_nmf_analysis(
    image_path: str,
    window_size: int,
    n_components: int,
    window_step: int,
    fft_nmf_settings: dict,
    logger: logging.Logger
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Run sliding FFT + NMF analysis using AtomAI.
    This is a standalone tool callable by any agent.
    """
    try:        
        fft_output_dir = fft_nmf_settings.get('output_dir', 'microscopy_analysis')
        os.makedirs(fft_output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        safe_base_name = "".join(c if c.isalnum() else "_" for c in base_name)
        fft_output_base = os.path.join(fft_output_dir, f"{safe_base_name}_output")
        
        analyzer = SlidingFFTNMF(
            window_size_x=window_size if window_size and window_size > 0 else None,
            window_size_y=window_size if window_size and window_size > 0 else None,
            window_step_x=window_step if window_step and window_step > 0 else None,
            window_step_y=window_step if window_step and window_step > 0 else None,
            interpolation_factor=fft_nmf_settings.get('interpolation_factor', 2),
            zoom_factor=fft_nmf_settings.get('zoom_factor', 2),
            hamming_filter=fft_nmf_settings.get('hamming_filter', True),
            components=n_components
        )
        
        # atomai's 'analyze_image' can take the path directly
        components, abundances = analyzer.analyze_image(image_path, output_path=fft_output_base)
        
        logger.info(f"   (Tool Info: FFT-NMF analysis complete. Components: {components.shape})")
        
        _save_fft_nmf_plots(
            components, 
            abundances, 
            image_path, 
            fft_nmf_settings, 
            logger
        )

        return components, abundances
        
    except Exception as fft_e:
        logger.error(f"❌ Tool Failed: AtomAI Sliding FFT + NMF analysis failed: {fft_e}", exc_info=True)
        return None, None

def _save_fft_nmf_plots(
    components: np.ndarray, 
    abundances: np.ndarray, 
    image_path: str, 
    fft_nmf_settings: dict, 
    logger: logging.Logger
):
    """Creates and saves nice plots for each NMF component and its abundance map."""
    try:
        output_dir = fft_nmf_settings.get('visualization_dir', 'fft_nmf_visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        num_components = components.shape[0]
        logger.info(f"   (Tool Info: Creating and saving {num_components} NMF visualization plots...)")

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        safe_base_name = "".join(c if c.isalnum() else "_" for c in base_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for i in range(num_components):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'NMF Result {i+1}/{num_components}', fontsize=16)

            comp_img = np.log1p(components[i]) # Log-scale for viz
            ax1.imshow(comp_img, cmap='inferno')
            ax1.set_title(f'Component {i+1} (FFT Pattern)')
            ax1.axis('off')

            im = ax2.imshow(abundances[i], cmap='inferno')
            ax2.set_title(f'Abundance Map {i+1} (Spatial Location)')
            ax2.axis('off')
            fig.colorbar(im, ax=ax2, label="Abundance", fraction=0.046, pad=0.04)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            plot_filename = f"{safe_base_name}_nmf_plot_{i+1}_{timestamp}.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)

        logger.info(f"   (Tool Info: ✅ Saved NMF visualizations to: {output_dir})")

    except Exception as e:
        logger.error(f"   (Tool Info: Failed to create or save NMF plots: {e})", exc_info=True)