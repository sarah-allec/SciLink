import json
import os
from PIL import Image
import logging
import numpy as np
from datetime import datetime
from atomai.models import ParticleAnalyzer
from .image_processor import load_image

def run_sam_analysis(
    image_array: np.ndarray, 
    params: dict
) -> dict:
    """
    Runs the atomai.ParticleAnalyzer.analyze() method
    with a given set of parameters.
    """
    # Analyzer is initialized on each call. This is stateless and safe.
    analyzer = ParticleAnalyzer(
        checkpoint_path=params.get('checkpoint_path', None),
        model_type=params.get('model_type', 'vit_h'),
        device=params.get('device', 'auto')
    )
    sam_result = analyzer.analyze(image_array, params=params)
    return sam_result

def visualize_sam_results(
    sam_result: dict
) -> np.ndarray:
    """
    Runs the atomai.ParticleAnalyzer.visualize_particles() method
    to generate an overlay image.
    """
    overlay_image = ParticleAnalyzer.visualize_particles(
        sam_result, 
        show_plot=False, 
        show_labels=True, 
        show_centroids=True
    )
    return overlay_image

def calculate_sam_statistics(
    sam_result: dict,
    image_path: str, # Need original image path for scaling
    preprocessed_image_shape: tuple,
    nm_per_pixel: float | None
) -> dict:
    """
    Calculates a comprehensive dictionary of morphological statistics
    from a raw sam_result.
    """
    logger = logging.getLogger(__name__)
    logger.info("   (Tool Info: Extracting morphological statistics...)")
    particles_df = ParticleAnalyzer.particles_to_dataframe(sam_result)
    
    summary_stats = {}
    current_params = sam_result.get("parameters", {})

    if not particles_df.empty:
        # Calculate scaling based on original image and preprocessed image
        original_shape = load_image(image_path).shape
        pixel_rescaling_factor = preprocessed_image_shape[0] / original_shape[0]
        
        if nm_per_pixel is not None and nm_per_pixel > 0:
            linear_scale_factor = (1 / pixel_rescaling_factor) * nm_per_pixel
            area_scale_factor = linear_scale_factor ** 2
            unit_suffix = "nm"
            area_unit_suffix = "nm_sq"
        else:
            linear_scale_factor = 1 / pixel_rescaling_factor
            area_scale_factor = linear_scale_factor ** 2
            unit_suffix = "pixels"
            area_unit_suffix = "pixels_sq"
            
        summary_stats = {
            'total_particles': sam_result['total_count'],
            f'mean_area_{area_unit_suffix}': float(particles_df['area'].mean()) * area_scale_factor,
            f'std_area_{area_unit_suffix}': float(particles_df['area'].std()) * area_scale_factor,
            f'area_range_{area_unit_suffix}': [float(particles_df['area'].min()) * area_scale_factor, float(particles_df['area'].max()) * area_scale_factor],
            f'area_percentiles_{area_unit_suffix}': [p * area_scale_factor for p in particles_df['area'].quantile([0.25, 0.5, 0.75]).tolist()],
            'mean_circularity': float(particles_df['circularity'].mean()),
            'std_circularity': float(particles_df['circularity'].std()),
            'circularity_range': [float(particles_df['circularity'].min()), float(particles_df['circularity'].max())],
            'mean_aspect_ratio': float(particles_df['aspect_ratio'].mean()),
            'std_aspect_ratio': float(particles_df['aspect_ratio'].std()),
            'aspect_ratio_range': [float(particles_df['aspect_ratio'].min()), float(particles_df['aspect_ratio'].max())],
            'mean_solidity': float(particles_df['solidity'].mean()),
            'std_solidity': float(particles_df['solidity'].std()),
            'solidity_range': [float(particles_df['solidity'].min()), float(particles_df['solidity'].max())],
            f'mean_equiv_diameter_{unit_suffix}': float(particles_df['equiv_diameter'].mean()) * linear_scale_factor,
            f'std_equiv_diameter_{unit_suffix}': float(particles_df['equiv_diameter'].std()) * linear_scale_factor,
            f'equiv_diameter_range_{unit_suffix}': [float(particles_df['equiv_diameter'].min()) * linear_scale_factor, float(particles_df['equiv_diameter'].max()) * linear_scale_factor],
            f'mean_perimeter_{unit_suffix}': float(particles_df['perimeter'].mean()) * linear_scale_factor,
            f'std_perimeter_{unit_suffix}': float(particles_df['perimeter'].std()) * linear_scale_factor,
            f'perimeter_range_{unit_suffix}': [float(particles_df['perimeter'].min()) * linear_scale_factor, float(particles_df['perimeter'].max()) * linear_scale_factor],
            'final_parameters': current_params,
            'physical_scale_nm_per_pixel': nm_per_pixel if nm_per_pixel is not None else "N/A"
        }
    else:
         summary_stats = {
             'total_particles': 0, 
             'final_parameters': current_params,
             'physical_scale_nm_per_pixel': nm_per_pixel if nm_per_pixel is not None else "N/A"
        }
    
    logger.info(f"   (Tool Info: Statistics calculation complete. Final count: {sam_result['total_count']} particles.)")
    return summary_stats

def save_sam_visualization(
    overlay_image: np.ndarray, 
    stage: str, 
    cycle: int, 
    particle_count: int, 
    params: dict,
    logger: logging.Logger
):
    """Save visualization images for each refinement step."""
    try:
        output_dir = "sam_analysis_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stage}_cycle{cycle:02d}_{particle_count}particles_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        Image.fromarray(overlay_image).save(filepath)
        
        params_filename = f"{stage}_cycle{cycle:02d}_params_{timestamp}.txt"
        params_filepath = os.path.join(output_dir, params_filename)
        with open(params_filepath, 'w') as f:
            f.write(f"Stage: {stage}\nCycle: {cycle}\nParticle Count: {particle_count}\n")
            f.write(f"Parameters:\n{json.dumps(params, indent=2)}")
        
        logger.info(f"   (Tool Info: 📸 Saved {stage} visualization: {filename})")
        
    except Exception as e:
        logger.error(f"   (Tool Info: Failed to save visualization: {e})")