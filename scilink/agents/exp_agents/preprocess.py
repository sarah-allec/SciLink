import numpy as np
from scipy.ndimage import median_filter
import logging
import json
from typing import Tuple, Dict, Any

from .base_agent import BaseAnalysisAgent
from .instruct import PRE_PROCESSING_STRATEGY_INSTRUCTIONS

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class HyperspectralPreprocessingAgent(BaseAnalysisAgent):
    """
    An agent that uses an LLM to determine the optimal pre-processing strategy
    for a hyperspectral data cube and then applies it.

    The pipeline involves:
    1. LLM-driven strategy selection (despiking, masking).
    2. Applying the chosen pre-processing steps.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the pre-processing agent."""
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("HyperspectralPreprocessingAgent initialized.")

    def run_preprocessing(self, hspy_data: np.ndarray, system_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the full LLM-guided pre-processing pipeline.

        Args:
            hspy_data (np.ndarray): The raw (h, w, e) hyperspectral data.
            system_info (Dict[str, Any]): Metadata about the sample.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - The processed (cleaned) hyperspectral data.
                - The 2D boolean mask that was applied.
        """
        if hspy_data.ndim != 3:
            self.logger.error(f"Input data must be 3D (h, w, e), but got {hspy_data.ndim}D. Skipping processing.")
            return hspy_data, np.ones(hspy_data.shape[:2], dtype=bool)

        # 1. Ask LLM for the best strategy
        strategy = self._llm_select_preprocessing_strategy(hspy_data, system_info)

        # 2. Apply that strategy
        processed_data, mask_2d = self._apply_preprocessing(hspy_data, strategy)

        return processed_data, mask_2d

    def _llm_select_preprocessing_strategy(self, hspy_data: np.ndarray, system_info: dict) -> dict:
        """
        Asks an LLM to choose the best pre-processing steps based on data stats.
        """
        self.logger.info("\n\n🤖 -------------------- DATA AGENT STEP: PRE-PROCESSING STRATEGY SELECTION -------------------- 🤖\n")
        try:
            # Calculate robust statistics
            data_min = np.min(hspy_data)
            data_max = np.max(hspy_data)
            
            # Use a subset for faster percentile calculation if data is large
            sample_data = hspy_data
            if np.prod(hspy_data.shape) > 1e7: # > 10 million points
                self.logger.debug(f"Data is large ({np.prod(hspy_data.shape)} points), sampling for statistics.")
                indices = np.random.choice(hspy_data.size, 10**7, replace=False)
                sample_data = hspy_data.ravel()[indices]

            prompt_parts = [
                PRE_PROCESSING_STRATEGY_INSTRUCTIONS,
                "\n--- Data Statistics ---",
                f"Data Shape: {hspy_data.shape}",
                f"Data Mean: {np.mean(sample_data):.4e}",
                f"Data Std: {np.std(sample_data):.4e}",
                f"Data Min: {data_min:.4e}",
                f"Data Max: {data_max:.4e}",
                f"1st Percentile: {np.percentile(sample_data, 1):.4e}",
                f"5th Percentile (potential mask region): {np.percentile(sample_data, 5):.4e}",
                f"50th Percentile (Median): {np.percentile(sample_data, 50):.4e}",
                f"99th Percentile: {np.percentile(sample_data, 99):.4e}",
                f"99.9th Percentile (potential spike region): {np.percentile(sample_data, 99.9):.4e}",
            ]
            
            system_info_section = self._build_system_info_prompt_section(system_info)
            if system_info_section:
                prompt_parts.append(system_info_section)

            prompt_parts.append("\n\nProvide your strategy in the requested JSON format.")

            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM pre-processing strategy selection failed: {error_dict}. Using default strategy.")
                raise ValueError("LLM parsing failed")

            self.logger.info(f"LLM Pre-processing Strategy: {result_json.get('reasoning', 'No reasoning provided.')}")
            return result_json

        except Exception as e:
            self.logger.error(f"LLM strategy selection failed: {e}. Falling back to default (clip and mask).")
            # Fallback strategy (Corrected percentile)
            return {
              "apply_despike": False,
              "despike_kernel_size": 3,
              "apply_masking": True,
              "mask_threshold_percentile": 5.0, # Use a low percentile to remove background
              "reasoning": "Fallback: Default clipping and masking strategy."
            }

    def _apply_preprocessing(self, hspy_data: np.ndarray, strategy: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies a robust pre-processing pipeline:
        1. Despiking (Median Filter) to remove "bad" high-intensity outliers.
        2. Clipping to zero to remove "bad" negative noise.
        3. Robust percentile-based masking to remove "bad" low-signal background.
        """
        self.logger.info("\n\n🤖 -------------------- DATA AGENT STEP: APPLYING PRE-PROCESSING -------------------- 🤖\n")
        data_to_process = hspy_data.copy()
        mask_2d = None

        # 1. Apply Despiking (Median Filter) - THIS IS THE TRUE OUTLIER REMOVAL
        if strategy.get('apply_despike', False):
            kernel_size = int(strategy.get('despike_kernel_size', 3))
            # Ensure kernel is 3D for 3D filter
            kernel_tuple = (kernel_size, kernel_size, 1)
            self.logger.info(f"Applying 3D Median Filter (despike) with kernel {kernel_tuple}...")
            data_to_process = median_filter(data_to_process, size=kernel_tuple)
        
        # 2. Clip all negative values to zero
        self.logger.info("Clipping all negative data points to 0.0...")
        np.clip(data_to_process, 0, None, out=data_to_process)

        # 3. Calculate Masking strategy from the NOW CLEANED (despiked & non-negative) data
        if strategy.get('apply_masking', True):
            # Summing is more robust to noise than mean for masking
            total_intensities = np.sum(data_to_process, axis=2) # (h, w)
            
            # Find all "signal" pixels (anything that isn't zero)
            # Use a small epsilon to avoid floating point issues
            signal_pixels = total_intensities[total_intensities > 1e-9]
            
            if signal_pixels.size == 0:
                self.logger.warning("No signal found after despiking and clipping. Aborting mask.")
                mask_2d = np.ones(data_to_process.shape[:2], dtype=bool) # Keep everything
            else:
                # Use the threshold *from the LLM's strategy*
                threshold_percentile = float(strategy.get('mask_threshold_percentile', 5.0))
                self.logger.info(f"Using LLM-defined mask percentile: {threshold_percentile}")

                # Keep everything that is above the Nth percentile of the *signal*.
                # This removes the dimmest N% of the signal, which is likely just noise.
                intensity_threshold = np.percentile(signal_pixels, threshold_percentile)
                
                self.logger.info(f"Calculated robust intensity mask threshold: {intensity_threshold:.4e}")
                mask_2d = total_intensities > intensity_threshold
                
                num_kept = np.sum(mask_2d)
                total_pixels = mask_2d.size
                self.logger.info(f"Mask will *keep* {num_kept} / {total_pixels} pixels ({(num_kept/total_pixels)*100:.1f}%)")

        # 4. Apply Mask (if created)
        if mask_2d is not None:
            self.logger.info("Applying final intensity mask to data...")
            # Use broadcasting to efficiently multiply
            data_to_process *= mask_2d[..., np.newaxis]
        else:
            self.logger.info("No mask was created, returning despiked and clipped data.")
            # If no mask was created (e.g., apply_masking=False), create an all-True mask
            mask_2d = np.ones(data_to_process.shape[:2], dtype=bool)

        return data_to_process, mask_2d