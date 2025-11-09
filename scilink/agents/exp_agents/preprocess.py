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
    and deterministically calculates data quality.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the pre-processing agent."""
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("HyperspectralPreprocessingAgent initialized.")
        
    def _calculate_statistics(self, hspy_data: np.ndarray) -> Dict[str, Any]:
        """Calculates robust statistics for the LLM and for deterministic SNR."""
        self.logger.debug("Calculating robust statistics...")
        
        # Use a subset for faster percentile calculation if data is large
        sample_data = hspy_data
        if np.prod(hspy_data.shape) > 1e7: # > 10 million points
            self.logger.debug(f"Data is large ({np.prod(hspy_data.shape)} points), sampling for statistics.")
            indices = np.random.choice(hspy_data.size, 10**7, replace=False)
            sample_data = hspy_data.ravel()[indices]

        stats = {
            "shape": hspy_data.shape,
            "mean": np.mean(sample_data),
            "std": np.std(sample_data),
            "min": np.min(hspy_data),
            "max": np.max(hspy_data),
            "p1": np.percentile(sample_data, 1),
            "p5": np.percentile(sample_data, 5),
            "p50": np.percentile(sample_data, 50),
            "p99": np.percentile(sample_data, 99),
            "p99_9": np.percentile(sample_data, 99.9)
        }
        return stats

    def _calculate_snr(self, stats: Dict[str, Any]) -> Tuple[float, str]:
        """
        Calculates a robust, deterministic SNR from the pre-calculated stats.
        
        Returns:
            Tuple[float, str]: (snr_estimate, reasoning_string)
        """
        self.logger.debug("Calculating deterministic SNR...")
        
        # A robust heuristic for SNR, especially for sparse (mostly zero) data.
        # It measures the "signal range" (99th percentile) against the
        # "noise" or "background" level (the median, or p50).
        
        signal = stats["p99"]
        noise = stats["p50"]
        
        # Avoid division by zero if median is 0
        if noise > 1e-9:
            snr = (signal - noise) / noise
            reasoning = f"Calculated as (P99 - P50) / P50: ({signal:.2e} - {noise:.2e}) / {noise:.2e}"
        else:
            # Fallback if median is zero: use std dev (less ideal but won't crash)
            if stats["std"] > 1e-9:
                snr = signal / stats["std"]
                reasoning = f"Calculated as P99 / Std (P50 was zero): {signal:.2e} / {stats['std']:.2e}"
            else:
                snr = 0.0 # No signal or no variance
                reasoning = "SNR is 0.0 (no variance or signal)"

        # Cap SNR at a reasonable max to avoid infinity/huge numbers
        snr = min(max(snr, 0.0), 1000.0) 
        
        return float(snr), reasoning


    def run_preprocessing(self, hspy_data: np.ndarray, system_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Runs the full LLM-guided pre-processing pipeline.
        """
        if hspy_data.ndim != 3:
            self.logger.error(f"Input data must be 3D (h, w, e), but got {hspy_data.ndim}D. Skipping processing.")
            default_mask = np.ones(hspy_data.shape[:2], dtype=bool)
            default_quality = {
                "snr_estimate": 0.0,
                "reasoning": "Processing skipped: Input data was not 3D."
            }
            return hspy_data, default_mask, default_quality

        # 1. Calculate robust statistics ONCE
        stats = self._calculate_statistics(hspy_data)
        
        # 2. Get cleaning strategy from LLM (Task 1)
        strategy = self._llm_select_preprocessing_strategy(stats, system_info)
        
        # 3. Deterministically calculate SNR (Task 2)
        snr_value, snr_reasoning = self._calculate_snr(stats)
        
        # 4. Construct the final data_quality object
        data_quality = {
            "snr_estimate": snr_value,
            "reasoning": snr_reasoning
        }
        self.logger.info(f"Deterministic Data Quality: SNR = {snr_value:.2f} ({snr_reasoning})")

        # 5. Apply the LLM's cleaning strategy
        processed_data, mask_2d = self._apply_preprocessing(hspy_data, strategy)

        # 6. Return all results
        return processed_data, mask_2d, data_quality

    def _llm_select_preprocessing_strategy(self, stats: Dict[str, Any], system_info: dict) -> dict:
        """
        Asks an LLM to choose the best pre-processing steps based on data stats.
        Returns ONLY the strategy dictionary.
        """
        self.logger.info("\n\n🤖 -------------------- DATA AGENT STEP: PRE-PROCESSING STRATEGY SELECTION -------------------- 🤖\n")
        try:
            prompt_parts = [
                PRE_PROCESSING_STRATEGY_INSTRUCTIONS,
                "\n--- Data Statistics ---",
                f"Data Shape: {stats['shape']}",
                f"Data Mean: {stats['mean']:.4e}",
                f"Data Std: {stats['std']:.4e}",
                f"Data Min: {stats['min']:.4e}",
                f"Data Max: {stats['max']:.4e}",
                f"1st Percentile: {stats['p1']:.4e}",
                f"5th Percentile: {stats['p5']:.4e}",
                f"50th Percentile (Median): {stats['p50']:.4e}",
                f"99th Percentile: {stats['p99']:.4e}",
                f"99.9th Percentile: {stats['p99_9']:.4e}",
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
            return result_json # This is the strategy dict

        except Exception as e:
            self.logger.error(f"LLM strategy selection failed: {e}. Falling back to default (clip and mask).")
            # Fallback strategy
            return {
              "apply_despike": False,
              "despike_kernel_size": 3,
              "apply_masking": True,
              "mask_threshold_percentile": 5.0,
              "reasoning": "Fallback: Default clipping and masking strategy."
            }

    def _apply_preprocessing(self, hspy_data: np.ndarray, strategy: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies a robust pre-processing pipeline:
        (No changes needed here)
        """
        self.logger.info("\n\n🤖 -------------------- DATA AGENT STEP: APPLYING PRE-PROCESSING -------------------- 🤖\n")
        data_to_process = hspy_data.copy()
        mask_2d = None

        # 1. Apply Despiking
        if strategy.get('apply_despike', False):
            kernel_size = int(strategy.get('despike_kernel_size', 3))
            kernel_tuple = (kernel_size, kernel_size, 1)
            self.logger.info(f"Applying 3D Median Filter (despike) with kernel {kernel_tuple}...")
            data_to_process = median_filter(data_to_process, size=kernel_tuple)
        
        # 2. Clip all negative values to zero
        # This is a hard-coded safety step, independent of LLM strategy
        self.logger.info("Clipping all negative data points to 0.0...")
        np.clip(data_to_process, 0, None, out=data_to_process)

        # 3. Calculate Masking strategy
        if strategy.get('apply_masking', False):
            total_intensities = np.sum(data_to_process, axis=2) # (h, w)
            signal_pixels = total_intensities[total_intensities > 1e-9]
            
            if signal_pixels.size == 0:
                self.logger.warning("No signal found after despiking and clipping. Aborting mask.")
                mask_2d = np.ones(data_to_process.shape[:2], dtype=bool)
            else:
                threshold_percentile = float(strategy.get('mask_threshold_percentile', 5.0))
                self.logger.info(f"Using LLM-defined mask percentile: {threshold_percentile}")

                intensity_threshold = np.percentile(signal_pixels, threshold_percentile)
                
                self.logger.info(f"Calculated robust intensity mask threshold: {intensity_threshold:.4e}")
                mask_2d = total_intensities > intensity_threshold
                
                num_kept = np.sum(mask_2d)
                total_pixels = mask_2d.size
                self.logger.info(f"Mask will *keep* {num_kept} / {total_pixels} pixels ({(num_kept/total_pixels)*100:.1f}%)")

        # 4. Apply Mask
        if mask_2d is not None:
            self.logger.info("Applying final intensity mask to data...")
            data_to_process *= mask_2d[..., np.newaxis]
        else:
            self.logger.info("No mask was created, returning despiked and clipped data.")
            mask_2d = np.ones(data_to_process.shape[:2], dtype=bool)

        return data_to_process, mask_2d