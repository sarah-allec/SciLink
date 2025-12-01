import logging
import numpy as np
import json
import os
import re
from datetime import datetime
import base64
import cv2
from typing import Callable
from google.generativeai.types import GenerationConfig

import traceback
import matplotlib.pyplot as plt
from io import BytesIO

from ....tools import hyperspectral_tools as tools
from ....tools.image_processor import load_image
from ..preprocess import HyperspectralPreprocessingAgent
from ..instruct import (
    COMPONENT_INITIAL_ESTIMATION_INSTRUCTIONS,
    COMPONENT_SELECTION_WITH_ELBOW_INSTRUCTIONS,
    SPECTROSCOPY_REFINEMENT_INSTRUCTIONS,
    SPECTROSCOPY_HOLISTIC_SYNTHESIS_INSTRUCTIONS,
    SPECTROSCOPY_REFLECTION_INSTRUCTIONS,
    SPECTROSCOPY_REFLECTION_UPDATE_INSTRUCTIONS
)

AGENT_METADATA_KEYS_TO_STRIP = [
    'enable_human_feedback', 
    'run_preprocessing', 
    'output_dir', 
    'visualization_dir', 
    
    'enabled', 
    'auto_components', 
    'min_auto_components', 
    'max_auto_components',
    
    # Other potential non-tool keys
]

def _sanitize_filename(text: str) -> str:
    """Helper to create safe filenames from labels."""
    # Replace spaces with underscores, remove non-alphanumeric chars except _ and -
    safe_text = re.sub(r'[^\w\-\_]', '', text.replace(" ", "_"))
    return safe_text

def _create_grid_from_images(image_bytes_list: list, logger: logging.Logger) -> bytes:
    """
    Stitches a list of JPEG bytes into a single grid image using OpenCV.
    Used to create a 'Validated Summary Grid' from individual validated plots.
    """
    if not image_bytes_list:
        return None
        
    try:
        # Decode all images
        images = []
        for b in image_bytes_list:
            nparr = np.frombuffer(b, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        
        if not images:
            return None

        n_imgs = len(images)
        
        # If only one, return it directly (re-encoded)
        if n_imgs == 1:
            return image_bytes_list[0]

        # Determine grid size (target ~2 columns)
        cols = 2
        rows = (n_imgs + cols - 1) // cols
        
        # Find max dimensions to standardize cells
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        
        # Create blank canvas
        grid_h = rows * max_h
        grid_w = cols * max_w
        grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8) + 255 # White background
        
        for idx, img in enumerate(images):
            r = idx // cols
            c = idx % cols
            
            # Resize current img to fit cell if needed (maintain aspect ratio logic could go here, 
            # but usually plots are uniform size. We'll center it.)
            h, w = img.shape[:2]
            
            y_offset = r * max_h
            x_offset = c * max_w
            
            # Simple copy (top-left alignment for simplicity, or center)
            grid_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
            
        # Encode back to jpeg
        retval, buf = cv2.imencode('.jpg', grid_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    except Exception as e:
        logger.warning(f"Failed to stitch validation grid: {e}")
        return None


class RunPreprocessingController:
    """
    [🛠️ Tool Step]
    Runs the HyperspectralPreprocessingAgent.
    """
    def __init__(self, logger: logging.Logger, preprocessor: HyperspectralPreprocessingAgent):
        self.logger = logger
        self.preprocessor = preprocessor

    def execute(self, state: dict) -> dict:
        self.logger.info("\n\n🛠️ --- CALLING TOOL: PREPROCESSING AGENT --- 🛠️\n")
        if not self.preprocessor:
            self.logger.warning("Preprocessing skipped: agent not initialized.")
            state["data_quality"] = {"reasoning": "Preprocessing skipped: agent not initialized."}
            return state

        # Check the runtime flag set by the agent
        if not state.get("settings", {}).get("run_preprocessing", True):
            self.logger.info("Preprocessing skipped for this refinement iteration (run_preprocessing=False).")
            self.logger.info("Calculating statistics on *current* masked data for the next step...")
            
            try:
                # We still need stats (like SNR and shape) for the *next* controller
                stats = self.preprocessor._calculate_statistics(state["hspy_data"])
                snr_value, snr_reasoning = self.preprocessor._calculate_snr(stats)
                state["data_quality"] = {
                    "snr_estimate": snr_value,
                    "reasoning": f"SNR of *current iteration* data: {snr_reasoning}"
                }
                state["preprocessing_mask"] = np.ones(state["hspy_data"].shape[:2], dtype=bool)
                self.logger.info(f"✅ Tool Complete: Statistics calculated. SNR = {snr_value:.2f}")
                return state 
            except Exception as e:
                self.logger.error(f"❌ Tool Failed: Stat calculation on refinement data failed: {e}", exc_info=True)
                state["error_dict"] = {"error": "Stat calculation on refinement data failed", "details": str(e)}
                return state

        try:
            processed_data, mask, data_quality = self.preprocessor.run_preprocessing(
                state["hspy_data"], 
                state["system_info"]
            )
            state["hspy_data"] = processed_data 
            state["preprocessing_mask"] = mask
            state["data_quality"] = data_quality
            self.logger.info("✅ Tool Complete: Full preprocessing finished.")
        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Preprocessing failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Preprocessing failed", "details": str(e)}
        return state

class GetInitialComponentParamsController:
    """
    [🧠 LLM Step]
    Asks LLM for initial n_components.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.instructions = COMPONENT_INITIAL_ESTIMATION_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🧠 --- LLM STEP: ESTIMATE INITIAL N_COMPONENTS --- 🧠\n")
        
        h, w, e = state["hspy_data"].shape
        data_quality = state.get("data_quality", {})
        
        prompt_parts = [self.instructions]
        prompt_parts.append(f"\n\n--- Hyperspectral Data Information ---")
        prompt_parts.append(f"Data dimensions: {h}x{w} spatial pixels, {e} spectral channels")
        prompt_parts.append(f"\n--- Data Quality Assessment (from Preprocessor) ---")
        prompt_parts.append(f"- Robust SNR Estimate: {data_quality.get('snr_estimate', 'N/A')}")
        prompt_parts.append(f"- Assessment: {data_quality.get('reasoning', 'N/A')}")
        
        if state.get("system_info"):
            sys_info_str = json.dumps(state["system_info"], indent=2)
            prompt_parts.append(f"\n\n--- System Information ---\n{sys_info_str}")
        
        prompt_parts.append("\n\nBased on the system description and data characteristics, estimate the optimal number of spectral components.")
        
        param_gen_config = GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM initial estimation failed: {error_dict}. Using default.")
                n_components = 4 
            else:
                n_components = result_json.get('estimated_components', 4)
                reasoning = result_json.get('reasoning', 'No reasoning provided.')
                self.logger.info(f"LLM initial estimate: {n_components} components. Reasoning: {reasoning}")
                
                print("\n" + "="*80)
                print("🧠 LLM REASONING (GetInitialComponentParamsController)")
                print(f"  Suggested n_components: {n_components}")
                print(f"  Explanation: {reasoning}")
                print("="*80 + "\n")
                
                if not (isinstance(n_components, int) and 2 <= n_components <= 15):
                    self.logger.warning(f"Invalid LLM estimate {n_components}, using default 4.")
                    n_components = 4
                    
            state["initial_n_components"] = n_components
            self.logger.info(f"✅ LLM Step Complete: Initial component estimate = {n_components}.")

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: Initial component estimation: {e}", exc_info=True)
            state["initial_n_components"] = 4 
            
        return state

class RunComponentTestLoopController:
    """
    [🛠️ Tool Step]
    Loops from min to max components, runs spectral unmixing.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: COMPONENT TEST LOOP --- 🛠️\n")

        tool_settings = self.settings.copy()
        for key in AGENT_METADATA_KEYS_TO_STRIP:
            tool_settings.pop(key, None)
        
        initial_estimate = state.get("initial_n_components", 4)
        min_c = self.settings.get('min_auto_components', 2)
        max_c = self.settings.get('max_auto_components', min(initial_estimate + 4, 12))
        component_range = list(range(min_c, max_c + 1))
        
        errors = []
        visual_examples = [] 
        
        for n_comp in component_range:
            try:
                components, abundance_maps, error = tools.run_spectral_unmixing(
                    state["hspy_data"], n_comp, tool_settings, self.logger
                )
                errors.append(error)
                self.logger.info(f"  (Loop {n_comp}/{max_c}): Error = {error:.4f}")

                if n_comp == min_c or n_comp == max_c or n_comp == initial_estimate:
                    summary_bytes = tools.create_nmf_summary_plot(
                        components, abundance_maps, n_comp, state["system_info"], self.logger
                    )
                    if summary_bytes:
                        visual_examples.append({
                            'n_components': n_comp,
                            'image': summary_bytes,
                            'label': f"{n_comp} Components ({'Min' if n_comp==min_c else 'Max' if n_comp==max_c else 'Initial Estimate'})"
                        })
                        
                        try:
                            output_dir = self.settings.get('output_dir', 'spectroscopy_output')
                            os.makedirs(output_dir, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            iter_title = _sanitize_filename(state.get('iteration_title', 'iter'))
                            filename = f"{iter_title}_TestLoop_{n_comp}comp_{timestamp}.jpeg"
                            filepath = os.path.join(output_dir, filename)
                            
                            with open(filepath, 'wb') as f:
                                f.write(summary_bytes)
                            self.logger.info(f"📸 Saved component test plot to: {filepath}")
                        except Exception as e:
                            self.logger.warning(f"Failed to save component test plot: {e}")
            except Exception as e:
                self.logger.warning(f"  (Loop {n_comp}/{max_c}): Failed. {e}")
                errors.append(np.inf)
        
        state["component_test_range"] = component_range
        state["component_test_errors"] = errors
        state["component_test_visuals"] = visual_examples
        self.logger.info("✅ Tool Complete: Component test loop finished.")
        return state

class CreateElbowPlotController:
    """
    [🛠️ Tool Step]
    Generates the elbow plot.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: CREATE ELBOW PLOT --- 🛠️\n")
        
        plot_bytes = tools.create_elbow_plot(
            state["component_test_range"],
            state["component_test_errors"],
            self.logger
        )
        state["elbow_plot_bytes"] = plot_bytes
        if plot_bytes:
            self.logger.info("✅ Tool Complete: Elbow plot created.")
            try:
                output_dir = self.settings.get('output_dir', 'spectroscopy_output')
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                iter_title = _sanitize_filename(state.get('iteration_title', 'iter'))
                filename = f"{iter_title}_Elbow_Plot_{timestamp}.jpeg"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(plot_bytes)
                self.logger.info(f"📸 Saved elbow plot to: {filepath}")
            except Exception as e:
                self.logger.warning(f"Failed to save elbow plot: {e}")
        else:
            self.logger.warning("Tool Warning: Elbow plot creation failed.")
        return state

class GetFinalComponentSelectionController:
    """
    [🧠 LLM Step]
    Asks LLM to pick the best n_components.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.instructions = COMPONENT_SELECTION_WITH_ELBOW_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🧠 --- LLM STEP: SELECT FINAL N_COMPONENTS --- 🧠\n")
        
        initial_estimate = state.get("initial_n_components", 4)
        component_range = state.get("component_test_range", [])
        
        if not state.get("elbow_plot_bytes") or not state.get("component_test_visuals"):
            self.logger.warning("Missing elbow plot or visual examples. Using initial estimate.")
            state["final_n_components"] = initial_estimate
            return state

        prompt_parts = [self.instructions]
        prompt_parts.append(f"\n\n--- Context ---")
        prompt_parts.append(f"Initial LLM estimate: {initial_estimate} components")
        prompt_parts.append(f"Tested component range: {component_range}")
        
        prompt_parts.append(f"\n\n--- Quantitative Analysis: Reconstruction Error ---")
        prompt_parts.append("Elbow Plot (Error vs. Number of Components):")
        prompt_parts.append({"mime_type": "image/jpeg", "data": state["elbow_plot_bytes"]})
        
        prompt_parts.append(f"\n\n--- Qualitative Analysis: Visual Examples ---")
        for viz in state.get("component_test_visuals", []):
            prompt_parts.append(f"\n\n**{viz['label']}:**")
            prompt_parts.append({"mime_type": "image/jpeg", "data": viz['image']})

        prompt_parts.append(f"\n\nBased on the elbow plot AND the visual examples, decide the optimal number of components.")
        
        param_gen_config = GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM final selection failed: {error_dict}. Using initial estimate.")
                final_n_components = initial_estimate
            else:
                final_n_components = result_json.get('final_components', initial_estimate)
                reasoning = result_json.get('reasoning', 'No reasoning provided.')
                self.logger.info(f"LLM final decision: {final_n_components} components. Reasoning: {reasoning}")

                print("\n" + "="*80)
                print("🧠 LLM REASONING (GetFinalComponentSelectionController)")
                print(f"  Final n_components: {final_n_components}")
                print(f"  Explanation: {reasoning}")
                print("="*80 + "\n")

                if not (isinstance(final_n_components, int) and final_n_components in component_range):
                    self.logger.warning(f"Invalid LLM final choice {final_n_components}, using initial estimate.")
                    final_n_components = initial_estimate
            
            state["final_n_components"] = final_n_components
            self.logger.info(f"✅ LLM Step Complete: Final component selection = {final_n_components}.")

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: Final component selection: {e}", exc_info=True)
            state["final_n_components"] = initial_estimate 
            
        return state

class RunFinalSpectralUnmixingController:
    """
    [🛠️ Tool Step]
    Runs spectral unmixing one last time.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: FINAL SPECTRAL UNMIXING --- 🛠️\n")
        
        final_n_components = state.get("final_n_components")
        if not final_n_components:
            final_n_components = self.settings.get('n_components', 4)
            self.logger.warning(f"Auto-selection failed. Using fixed component count: {final_n_components}")
            state["final_n_components"] = final_n_components

        tool_settings = self.settings.copy()
        for key in AGENT_METADATA_KEYS_TO_STRIP:
            tool_settings.pop(key, None)
            
        try:
            components, abundance_maps, error = tools.run_spectral_unmixing(
                state["hspy_data"], final_n_components, tool_settings, self.logger
            )
            state["final_components"] = components
            state["final_abundance_maps"] = abundance_maps
            state["final_reconstruction_error"] = error
            self.logger.info(f"✅ Tool Complete: Final unmixing done. Error: {error:.4f}")
        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Final unmixing: {e}", exc_info=True)
            state["error_dict"] = {"error": "Final spectral unmixing failed", "details": str(e)}
        return state

class CreateAnalysisPlotsController:
    """
    [🛠️ Tool Step]
    Generates high-quality visualization pairs for the Agent.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: CREATE ANALYSIS PLOTS --- 🛠️\n")
        
        components = state.get("final_components")
        abundance_maps = state.get("final_abundance_maps")
        
        iter_title_raw = state.get("iteration_title", "Global_Analysis")
        iter_prefix = _sanitize_filename(iter_title_raw)

        if components is None or abundance_maps is None:
            self.logger.warning("Skipping plot creation: final components/maps not found.")
            return state

        output_dir = self.settings.get('output_dir', 'spectroscopy_output')
        
        final_plots = []
        validated_bytes_list = [] 
        
        # --- 1. Generate Validated Pairs ---
        self.logger.info(f"Generating Validated Analysis Plots for {components.shape[0]} components...")
        
        for i in range(components.shape[0]):
            plot_bytes = tools.create_validated_component_pair(
                state["hspy_data"], 
                components[i], 
                abundance_maps[..., i], 
                i, 
                state["system_info"],
                self.logger
            )
            
            if plot_bytes:
                label = f"Component {i+1} Analysis"
                final_plots.append({'label': label, 'bytes': plot_bytes})
                validated_bytes_list.append(plot_bytes)
                
                # Save using tool
                label_safe = _sanitize_filename(label)
                tools.save_image_bytes(
                    plot_bytes, output_dir, 
                    f"{iter_prefix}_{label_safe}.jpeg", self.logger
                )

        state["component_pair_plots"] = final_plots
        for plot in final_plots:
            state["analysis_images"].append({"label": plot['label'], "data": plot['bytes']})

        # --- 2. Create Summary Grid ---
        try:
            self.logger.info("  (Tool Info: Stitching validated plots into Summary Grid...)")
            summary_bytes = tools.create_image_grid(validated_bytes_list, self.logger)

            if summary_bytes:
                label = "NMF Summary Grid"
                tools.save_image_bytes(
                    summary_bytes, output_dir, 
                    f"{iter_prefix}_{_sanitize_filename(label)}.jpeg", self.logger
                )
                
                state["analysis_images"].append({"label": label, "data": summary_bytes})

        except Exception as e:
            self.logger.warning(f"Failed to create/save NMF summary plot: {e}")

        # --- 3. Structure Overlays ---
        if state.get("structure_image_path"):
            try:
                # Load image (Controller logic)
                structure_img = load_image(state["structure_image_path"])
                if structure_img.ndim == 3:
                    structure_img = cv2.cvtColor(structure_img, cv2.COLOR_RGB2GRAY)
                
                # Create (Tool logic)
                overlay_bytes = tools.create_multi_abundance_overlays(
                    structure_img, abundance_maps, threshold_percentile=85.0 
                )
                state["structure_overlay_bytes"] = overlay_bytes
                
                if overlay_bytes:
                    label = "Structure-Abundance Overlays"
                    tools.save_image_bytes(
                        overlay_bytes, output_dir, 
                        f"{iter_prefix}_{_sanitize_filename(label)}.jpeg", self.logger
                    )
                    state["analysis_images"].append({"label": label, "data": overlay_bytes})
                
            except Exception as e:
                self.logger.warning(f"Failed to create structure overlays: {e}")

        self.logger.info("✅ Tool Complete: Final analysis plots created and saved.")
        return state    

class BuildHyperspectralPromptController:
    """
    [📝 Prep Step]
    Assembles all results into the final prompt for interpretation.
    THIS IS FOR A SINGLE ITERATION, NOT THE FINAL SYNTHESIS.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n📝 --- PREP STEP: BUILDING FINAL PROMPT --- 📝\n")
        
        # 1. Base Instruction & Context
        prompt_parts = [state["instruction_prompt"]]

        if state.get("parent_refinement_reasoning"):
            prompt_parts.append("\n\n### 🔍 CONTEXT: Why are we performing this focused analysis?")
            prompt_parts.append(f"**Reasoning from previous step:** \"{state['parent_refinement_reasoning']}\"")
            prompt_parts.append("Use this context to guide your interpretation.")
        
        # 2. Data Metadata
        h, w, e = state["hspy_data"].shape
        _, energy_xlabel, _ = tools.create_energy_axis(e, state["system_info"])
        
        prompt_parts.append(f"\n\nHyperspectral Data Information:")
        prompt_parts.append(f"- Data shape: ({h}, {w}, {e})")
        prompt_parts.append(f"- X-axis: {energy_xlabel}")
        
        if state.get("final_components") is not None:
            prompt_parts.append(f"- Spectral unmixing method: {state['settings'].get('method', 'nmf').upper()}")
            prompt_parts.append(f"- Number of components: {state['final_n_components']}")
            prompt_parts.append(f"- Final Reconstruction Error: {state.get('final_reconstruction_error', 'N/A'):.4f}")

        # 3. Component Analysis (Dynamic Instructions)
        current_depth = state.get("current_depth", 0)
        
        if state.get("component_pair_plots"):
            prompt_parts.append("\n\n**Spectral Component Analysis:**")
            
            if current_depth == 0:
                # Standard Instructions for Depth 0
                prompt_parts.append("Below are the NMF components extracted from the global dataset.")
                prompt_parts.append("For each component, the LEFT image is the Spectral Signature and the RIGHT image is the Spatial Abundance.")
            else:
                # Validation Instructions for Depth > 0
                prompt_parts.append("### 🧪 Quantitative Validation Mode (Split-Panel Analysis)")
                prompt_parts.append("Because this is a focused refinement, the plots are more detailed to help you detect artifacts.")
                prompt_parts.append("Each figure contains:")
                
                prompt_parts.append("\n**1. LEFT: Spatial Distribution**")
                prompt_parts.append("- Shows where this component is located physically.")
                
                prompt_parts.append("\n**2. RIGHT (TOP PANEL): Spectral Fit & Variance**")
                prompt_parts.append("- **Black Line (Mean):** The abundance-weighted average spectrum of the raw data (Ground Truth).")
                prompt_parts.append("- **Red Dashed Line (Model):** The NMF component (Mathematical Model).")
                prompt_parts.append("- **Blue Shaded Band:** The Weighted Standard Deviation ($\pm 1\sigma$). This represents the natural heterogeneity of the data in this region.")
                
                prompt_parts.append("\n**3. RIGHT (BOTTOM PANEL): Residuals**")
                prompt_parts.append("- **Gray Area:** The difference between the Data and the Model.")

                prompt_parts.append("\n### ⚠️ CRITICAL INTERPRETATION RULES")
                prompt_parts.append("Use the **Blue Band** to distinguish between 'Messy Data' and 'Bad Model':")
                prompt_parts.append("1. **Valid Fit:** If the Red Line (Model) stays mostly **INSIDE** the Blue Band, the component is valid, even if it doesn't match the Black Line perfectly. The mismatch is just natural variation.")
                prompt_parts.append("2. **Hallucination (Artifact):** If the Red Line creates a peak that goes significantly **OUTSIDE** the Blue Band (and the Black Line is flat there), NMF has 'invented' a feature. **Reject this feature.**")
                prompt_parts.append("3. **Missed Physics:** If the Bottom Panel (Residual) shows a large, structured peak, NMF failed to capture a real physical/chemical feature present in the data.")

            # Append the plots
            for plot in state["component_pair_plots"]:
                prompt_parts.append(f"\n{plot['label']}:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": plot['bytes']})

        # 4. Structure Overlays (if available)
        if state.get("structure_overlay_bytes"):
            prompt_parts.append("\n\n**Structure-Abundance Correlation Analysis:**")
            prompt_parts.append("Overlays showing where components are concentrated on the structural image.")
            prompt_parts.append({"mime_type": "image/jpeg", "data": state["structure_overlay_bytes"]})
            
            # Ensure storage for synthesis
            found = False
            for img in state.get("analysis_images", []):
                if img.get("label") == "Structure-Abundance Overlays": found = True
            if not found:
                state["analysis_images"].append({
                    "label": "Structure-Abundance Overlays",
                    "data": state["structure_overlay_bytes"]
                })

        # 5. System Metadata & Formatting
        if state.get("system_info"):
            sys_info_str = json.dumps(state["system_info"], indent=2)
            prompt_parts.append(f"\n\nAdditional System Information (Metadata):\n{sys_info_str}")

        prompt_parts.append("\n\nProvide your analysis in the requested JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        self.logger.info("✅ Prep Step Complete: Final prompt is ready.")
        return state


class SelectRefinementTargetController:
    """
    [🧠 LLM Step]
    Asks the LLM if a refinement (zoom-in) is needed and where.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.instructions = SPECTROSCOPY_REFINEMENT_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🧠 --- LLM STEP: SELECT REFINEMENT TARGET --- 🧠\n")

        prompt_parts = [self.instructions]
        prompt_parts.append(f"\n\n--- Current Analysis: {state.get('iteration_title', 'Analysis')} ---")
        
        # Add system info
        if state.get("system_info"):
            sys_info_str = json.dumps(state["system_info"], indent=2)
            prompt_parts.append(f"\n\n--- System Information ---\n{sys_info_str}")

        # Add plots from the current iteration
        prompt_parts.append("\n\n--- Analysis Results ---")
        analysis_images = state.get("analysis_images", [])
        if not analysis_images:
            self.logger.warning("No analysis images found for refinement selection.")
            prompt_parts.append("(No visual results available)")
        
        for img in analysis_images:
            # Robustly get bytes
            image_bytes = img.get('data') or img.get('bytes') 
            if image_bytes:
                prompt_parts.append(f"\n{img['label']}:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
            else:
                self.logger.warning(f"Could not find image bytes for plot: {img.get('label')}")

        prompt_parts.append("\n\nBased on these results, decide if a focused refinement is needed.")

        param_gen_config = GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"LLM refinement selection failed: {error_dict}. Stopping loop.")
                state["refinement_decision"] = {"refinement_needed": False, "reasoning": "LLM selection failed."}
                return state

            # Get Raw Targets
            raw_targets = result_json.get("targets", [])
            is_needed = result_json.get("refinement_needed", False)

            # Priority Filtering (Custom Code vs Standard)
            custom_code_targets = [t for t in raw_targets if t.get('type') == 'custom_code']
            standard_targets = [t for t in raw_targets if t.get('type') != 'custom_code']
            
            final_targets = []
            requires_custom_code = False
            
            if custom_code_targets:
                # Winner-Takes-All: If code is needed, focus ONLY on that.
                # We pick the first custom target and ignore standard zooms for this turn.
                top_target = custom_code_targets[0] 
                self.logger.info(f"🎯 Priority Target Selected (Custom Code): {top_target.get('description')}")
                final_targets = [top_target]
                requires_custom_code = True
            else:
                # Otherwise, proceed with standard targets
                final_targets = standard_targets
                requires_custom_code = False

            # Store the final decision with the filtered targets and the FLAG
            state["refinement_decision"] = {
                "refinement_needed": is_needed,
                "reasoning": result_json.get("reasoning", "No reasoning provided."),
                "targets": final_targets,                
                "requires_custom_code": requires_custom_code 
            }

            self.logger.info(f"✅ LLM Step Complete: Refinement decision: {state['refinement_decision']['reasoning']}")
            
            print("\n" + "="*80)
            print("🧠 LLM REASONING (SelectRefinementTargetController)")
            print(f"  Refinement Needed: {is_needed}")
            print(f"  Custom Code Triggered: {requires_custom_code}")
            print(f"  Explanation: {state['refinement_decision']['reasoning']}")
            print(f"  Targets Found: {len(final_targets)}")
            if final_targets:
                for i, t in enumerate(final_targets):
                    print(f"    Target {i+1} ({t.get('type')}): {t.get('description')}")
            print("="*80 + "\n")

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: Refinement selection: {e}", exc_info=True)
            state["refinement_decision"] = {"refinement_needed": False, "reasoning": f"Exception: {e}"}
            
        return state
    

class GenerateRefinementTasksController:
    """
    [🛠️ Tool Step]
    Takes the list of targets from the LLM and generates new tasks.
    """
    MIN_SPECTRAL_CHANNELS = 10 

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        self.logger.info("\n\n🛠️ --- CALLING TOOL: GENERATE REFINEMENT TASKS --- 🛠️\n")
        
        decision = state.get("refinement_decision")
        if not decision or not decision.get("refinement_needed") or not decision.get("targets"):
            state["new_tasks"] = []
            return state

        new_tasks = []
        current_depth = state.get("current_depth", 0)
        next_depth = current_depth + 1
        
        # Iterate through targets with an index (1-based for humans)
        for i, target in enumerate(decision["targets"], 1):
            try:
                t_type = target.get("type")
                t_value = target.get("value")
                t_desc = target.get("description", "refinement")
                
                # Create the Structured ID
                # D = Depth, T = Target Number
                short_title = f"Focused_Analysis_D{next_depth}_T{i}"
                
                new_data = None
                new_sys_info = state["system_info"]

                if t_type == "spatial":
                    self.logger.info(f"Processing Spatial Task {i}: {t_desc}")
                    component_index = int(t_value)
                    if component_index > 0: component_index -= 1 
                    else: component_index = 0
                    
                    new_data = tools.apply_spatial_mask(
                        state["hspy_data"], state["final_abundance_maps"], component_index
                    )

                elif t_type == "spectral":
                    self.logger.info(f"Processing Spectral Task {i}: {t_desc}")
                    
                    # t_value comes from LLM as physical units, e.g., [0.6, 1.0]
                    target_start_ev, target_end_ev = t_value[0], t_value[1]
                    
                    # 1. Reconstruct the full energy axis for the ORIGINAL data
                    h, w, e = state["original_hspy_data"].shape
                    # (Assuming you have access to create_energy_axis from tools)
                    energy_axis, _, _ = tools.create_energy_axis(e, state["system_info"])
                    
                    # 2. Find the closest integer indices for these physical values
                    start_idx, end_idx = tools.convert_energy_to_indices(
                        energy_axis, 
                        target_start_ev, 
                        target_end_ev, 
                        min_channels=self.MIN_SPECTRAL_CHANNELS
                    )

                    # 3. Perform the slicing using INDICES
                    # Note: We pass the indices to the tool, NOT the physical values
                    # Use the calculated safe indices to get safe physical range for the tool
                    safe_physical_range = [energy_axis[start_idx], energy_axis[end_idx]]

                    new_data, _ = tools.apply_spectral_slice(
                        state["original_hspy_data"], 
                        state["system_info"], 
                        safe_physical_range
                    )

                    # 4. Update System Info with the NEW Physical Range
                    new_sys_info = state["system_info"].copy()
                    new_sys_info['energy_range'] = {
                        'start': float(energy_axis[start_idx]),
                        'end': float(energy_axis[end_idx]),
                        'units': state["system_info"].get('energy_range', {}).get('units', 'units')
                    }
                    
                    self.logger.info(f"Recalibrated axis: {state['system_info']['energy_range']['start']:.2f}-{state['system_info']['energy_range']['end']:.2f} -> {new_sys_info['energy_range']['start']:.2f}-{new_sys_info['energy_range']['end']:.2f}")

                    if new_data.shape[-1] < self.MIN_SPECTRAL_CHANNELS:
                        self.logger.warning(f"Skipping spectral task '{short_title}': too few channels.")
                        continue

                if new_data is not None:
                    task = {
                        "data": new_data,
                        "system_info": new_sys_info,
                        # SHORT TITLE for Reports/Files
                        "title": short_title, 
                        # LONG DESCRIPTION for LLM Context (Re-injected later)
                        "parent_reasoning": t_desc, 
                        "source_depth": next_depth
                    }
                    new_tasks.append(task)

            except Exception as e:
                self.logger.error(f"Failed to generate task for target {target}: {e}")

        state["new_tasks"] = new_tasks
        self.logger.info(f"✅ Generated {len(new_tasks)} new analysis tasks.")
        return state
    

class BuildHolisticSynthesisPromptController:
    """
    [📝 Prep Step]
    Assembles ALL iteration results into the final prompt for synthesis.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.instructions = SPECTROSCOPY_HOLISTIC_SYNTHESIS_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n📝 --- PREP STEP: BUILDING FINAL SYNTHESIS PROMPT --- 📝\n")
        
        prompt_parts = [self.instructions]
        
        all_results = state.get("all_iteration_results", [])
        if not all_results:
            self.logger.error("No iteration results found to synthesize.")
            state["error_dict"] = {"error": "No iteration results found for synthesis."}
            return state

        # 1. System Info
        if state.get("system_info"):
            sys_info_str = json.dumps(state["system_info"], indent=2)
            prompt_parts.append(f"\n\n--- System Information ---\n{sys_info_str}")

        # 2. Build Context for Each Iteration
        all_images = []

        for i, iter_result in enumerate(all_results):
            raw_title = iter_result.get('iteration_title', f'Iteration_{i}')
            iter_ref_id = _sanitize_filename(raw_title)
            
            prompt_parts.append(f"\n\n### SECTION {i+1}: {raw_title}")
            
            # Context: Why did we do this?
            context_desc = iter_result.get('parent_refinement_reasoning') 
            if context_desc:
                prompt_parts.append(f"**Target Description:** \"{context_desc}\"")

            # --- DYNAMIC ANALYSIS INJECTION
            # Retrieve the list of features generated by the custom code
            custom_meta_list = iter_result.get("custom_analysis_metadata_list")
            
            if custom_meta_list:
                prompt_parts.append(f"\n**🔍 DYNAMIC ANALYSIS FINDINGS (Physics-Based Mapping):**")
                prompt_parts.append("The following features were mathematically modeled using custom Python code:")
                
                # Loop through every feature in the list
                for idx, meta in enumerate(custom_meta_list, 1):
                    name = meta.get('name', 'Custom Feature')
                    desc = meta.get('description', 'N/A')
                    units = meta.get('units', 'a.u.')
                    stats = meta.get('stats', {})
                    
                    prompt_parts.append(f"\n   **Feature {idx}: {name}**")
                    prompt_parts.append(f"   - Physical Interpretation: {desc}")
                    prompt_parts.append(f"   - Units: {units}")
                    
                    # Crash Fix: Use .get(key, 0.0) to handle missing stats gracefully
                    if stats:
                        s_min = stats.get('min', 0.0)
                        s_max = stats.get('max', 0.0)
                        s_mean = stats.get('mean', 0.0)
                        prompt_parts.append(f"   - Statistics: Min {s_min:.2f}, Max {s_max:.2f}, Mean {s_mean:.2f}")
                
                prompt_parts.append("\n-> **INSTRUCTION:** Use these specific physical maps to validate or correct the NMF results.")
            
            # Text Summary (Standard NMF Analysis)
            iter_analysis = iter_result.get('iteration_analysis_text')
            if iter_analysis:
                prompt_parts.append(f"\n**Previous NMF Analysis Summary:**\n{iter_analysis}")
            
            # Visual Evidence
            iter_images = iter_result.get('analysis_images', [])
            if iter_images:
                prompt_parts.append(f"\n**Visual Evidence for {raw_title}:**")
                for img in iter_images:
                    image_bytes = img.get('data') or img.get('bytes')
                    raw_label = img.get('label', 'Unknown_Plot')
                    
                    if image_bytes:
                        # Create a unique semantic ID for citation
                        unique_ref = f"[{iter_ref_id}] {raw_label}"
                        
                        prompt_parts.append(f"\n**{unique_ref}**")
                        prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
                        
                        # Update label in the image object itself for the Report Generation step
                        # (This ensures the HTML report filters correctly)
                        img['label'] = unique_ref 
                        all_images.append(img)

        # 3. EXPLICIT REPORTING INSTRUCTIONS
        prompt_parts.append("\n\n### 📝 CRITICAL REPORTING INSTRUCTIONS")
        prompt_parts.append("1. **AT THE END of your 'detailed_analysis' text**, you MUST append a section titled **'### Key Evidence'**.")
        prompt_parts.append("2. In that section, you MUST list the supporting figures using their **EXACT bolded titles** provided above.")
        prompt_parts.append("\n**Required Format for Evidence Section:**")
        prompt_parts.append("### Key Evidence")
        prompt_parts.append("- **[Exact_ID_From_Above] Image Title**: Explanation of evidence.")

        prompt_parts.append("\n\nProvide your final, synthesized analysis in the requested JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        state["analysis_images"] = all_images 
        
        self.logger.info("✅ Prep Step Complete: Final synthesis prompt is ready.")
        return state
    

class GenerateHTMLReportController:
    """
    [🛠️ Tool Step]
    Generates a beautiful, human-readable HTML report.
    
    - Citation-Based Filtering: Scans the 'detailed_analysis' text. 
      Only displays images that the LLM explicitly referenced by name.
    - Fallback: If the LLM references nothing, falls back to 'Smart Filtering' 
      (showing Grids and hiding redundant components) to ensure the report isn't empty.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def _image_to_base64(self, image_bytes: bytes) -> str:
        """Helper to convert bytes to base64 string for HTML embedding."""
        return base64.b64encode(image_bytes).decode('utf-8')

    def _filter_by_citations(self, text: str, all_images: list) -> list:
        """
        Selects images based on 'Concept Triggers' rather than strict string matching.
        If the text discusses a scientific method (e.g., NMF), the relevant summary plots are forced to display.
        """
        cited_images = []
        lower_text = text.lower()
        
        for img in all_images:
            raw_label = img.get('label', '')
            label_lower = raw_label.lower()
            
            # --- 1. Exact & Direct Match ---
            if raw_label in text:
                cited_images.append(img)
                continue
            
            # Check for label without the [ID] prefix
            # e.g. Label: "[Global_Analysis] NMF Summary Grid" -> Match: "NMF Summary Grid"
            clean_name = re.sub(r'\[.*?\]', '', label_lower).strip()
            if clean_name and clean_name in lower_text:
                cited_images.append(img)
                continue

            # --- 2. Concept Triggers (The Safety Net) ---
            
            # TRIGGER: NMF / Spectral Unmixing
            # If the plot is an NMF Grid and the text mentions "NMF" or "Components", show it.
            if "nmf summary grid" in label_lower:
                if "nmf" in lower_text or "component" in lower_text or "unmixing" in lower_text:
                    cited_images.append(img)
                    continue

            # TRIGGER: Custom / Dynamic Analysis
            # If the plot is a Custom Analysis, check if the specific feature name (e.g. "Peak Center") is mentioned.
            if "custom analysis" in label_lower and ":" in label_lower:
                # Extract feature name: "[ID] Custom Analysis: Peak Center" -> "peak center"
                try:
                    feature_name = label_lower.split(":", 1)[1].strip()
                    if feature_name and feature_name in lower_text:
                        cited_images.append(img)
                        continue
                except IndexError:
                    pass

            # TRIGGER: Structure / Morphology
            # If the plot is a Structure Overlay and text mentions Structure/Correlation, show it.
            if "structure" in label_lower and "overlay" in label_lower:
                if "structure" in lower_text or "morphology" in lower_text or "correlation" in lower_text:
                    cited_images.append(img)
                    continue

            # --- 3. Iteration Context Match ---
            # If the text explicitly names an iteration (e.g. "Global Analysis"), 
            # ensure the main summary grid for that iteration is shown.
            match = re.match(r"\[(.*?)\]", raw_label)
            if match:
                iter_id_clean = match.group(1).replace("_", " ").lower() # e.g. "global analysis"
                if iter_id_clean in lower_text and ("grid" in label_lower or "custom" in label_lower):
                    cited_images.append(img)
                    continue

        # --- Deduplicate ---
        unique_images = []
        seen = set()
        for img in cited_images:
            if img['label'] not in seen:
                unique_images.append(img)
                seen.add(img['label'])

        # --- 4. Final Fail-Safe ---
        # If the filter returned <= 1 image, force the Global NMF Summary to appear 
        # to ensure the report always has context.
        if len(unique_images) <= 1:
            for img in all_images:
                if "global" in img['label'].lower() and "nmf summary" in img['label'].lower():
                    if img['label'] not in seen:
                        unique_images.insert(0, img) # Insert at top
                        seen.add(img['label'])

        return unique_images

    def _filter_redundant_heuristic(self, all_images: list) -> list:
        """
        Backup Strategy: If LLM fails to cite images, use logic to pick the best ones.
        Hides individual components if a Grid exists.
        """
        iterations_with_grid = set()
        for img in all_images:
            label = img.get('label', '')
            if "NMF Summary Grid" in label:
                match = re.match(r"\[(.*?)\]", label)
                if match: iterations_with_grid.add(match.group(1))

        filtered_images = []
        for img in all_images:
            label = img.get('label', '')
            match = re.match(r"\[(.*?)\]", label)
            if match and match.group(1) in iterations_with_grid:
                if "Component" in label and "Analysis" in label:
                    continue # Skip component if grid exists
            filtered_images.append(img)
        return filtered_images

    def execute(self, state: dict) -> dict:
        self.logger.info("\n\n📄 --- TOOL STEP: GENERATING HTML REPORT --- 📄\n")
        
        result_json = state.get("result_json")
        if not result_json:
            self.logger.warning("Skipping report generation: No result_json found.")
            return state

        # Extract Data
        detailed_analysis = result_json.get("detailed_analysis", "No analysis provided.")
        scientific_claims = result_json.get("scientific_claims", [])
        system_info = state.get("system_info", {})
        all_images = state.get("analysis_images", [])
        
        # --- SELECTION LOGIC ---
        # 1. Try Strict Citation
        display_images = self._filter_by_citations(detailed_analysis, all_images)
        selection_method = "Strict Text Citation"

        # 2. Fallback to Heuristic if strict failed (LLM didn't follow instructions)
        if not display_images:
            self.logger.warning("LLM did not explicitly cite any images. Falling back to heuristic filter.")
            display_images = self._filter_redundant_heuristic(all_images)
            selection_method = "Heuristic (Backup)"

        self.logger.info(f"Report Generation: Selected {len(display_images)} images using method: {selection_method}")

        # Output Setup
        output_dir = self.settings.get('output_dir', 'spectroscopy_output')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Hyperspectral_Report_{file_timestamp}.html"
        filepath = os.path.join(output_dir, filename)

        # --- HTML CONSTRUCTION ---
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Hyperspectral Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }}
                .container {{ background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 30px; }}
                h3 {{ color: #16a085; }}
                .metadata-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #bdc3c7; margin-bottom: 20px; }}
                .analysis-text {{ white-space: pre-wrap; background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; }}
                .claim-card {{ background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin-bottom: 15px; }}
                .claim-title {{ font-weight: bold; font-size: 1.1em; color: #0e6655; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 25px; margin-top: 20px; }}
                .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                .image-card img {{ max-width: 100%; height: auto; border-radius: 3px; cursor: pointer; transition: transform 0.2s; }}
                .image-card img:hover {{ transform: scale(1.01); }}
                .image-label {{ margin-top: 12px; font-weight: bold; color: #444; font-size: 1em; border-top: 1px solid #eee; padding-top: 10px; }}
                .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔬 Hyperspectral Analysis Report</h1>
                <div class="metadata-box">
                    <p><strong>Date:</strong> {timestamp}</p>
                    <p><strong>Data Source:</strong> {state.get('image_path', 'N/A')}</p>
                    <p><strong>System Info:</strong> {json.dumps(system_info)}</p>
                </div>

                <h2>1. Synthesized Scientific Analysis</h2>
                <div class="analysis-text">{detailed_analysis}</div>

                <h2>2. Key Evidence (Visual Gallery)</h2>
                <p>These figures are explicitly cited in the analysis above.</p>
                <div class="image-grid">
        """

        for img in display_images:
            label = img.get('label', 'Unknown Figure')
            data = img.get('data') or img.get('bytes')
            
            if data:
                b64_str = self._image_to_base64(data)
                safe_id = _sanitize_filename(label)
                
                html_content += f"""
                    <div class="image-card" id="{safe_id}">
                        <img src="data:image/jpeg;base64,{b64_str}" alt="{label}" loading="lazy">
                        <div class="image-label">{label}</div>
                    </div>
                """

        html_content += """
                </div>
                <h2>3. Key Scientific Claims</h2>
        """

        if not scientific_claims:
            html_content += "<p>No specific claims generated.</p>"
        else:
            for i, claim in enumerate(scientific_claims, 1):
                html_content += f"""
                <div class="claim-card">
                    <div class="claim-title">Claim {i}: {claim.get('claim', 'N/A')}</div>
                    <p><strong>Impact:</strong> {claim.get('scientific_impact', 'N/A')}</p>
                    <p><strong>Research Question:</strong> <em>{claim.get('has_anyone_question', 'N/A')}</em></p>
                </div>
                """

        html_content += """
                <div class="footer">
                    Generated by SciLink Hyperspectral Analysis Agent
                </div>
            </div>
        </body>
        </html>
        """

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"✅ REPORT GENERATED: {filepath}")
            if "result_paths" not in state: state["result_paths"] = []
            state["result_paths"].append(filepath)
        except Exception as e:
            self.logger.error(f"❌ Failed to write HTML report: {e}")

        return state
    

class RunDynamicAnalysisController:
    """
    [🧠 + 💻] The 'Code Interpreter' / 'Dynamic Analyst'.
    """
    MAX_RETRIES = 5

    def __init__(self, model, logger, generation_config, safety_settings, parse_fn):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn

    def execute(self, state: dict) -> dict:
        decision = state.get("refinement_decision", {})
        targets = decision.get("targets", [])
        
        # Filter strictly for custom code requests
        custom_targets = [t for t in targets if t.get('type') == 'custom_code']
        
        # Gatekeeping: If no code requested, skip
        if not custom_targets and not decision.get("requires_custom_code", False):
            return state

        self.logger.info(f"\n\n💻 --- DYNAMIC ANALYSIS: PROCESSING {len(custom_targets)} TASKS --- 💻\n")

        # --- SETUP OUTPUT PATHS ---
        output_dir = state.get("settings", {}).get("output_dir", "spectroscopy_output")
        os.makedirs(output_dir, exist_ok=True)
        
        iter_title = _sanitize_filename(state.get("iteration_title", "iter"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- PREPARE DATA CONTEXT ---
        h, w, e = state["hspy_data"].shape
        
        # Axis & Unit Detection
        sys_info = state.get("system_info", {})
        axis_units = "unknown units"
        
        if "energy_axis" not in state:
            if sys_info.get("energy_range"):
                start = sys_info["energy_range"]["start"]
                end = sys_info["energy_range"]["end"]
                state["energy_axis"] = np.linspace(start, end, e)
                axis_units = sys_info["energy_range"].get("units", "arbitrary units")
            else:
                state["energy_axis"] = np.arange(e)
                axis_units = "channels"
        else:
            # Try to grab units if existing
            axis_units = sys_info.get("energy_range", {}).get("units", "arbitrary units")

        self.logger.info(f"Data Axis Units detected as: {axis_units}")

        # Master containers for ALL scripts run in this session
        all_valid_maps = []
        all_valid_meta = []

        # --- MAIN LOOP: Process each target description separately ---
        for i, target in enumerate(custom_targets, 1):
            target_desc = target.get("description", "Analyze feature")
            self.logger.info(f"👉 Task {i}/{len(custom_targets)}: {target_desc}")

            # 1. Define Prompt for this specific task
            base_prompt = f"""
            You are a Python Data Scientist specialized in Spectroscopy. 
            The standard NMF tool failed to model a spectral feature described as: "{target_desc}".
            
            Your task: Write a Python function to mathematically model this feature. 
            Since complex features often require multiple parameters (e.g., Peak Position AND Peak Width), your function must be able to return MULTIPLE maps.

            ### 1. DATA CONTEXT
            - Input Data `hspy_data`: Shape ({h}, {w}, {e}) (Numpy array)
            - X-Axis `axis`: Shape ({e},) (Numpy array). **Units: {axis_units}**

            ### 2. EXECUTION ENVIRONMENT (STRICT)
            Your code will run in a restricted `exec()` sandbox. 
            
            **PRE-IMPORTED LIBRARIES (Available Globally):**
            - `np`: The full NumPy library.
            - `scipy`: The top-level SciPy module.
            - `sklearn`: The top-level Scikit-Learn module.
            
            **PRE-IMPORTED FUNCTIONS (Direct Shortcuts):**
            - `curve_fit`, `nnls` (from scipy.optimize)
            - `linregress` (from scipy.stats)
            - `find_peaks` (from scipy.signal)
            - `gaussian_filter` (from scipy.ndimage)

            ### 3. CODING CONSTRAINTS
            1. **NO External Imports:** Do not import `os`, `sys`, `matplotlib`, or `warnings`. The sandbox does not support them.
            2. **SciPy Submodules:** If you need a specific SciPy submodule that is NOT in the shortcuts list (e.g., `scipy.interpolate` or `scipy.integrate`), you MUST write `import scipy.interpolate` **inside** your function definition before using it.
            3. **Standard Math:** Use `np.exp`, `np.log`, etc., instead of the `math` library.
            4. **Return Format:** You must return a dictionary, not a print statement or a plot.

            ### 4. YOUR GOAL
            Write a function `analyze_feature(data, axis)` that:
            1. Reshapes data to (pixels, energy).
            2. Implements the specific math required.
            3. Returns a DICTIONARY containing the results.

            ### REQUIRED RETURN FORMAT
            The function must return a Python dictionary with this structure:
            {{
                "maps": {{
                    "Feature_Name_1": np.ndarray,  # 2D map ({h}, {w})
                    "Feature_Name_2": np.ndarray   # 2D map ({h}, {w}) (Optional, if multiple features exist)
                }},
                
                "units": {{                 
                "Feature_Name_1": "nm",
                "Feature_Name_2": "a.u."
                }},    
                "description": str
            }}

            ### RESPONSE FORMAT
            Return a JSON object with:
            - "code": The valid Python code string containing the function `analyze_feature`.
            - "explanation": Brief physics logic.
            """

            current_prompt = base_prompt
            retries = 0
            task_success = False

            while retries < self.MAX_RETRIES:
                try:
                    # A. Generate Code
                    self.logger.info(f"    (Attempt {retries+1}) Asking LLM to write code...")
                    response = self.model.generate_content(current_prompt, generation_config=self.generation_config)
                    result_json, _ = self._parse_llm_response(response)
                    code_str = result_json.get("code", "")
                    
                    # B. Sandbox Execution setup (Reset for each task)
                    local_scope = {}
                    global_scope = {
                        "np": np,
                        "scipy": __import__("scipy"),
                        "sklearn": __import__("sklearn"),
                        "curve_fit": __import__("scipy.optimize", fromlist=["curve_fit"]).curve_fit,
                        "nnls": __import__("scipy.optimize", fromlist=["nnls"]).nnls,
                        "linregress": __import__("scipy.stats", fromlist=["linregress"]).linregress,
                        "find_peaks": __import__("scipy.signal", fromlist=["find_peaks"]).find_peaks,
                        "gaussian_filter": __import__("scipy.ndimage", fromlist=["gaussian_filter"]).gaussian_filter
                    }
                    
                    # Execute Code
                    exec(code_str, global_scope, local_scope)
                    
                    if "analyze_feature" not in local_scope:
                        raise ValueError("Function 'analyze_feature' was not found in generated code.")
                    
                    # C. Run Analysis on Real Data
                    self.logger.info("    Executing generated code...")
                    func = local_scope["analyze_feature"]
                    result_dict = func(state["hspy_data"], state["energy_axis"])
                    
                    # D. Code Output Validation
                    if not isinstance(result_dict, dict): raise ValueError("Function return must be a dict.")
                    
                    maps_dict = result_dict.get("maps")
                    if not maps_dict or not isinstance(maps_dict, dict):
                        raise ValueError("Return dict must contain a 'maps' key with a dictionary of 2D arrays.")

                    # --- SAVE THE SCRIPT ---
                    safe_task_name = _sanitize_filename(target_desc)[:30] # Limit length
                    script_filename = f"{iter_title}_T{i}_{safe_task_name}_{timestamp}.py"
                    script_path = os.path.join(output_dir, script_filename)
                    try:
                        with open(script_path, "w", encoding="utf-8") as f:
                            f.write(f"# Auto-generated Dynamic Analysis Script\n")
                            f.write(f"# Task: {target_desc}\n")
                            f.write(f"# Timestamp: {timestamp}\n\n")
                            f.write(code_str)
                        self.logger.info(f"    💾 Saved script to: {script_filename}")
                    except Exception as e:
                        self.logger.warning(f"    Failed to save script file: {e}")

                    # Iterate through returned maps from THIS script
                    any_map_valid = False
                    raw_units = result_dict.get("units", "a.u.")
                    desc = result_dict.get("description", "")

                    for feature_name, result_map in maps_dict.items():
                        # Shape Check
                        if result_map.shape != (h, w): 
                            self.logger.warning(f"    Skipping {feature_name}: Shape mismatch {result_map.shape}")
                            continue
                        
                        if np.all(np.isnan(result_map)):
                            self.logger.warning(f"    Skipping {feature_name}: Map contains only NaNs.")
                            continue

                        # --- 1. DETERMINE UNITS & NAMES FIRST (Fixes UnboundLocalError) ---
                        current_unit = "a.u."
                        if isinstance(raw_units, dict):
                            current_unit = raw_units.get(feature_name, "a.u.")
                        elif isinstance(raw_units, str):
                            current_unit = raw_units

                        safe_feat = _sanitize_filename(feature_name)

                        # --- 2. GENERATE DASHBOARD ---
                        dashboard_bytes = tools.create_feature_dashboard(result_map, feature_name, current_unit)

                        if dashboard_bytes:
                            # --- 3. VISUAL QC (On the Dashboard) ---
                            self.logger.info(f"    👀 Performing Visual QC on {feature_name}...")
                            qc_result, qc_critique = self._check_result_visually(dashboard_bytes, f"{target_desc} ({feature_name})")
                            
                            if not qc_result:
                                self.logger.warning(f"    ❌ QC Failed: {qc_critique}")
                                # Trigger retry by raising error
                                raise ValueError(f"Visual QC rejected the result. Critique: {qc_critique}")

                            # --- 4. SAVE & STORE (Only if QC Passed) ---
                            filename = f"{iter_title}_T{i}_{safe_feat}_Dashboard_{timestamp}.jpeg"
                            tools.save_image_bytes(dashboard_bytes, output_dir, filename, self.logger)

                            # Add to Agent Memory
                            if "analysis_images" not in state: state["analysis_images"] = []
                            state["analysis_images"].append({
                                "label": f"Custom Analysis: {feature_name}", 
                                "data": dashboard_bytes
                            })

                            # Collect Valid Data
                            all_valid_maps.append(result_map)
                            all_valid_meta.append({
                                "name": feature_name,
                                "units": current_unit,
                                "description": desc,
                                "stats": {
                                    "min": float(np.nanmin(result_map)), 
                                    "max": float(np.nanmax(result_map)),
                                    "mean": float(np.nanmean(result_map))
                                }
                            })
                            any_map_valid = True

                    if not any_map_valid:
                        raise ValueError("No valid maps generated from this script (QC or Shape failures).")
                    
                    task_success = True
                    break # Break retry loop on success

                except Exception as e:
                    error_msg = traceback.format_exc()
                    if "Visual QC" in str(e): error_msg = str(e) # Keep QC message clean
                    self.logger.warning(f"    ❌ Code/QC failed for Task {i}: {str(e)}")
                    retries += 1
                    current_prompt = base_prompt + f"\n\n### ❌ PREVIOUS CODE FAILED\nCritique:\n```text\n{error_msg}\n```\nFix the logic/math to address this critique and regenerate JSON."

            if not task_success:
                self.logger.error(f"    ⚠️ Task {i} failed after {self.MAX_RETRIES} attempts.")

        # --- FINAL AGGREGATION ---
        if not all_valid_maps:
            self.logger.warning("⚠️ All dynamic analysis tasks failed.")
            state["dynamic_analysis_failed"] = True
            return state

        # Stack maps from ALL scripts into one 3D array (H, W, N)
        state["final_abundance_maps"] = np.stack(all_valid_maps, axis=-1)
        state["custom_analysis_metadata_list"] = all_valid_meta
        state["method_used"] = "Dynamic Code Generation"
        state["new_tasks"] = [] 

        self.logger.info(f"✅ Dynamic Analysis Complete. Total unique maps generated: {len(all_valid_maps)}")
        return state

    def _check_result_visually(self, dashboard_bytes: bytes, feature_desc: str) -> tuple[bool, str]:
        """
        Judge the Dashboard (Map + Histogram)
        """
        check_prompt = [
            f"You are a Quality Assurance Scientist. You wrote code to model the feature: '{feature_desc}'.",
            "Below is the resulting 'Feature Dashboard' generated by your code.",
            "The **LEFT Panel** is the Spatial Map. The **RIGHT Panel** is the Statistical Histogram.",
            
            "\n### YOUR TASK",
            "Determine if this result represents a REAL physical feature or an ALGORITHM FAILURE.",
            
            "\n### FAILURE CRITERIA (Reject if ANY are true):",
            "1. **Map Failure (Left):** Is it pure 'salt-and-pepper' static noise with no structure? Is it completely empty/constant?",
            "2. **Histogram Failure (Right):** Is it a single sharp spike (Dirac delta)? This means the code output a constant value.",
            "3. **Complete Rail-Gazing:** The data is piled up at the min/max edges with **NO secondary distribution** visible. (i.e., The algorithm failed everywhere).",
            "4. **Artifacts:** Are there distinct rectangular blocks of NaN/Zeros that look like processing errors?",
            
            "\n### OUTPUT FORMAT",
            "Return a JSON object with:",
            "- 'valid': boolean",
            "- 'critique': string (If invalid, explain clear which panel failed and WHY, so the coder can fix the logic.)"
        ]
        
        # Pass the bytes directly
        check_prompt.append({"mime_type": "image/jpeg", "data": dashboard_bytes})
        
        try:
            # Use low temp for strictness
            config = GenerationConfig(response_mime_type="application/json", temperature=0.1)
            resp = self.model.generate_content(
                check_prompt, 
                generation_config=config,
                safety_settings=self.safety_settings
            )
            result, _ = self._parse_llm_response(resp)
            return result.get("valid", True), result.get("critique", "")
        except Exception as e:
            self.logger.warning(f"QC check crashed: {e}")
            return True, ""
    

class RunSelfReflectionController:
    """
    [🧠 CRITIC Step]
    Reviews the Draft 1 analysis against the images to catch hallucinations.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.instructions = SPECTROSCOPY_REFLECTION_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🧠 --- SELF-REFLECTION: REVIEWING ANALYSIS --- 🧠\n")

        # 1. Get the Draft 1 Analysis
        current_result = state.get("result_json")
        if not current_result:
            self.logger.warning("No analysis found to review.")
            return state
            
        draft_text = current_result.get("detailed_analysis", "")
        claims = current_result.get("scientific_claims", [])

        # 2. Build the Review Prompt
        prompt_parts = [self.instructions]
        prompt_parts.append("\n\n### DRAFT ANALYSIS TO REVIEW:")
        prompt_parts.append(f"{draft_text}")
        prompt_parts.append(f"\n\n### GENERATED CLAIMS:\n{json.dumps(claims, indent=2)}")

        # 3. Add Evidence (Images)
        # The critic needs to see the data to know if the text is lying.
        prompt_parts.append("\n\n### VISUAL EVIDENCE:")
        analysis_images = state.get("analysis_images", [])
        if not analysis_images:
            prompt_parts.append("(No images available for verification)")
        
        for img in analysis_images:
            image_bytes = img.get('data') or img.get('bytes')
            label = img.get('label', 'Unknown Plot')
            if image_bytes:
                prompt_parts.append(f"\n**{label}**")
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})

        # 4. Run Model
        try:
            param_gen_config = GenerationConfig(response_mime_type="application/json")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            review_json, error = self._parse_llm_response(response)
            
            if error:
                self.logger.warning("Reflection failed to parse. Assuming approval.")
                state["reflection_result"] = {"status": "approved"}
            else:
                state["reflection_result"] = review_json
                self.logger.info(f"✅ Reflection Complete. Status: {review_json.get('status')}")
                if review_json.get('status') != 'approved':
                    self.logger.info(f"   Critique: {review_json.get('critique')}")

        except Exception as e:
            self.logger.error(f"Reflection step crashed: {e}")
            state["reflection_result"] = {"status": "approved"} # Fail open

        return state


class ApplyReflectionUpdatesController:
    """
    [🧠 EDITOR Step]
    Applies the changes suggested by the critic, if any.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.instructions = SPECTROSCOPY_REFLECTION_UPDATE_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        
        review = state.get("reflection_result", {})
        if review.get("status") == "approved":
            self.logger.info("⏩ No revisions needed. Proceeding to report generation.")
            return state

        self.logger.info("\n\n🧠 --- REFINEMENT: APPLYING CRITICAL UPDATES --- 🧠\n")

        # 1. Setup Context
        original_result = state.get("result_json")
        critique_text = review.get("critique", "No critique provided.")
        
        prompt_parts = [self.instructions]
        prompt_parts.append(f"\n\n### CRITICAL REVIEW:\n{critique_text}")
        prompt_parts.append(f"\n\n### ORIGINAL DRAFT:\n{json.dumps(original_result, indent=2)}")
        
        # We re-attach images so the editor can verify what needs to be changed
        # (e.g., "Remove discussion of Component 3")
        prompt_parts.append("\n\n### VISUAL CONTEXT (For Reference):")
        for img in state.get("analysis_images", []):
            image_bytes = img.get('data') or img.get('bytes')
            label = img.get('label', 'Unknown Plot')
            if image_bytes:
                prompt_parts.append(f"\n**{label}**")
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})

        # 2. Run Model
        try:
            param_gen_config = GenerationConfig(response_mime_type="application/json")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            updated_json, error = self._parse_llm_response(response)
            
            if not error and updated_json:
                # OVERWRITE the result
                state["result_json"] = updated_json
                self.logger.info("✅ Analysis updated based on self-reflection.")
            else:
                self.logger.warning("Failed to parse updated analysis. Keeping original draft.")

        except Exception as e:
            self.logger.error(f"Refinement step crashed: {e}")
            # Do not overwrite state['result_json'], just keep the old one

        return state