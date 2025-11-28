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

from ....tools import hyperspectral_tools as tools
from ....tools.image_processor import load_image
from ..preprocess import HyperspectralPreprocessingAgent
from ..instruct import (
    COMPONENT_INITIAL_ESTIMATION_INSTRUCTIONS,
    COMPONENT_SELECTION_WITH_ELBOW_INSTRUCTIONS,
    SPECTROSCOPY_REFINEMENT_INSTRUCTIONS,
    SPECTROSCOPY_HOLISTIC_SYNTHESIS_INSTRUCTIONS,
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
    Generates all final visualizations for the LLM.
    
    UPDATED LOGIC:
    1. Strict Naming: Uses "{Iteration_Title}_{Label}.jpeg"
    2. Split Validation Logic:
       - Depth 0 (Global): Standard NMF plots.
       - Depth > 0 (Refinement): Validated NMF plots (Red vs Black lines).
    3. Validated Summary Grid: For Depth > 0, we STITCH individual validated plots
       to create a "Validation Grid" because the standard tool doesn't support validation.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: CREATE FINAL PLOTS --- 🛠️\n")
        
        components = state.get("final_components")
        abundance_maps = state.get("final_abundance_maps")
        current_depth = state.get("current_depth", 0)
        
        # Retrieve the stable iteration title (e.g., "Global_Analysis", "Focused_Analysis_D1_T1")
        iter_title_raw = state.get("iteration_title", "Global_Analysis")
        iter_prefix = _sanitize_filename(iter_title_raw)

        if components is None or abundance_maps is None:
            self.logger.warning("Skipping plot creation: final components/maps not found.")
            return state

        # Output directory setup
        output_dir = self.settings.get('output_dir', 'spectroscopy_output')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- 1. Generate Component Pairs ---
        final_plots = []
        # For Depth > 0, we keep the raw bytes to stitch them later
        validated_bytes_list = [] 
        
        if current_depth == 0:
            # GLOBAL ANALYSIS: Standard NMF Pairs
            self.logger.info("Global Analysis (Depth 0): Generating Standard NMF Pairs.")
            final_plots = tools.create_component_abundance_pairs(
                components, abundance_maps, state["system_info"], self.logger
            )
        else:
            # REFINEMENT ANALYSIS: Validated Pairs (Map + Validation Spectrum)
            self.logger.info(f"Refinement (Depth {current_depth}): Generating Validated Pairs.")
            for i in range(components.shape[0]):
                plot_bytes = tools.create_validated_component_pair(
                    state["hspy_data"], 
                    components[i], 
                    abundance_maps[..., i], 
                    i, 
                    self.logger
                )
                if plot_bytes:
                    final_plots.append({
                        'label': f"Component {i+1} Analysis", # Label without "Validated" string, implied by depth
                        'bytes': plot_bytes
                    })
                    validated_bytes_list.append(plot_bytes)

        state["component_pair_plots"] = final_plots
        
        # Save individual component plots to disk using SEMANTIC names
        try:
            for i, plot in enumerate(final_plots):
                # Label: "Component 1 Analysis" -> "Component_1_Analysis"
                label_safe = _sanitize_filename(plot['label'])
                filename = f"{iter_prefix}_{label_safe}.jpeg"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(plot['bytes'])
            
                # Add to analysis_images with the RAW label (Final synthesis controller handles the referencing)
                state["analysis_images"].append({
                    "label": plot['label'],
                    "data": plot['bytes']
                })
                
        except Exception as e:
            self.logger.warning(f"Failed to save component plots: {e}")
        
        # --- 2. NMF Summary Grid (The "Executive Summary" View) ---
        try:
            self.logger.info("  (Tool Info: Creating NMF summary plot...)")
            n_comp = state.get("final_n_components", components.shape[0])
            summary_bytes = None
            
            if current_depth == 0:
                # Standard Grid for Global
                summary_bytes = tools.create_nmf_summary_plot(
                    components, abundance_maps, n_comp, state["system_info"], self.logger
                )
            else:
                # Validated Grid for Refinement (Stitching)
                self.logger.info("  (Tool Info: Stitching validated plots into Summary Grid...)")
                summary_bytes = _create_grid_from_images(validated_bytes_list, self.logger)

            if summary_bytes:
                label = "NMF Summary Grid"
                label_safe = _sanitize_filename(label)
                filename = f"{iter_prefix}_{label_safe}.jpeg"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(summary_bytes)
                self.logger.info(f"📸 Saved NMF summary plot to: {filepath}")

                # Add to Final Report State
                state["analysis_images"].append({
                    "label": label, 
                    "data": summary_bytes
                })

        except Exception as e:
            self.logger.warning(f"Failed to create/save NMF summary plot: {e}")

        # --- 3. Structure Overlays (Optional) ---
        if state.get("structure_image_path"):
            try:
                structure_img = load_image(state["structure_image_path"])
                if len(structure_img.shape) == 3:
                    structure_img_gray = cv2.cvtColor(structure_img, cv2.COLOR_RGB2GRAY)
                else:
                    structure_img_gray = structure_img
                
                overlay_bytes = tools.create_multi_abundance_overlays(
                    structure_img_gray, abundance_maps,
                    threshold_percentile=85.0 
                )
                state["structure_overlay_bytes"] = overlay_bytes
                
                if overlay_bytes:
                    label = "Structure-Abundance Overlays"
                    label_safe = _sanitize_filename(label)
                    filename = f"{iter_prefix}_{label_safe}.jpeg"
                    filepath = os.path.join(output_dir, filename)

                    try:
                        with open(filepath, 'wb') as f:
                            f.write(overlay_bytes)
                        self.logger.info(f"📸 Saved structure overlay plot to: {filepath}")
                        
                        # Add to final report state
                        state["analysis_images"].append({
                            "label": label,
                            "data": overlay_bytes
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to save structure overlay plot: {e}")
                
            except Exception as e:
                self.logger.warning(f"Failed to create structure overlays: {e}")
                state["structure_overlay_bytes"] = None

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
                prompt_parts.append("### 🧪 Quantitative Validation Mode")
                prompt_parts.append("Because this is a focused refinement, we provide **Validated Analysis Plots**.")
                prompt_parts.append("- **Left Image (Map):** Spatial distribution of the component.")
                prompt_parts.append("- **Right Image (Validation Graph):** Overlays the NMF Model (Red) vs. the Abundance-Weighted Raw Data (Black).")
                prompt_parts.append("👉 **CRITICAL:** Trust the Black Line (Raw Data) for peak shapes and intensities. If the Red line creates peaks not seen in the Black line, they are artifacts.")

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

            targets = result_json.get("targets", [])
            is_needed = result_json.get("refinement_needed", False)

            # Store the final decision including the list of targets
            state["refinement_decision"] = {
                "refinement_needed": is_needed,
                "reasoning": result_json.get("reasoning", "No reasoning provided."),
                "targets": targets
            }

            self.logger.info(f"✅ LLM Step Complete: Refinement decision: {state['refinement_decision']['reasoning']}")
            
            print("\n" + "="*80)
            print("🧠 LLM REASONING (SelectRefinementTargetController)")
            print(f"  Refinement Needed: {is_needed}")
            print(f"  Explanation: {state['refinement_decision']['reasoning']}")
            print(f"  Targets Found: {len(targets)}")
            if targets:
                for i, t in enumerate(targets):
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
    
    Updated: Generates structured, short IDs for task titles
    (e.g., Focused_Analysis_D1_T1) instead of long descriptions.
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
                    
                    # t_value is expected to be [start_index, end_index]
                    slice_indices = list(t_value)
                    
                    # 1. Perform the slicing (existing logic)
                    new_data, _ = tools.apply_spectral_slice(
                        state["original_hspy_data"], state["system_info"], slice_indices
                    )
                    
                    # 2. Recalculate Metadata Calibration
                    new_sys_info = state["system_info"].copy()
                    
                    start_idx, end_idx = slice_indices[0], slice_indices[1]
                    
                    # Check if we have valid energy range data to interpolate
                    energy_meta = new_sys_info.get('energy_range', {})
                    if energy_meta and energy_meta.get('start') is not None and energy_meta.get('end') is not None:
                        
                        orig_start = float(energy_meta['start'])
                        orig_end = float(energy_meta['end'])
                        # Original number of channels
                        orig_len = state["original_hspy_data"].shape[-1]
                        
                        # Calculate dispersion (Energy per pixel)
                        dispersion = (orig_end - orig_start) / orig_len
                        
                        # Calculate NEW start and end based on the slice indices
                        new_start = orig_start + (start_idx * dispersion)
                        new_end = orig_start + (end_idx * dispersion)
                        
                        # Update the metadata for the new task
                        new_sys_info['energy_range'] = {
                            'start': new_start,
                            'end': new_end,
                            'units': energy_meta.get('units', 'units')
                        }
                        self.logger.info(f"Recalibrated axis: {orig_start}-{orig_end} -> {new_start:.2f}-{new_end:.2f}")

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
    
    Updated Fix:
    1. Removes dynamic "Figure X" counting which caused inconsistency.
    2. Uses stable, semantic labels: "[Iteration_Name] Plot_Name".
    3. Instructions updated to force the LLM to cite these semantic labels.
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
            # Get the semantic title (e.g., "Global_Analysis", "Focused_Analysis_D1_T1")
            raw_title = iter_result.get('iteration_title', f'Iteration_{i}')
            # Sanitize title for cleaner references (remove spaces if any)
            iter_ref_id = _sanitize_filename(raw_title)
            
            prompt_parts.append(f"\n\n### SECTION {i+1}: {raw_title}")
            
            # Re-inject Context
            context_desc = iter_result.get('parent_refinement_reasoning') 
            if context_desc:
                prompt_parts.append(f"**Target Description:** \"{context_desc}\"")
            
            # Text Summary
            iter_analysis = iter_result.get('iteration_analysis_text', 'No text summary.')
            prompt_parts.append(f"**Previous Analysis Summary:**\n{iter_analysis}")
            
            # Refinement Decision
            ref_decision = iter_result.get('refinement_decision', {})
            if ref_decision:
                prompt_parts.append(f"\n**Reason for Next Step:** {ref_decision.get('reasoning')}")

            # Visual Evidence
            iter_images = iter_result.get('analysis_images', [])
            if iter_images:
                prompt_parts.append(f"\n**Visual Evidence for {raw_title}:**")
                for img in iter_images:
                    # Robustly get bytes
                    image_bytes = img.get('data') or img.get('bytes')
                    raw_label = img.get('label', 'Unknown_Plot')
                    
                    if image_bytes:
                        # --- FIXED REFERENCE LOGIC ---
                        # Create a stable semantic ID instead of "Figure X"
                        # Format: [Context] Content
                        # Example: "[Global_Analysis] NMF Summary Grid"
                        unique_ref = f"[{iter_ref_id}] {raw_label}"
                        
                        prompt_parts.append(f"\n**{unique_ref}**")
                        prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
                        
                        # Update the label in the list we return, so the final report matches
                        img['label'] = unique_ref 
                        all_images.append(img)

        # 3. EXPLICIT REPORTING INSTRUCTIONS
        prompt_parts.append("\n\n### 📝 CRITICAL REPORTING INSTRUCTIONS")
        prompt_parts.append("1. Write a cohesive narrative synthesizing the findings from all iterations.")
        prompt_parts.append("2. **AT THE END of your 'detailed_analysis' text**, you MUST append a section titled **'### Key Evidence'**.")
        prompt_parts.append("3. In that section, you MUST list the supporting figures using their **EXACT bolded titles** provided above (the strings inside brackets).")
        
        prompt_parts.append("\n**Required Format for Evidence Section:**")
        prompt_parts.append("### Key Evidence")
        prompt_parts.append("- **[Global_Analysis] NMF Summary Grid**: Explain what this specific plot proves.")
        prompt_parts.append("- **[Focused_Analysis_D1_T1] Component 1 Analysis**: ...")
        prompt_parts.append("\n(Use the exact reference strings provided above. Do not invent figure numbers like 'Figure 1' unless they are part of the name.)")

        prompt_parts.append("\n\nProvide your final, synthesized analysis in the requested JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        state["analysis_images"] = all_images 
        
        self.logger.info("✅ Prep Step Complete: Final synthesis prompt is ready.")
        return state
    

class GenerateHTMLReportController:
    """
    [🛠️ Tool Step]
    Generates a beautiful, human-readable HTML report.
    
    UPDATED LOGIC:
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
        """Returns only images whose labels appear exactly in the analysis text."""
        cited_images = []
        for img in all_images:
            label = img.get('label', '')
            # Check if the label (e.g. "[Global_Analysis] NMF Summary Grid") is in the text
            if label in text:
                cited_images.append(img)
        return cited_images

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