import logging
import numpy as np
import json
import os
from datetime import datetime
import cv2
from typing import Callable
from google.generativeai.types import GenerationConfig

from ....tools import hyperspectral_tools as tools
from ....tools.image_processor import load_image
from ..preprocess import HyperspectralPreprocessingAgent
from ..instruct import (
    COMPONENT_INITIAL_ESTIMATION_INSTRUCTIONS,
    COMPONENT_SELECTION_WITH_ELBOW_INSTRUCTIONS,
    SPECTROSCOPY_REFINEMENT_SELECTION_INSTRUCTIONS,
    SPECTROSCOPY_HOLISTIC_SYNTHESIS_INSTRUCTIONS
)

class RunPreprocessingController:
    """
    [🛠️ Tool Step]
    Runs the HyperspectralPreprocessingAgent *only if* settings['run_preprocessing'] is True.
    If False (i.e., on a refinement iteration), it *only* calculates statistics
    for the next step.
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

        # --- THIS IS THE NEW LOGIC ---
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
                # Set an all-true mask, since no masking was performed on this iter
                state["preprocessing_mask"] = np.ones(state["hspy_data"].shape[:2], dtype=bool)
                self.logger.info(f"✅ Tool Complete: Statistics calculated. SNR = {snr_value:.2f}")
                return state # Skip the rest of the function
            except Exception as e:
                self.logger.error(f"❌ Tool Failed: Stat calculation on refinement data failed: {e}", exc_info=True)
                state["error_dict"] = {"error": "Stat calculation on refinement data failed", "details": str(e)}
                return state
        # --- END OF NEW LOGIC ---

        # This code now only runs for the *first* iteration (Global Analysis)
        try:
            processed_data, mask, data_quality = self.preprocessor.run_preprocessing(
                state["hspy_data"], 
                state["system_info"]
            )
            state["hspy_data"] = processed_data # Overwrite with processed data
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
    Asks LLM for initial n_components for spectral unmixing.
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
                n_components = 4 # Default fallback
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
            state["initial_n_components"] = 4 # Default fallback
            
        return state

class RunComponentTestLoopController:
    """
    [🛠️ Tool Step]
    Loops from min to max components, runs spectral unmixing, 
    and stores reconstruction errors.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: COMPONENT TEST LOOP --- 🛠️\n")
        
        initial_estimate = state.get("initial_n_components", 4)
        min_c = self.settings.get('min_auto_components', 2)
        max_c = self.settings.get('max_auto_components', min(initial_estimate + 4, 12))
        component_range = list(range(min_c, max_c + 1))
        
        errors = []
        visual_examples = [] # Store visuals for key component numbers
        
        for n_comp in component_range:
            try:
                components, abundance_maps, error = tools.run_spectral_unmixing(
                    state["hspy_data"], n_comp, self.settings, self.logger
                )
                errors.append(error)
                self.logger.info(f"  (Loop {n_comp}/{max_c}): Error = {error:.4f}")

                # Generate visual examples for min, max, and initial estimate
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
                            filename = f"component_test_summary_{n_comp}comp_{timestamp}.jpeg"
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
    Generates the elbow plot from the test loop results.
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
                filename = f"elbow_plot_{timestamp}.jpeg"
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
    Asks LLM to pick the best n_components using the elbow plot and visual examples.
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
            state["final_n_components"] = initial_estimate # Default fallback
            
        return state

class RunFinalSpectralUnmixingController:
    """
    [🛠️ Tool Step]
    Runs spectral unmixing one last time with the final selected n_components.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: FINAL SPECTRAL UNMIXING --- 🛠️\n")
        
        final_n_components = state.get("final_n_components")
        if not final_n_components:
            # If auto-comp failed, try falling back to fixed component count
            final_n_components = self.settings.get('n_components', 4)
            self.logger.warning(f"Auto-selection failed. Using fixed component count: {final_n_components}")
            state["final_n_components"] = final_n_components
            
        try:
            components, abundance_maps, error = tools.run_spectral_unmixing(
                state["hspy_data"], final_n_components, self.settings, self.logger
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
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: CREATE FINAL PLOTS --- 🛠️\n")
        
        components = state.get("final_components")
        abundance_maps = state.get("final_abundance_maps")
        
        if components is None or abundance_maps is None:
            self.logger.warning("Skipping plot creation: final components/maps not found.")
            return state

        # 1. Create component/abundance pairs
        pair_plots = tools.create_component_abundance_pairs(
            components, abundance_maps, state["system_info"], self.logger
        )
        state["component_pair_plots"] = pair_plots # list of {'label':..., 'bytes':...}
        
        try:
            output_dir = self.settings.get('output_dir', 'spectroscopy_output')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, plot in enumerate(pair_plots):
                filename = f"component_pair_{i+1}_{timestamp}.jpeg"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(plot['bytes'])
            self.logger.info(f"📸 Saved {len(pair_plots)} component pair plots to: {output_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to save component pair plots: {e}")
        
        # --- NEW BLOCK TO SAVE THE SUMMARY PLOT ---
        try:
            self.logger.info("  (Tool Info: Creating final NMF summary plot...)")
            n_comp = state.get("final_n_components", components.shape[0])
            summary_bytes = tools.create_nmf_summary_plot(
                components, abundance_maps, n_comp, state["system_info"], self.logger
            )
            if summary_bytes:
                output_dir = self.settings.get('output_dir', 'spectroscopy_output')
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Add iteration title to filename if available
                iter_title = state.get('iteration_title', 'iter').replace(" ", "_")
                filename = f"final_nmf_summary_{iter_title}_{n_comp}comp_{timestamp}.jpeg"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(summary_bytes)
                self.logger.info(f"📸 Saved final NMF summary plot to: {filepath}")
        except Exception as e:
            self.logger.warning(f"Failed to save final NMF summary plot: {e}")
        # --- END NEW BLOCK ---
        
        # 2. Create structure overlays if structure image exists
        if state.get("structure_image_path"):
            try:
                structure_img = load_image(state["structure_image_path"])
                if len(structure_img.shape) == 3:
                    structure_img_gray = cv2.cvtColor(structure_img, cv2.COLOR_RGB2GRAY)
                else:
                    structure_img_gray = structure_img
                
                overlay_bytes = tools.create_multi_abundance_overlays(
                    structure_img_gray, abundance_maps,
                    threshold_percentile=85.0 # Use a high percentile for overlays
                )
                state["structure_overlay_bytes"] = overlay_bytes
                
                if overlay_bytes:
                    try:
                        output_dir = self.settings.get('output_dir', 'spectroscopy_output')
                        os.makedirs(output_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"structure_overlays_{timestamp}.jpeg"
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, 'wb') as f:
                            f.write(overlay_bytes)
                        self.logger.info(f"📸 Saved structure overlay plot to: {filepath}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save structure overlay plot: {e}")
                
            except Exception as e:
                self.logger.warning(f"Failed to create structure overlays: {e}")
                state["structure_overlay_bytes"] = None
        
        self.logger.info("✅ Tool Complete: Final analysis plots created.")
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
        
        prompt_parts = [state["instruction_prompt"]]
        
        # Add data/unmixing info
        h, w, e = state["hspy_data"].shape
        _, energy_xlabel, _ = tools.create_energy_axis(e, state["system_info"])
        
        prompt_parts.append(f"\n\nHyperspectral Data Information:")
        prompt_parts.append(f"- Data shape: ({h}, {w}, {e})")
        prompt_parts.append(f"- X-axis: {energy_xlabel}")
        
        if state.get("final_components") is not None:
            prompt_parts.append(f"- Spectral unmixing method: {state['settings'].get('method', 'nmf').upper()}")
            prompt_parts.append(f"- Number of components: {state['final_n_components']}")
            prompt_parts.append(f"- Final Reconstruction Error: {state.get('final_reconstruction_error', 'N/A'):.4f}")
        else:
            prompt_parts.append("- No spectral unmixing performed.")

        # Add structure overlay
        if state.get("structure_overlay_bytes"):
            prompt_parts.append("\n\n**Structure-Abundance Correlation Analysis:**")
            prompt_parts.append("Overlays showing where NMF components (top 15%) are concentrated on the structural image.")
            prompt_parts.append({"mime_type": "image/jpeg", "data": state["structure_overlay_bytes"]})
            # Add to analysis_images for this iteration
            state["analysis_images"].append({
                "label": "Structure-Abundance Overlays",
                "data": state["structure_overlay_bytes"]
            })

        # Add component-abundance pairs
        if state.get("component_pair_plots"):
            prompt_parts.append("\n\n**Spectral Component Analysis (Component-Abundance Pairs):**")
            for plot in state["component_pair_plots"]:
                prompt_parts.append(f"\n{plot['label']}:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": plot['bytes']})
                
                # --- THIS IS THE FIX ---
                # The plot object is {'label':..., 'bytes':...}
                # We must add it to analysis_images as {'label':..., 'data':...}
                state["analysis_images"].append({
                    "label": plot['label'],
                    "data": plot['bytes']
                })
                # --- END FIX ---

        # Add system info
        if state.get("system_info"):
            sys_info_str = json.dumps(state["system_info"], indent=2)
            prompt_parts.append(f"\n\nAdditional System Information (Metadata):\n{sys_info_str}")

        prompt_parts.append("\n\nProvide your analysis in the requested JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        self.logger.info("✅ Prep Step Complete: Final prompt is ready.")
        return state

# --- NEW CONTROLLERS FOR RECURSIVE ANALYSIS ---

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
        self.instructions = SPECTROSCOPY_REFINEMENT_SELECTION_INSTRUCTIONS

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
            # This is the line that was failing
            image_bytes = img.get('data') or img.get('bytes') # Robustly get bytes
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

            state["refinement_decision"] = result_json
            self.logger.info(f"✅ LLM Step Complete: Refinement decision: {result_json.get('reasoning')}")
            
            # --- Pretty-print for user ---
            print("\n" + "="*80)
            print("🧠 LLM REASONING (SelectRefinementTargetController)")
            print(f"  Refinement Needed: {result_json.get('refinement_needed', 'Error')}")
            print(f"  Explanation: {result_json.get('reasoning', 'N/A')}")
            if result_json.get('refinement_needed'):
                print(f"  Target Type: {result_json.get('target_type')}")
                print(f"  Target Details: {result_json.get('target_details', {}).get('description')}")
            print("="*80 + "\n")
            # --- End pretty-print ---

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: Refinement selection: {e}", exc_info=True)
            state["refinement_decision"] = {"refinement_needed": False, "reasoning": f"Exception: {e}"}
            
        return state

class ApplyRefinementTargetController:
    """
    [🛠️ Tool Step]
    Slices the data for the next iteration based on the LLM's decision.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: APPLY REFINEMENT TARGET --- 🛠️\n")
        
        decision = state.get("refinement_decision")
        if not decision or not decision.get("refinement_needed"):
            self.logger.info("No refinement needed. Setting 'continue_loop' to False.")
            state["continue_loop"] = False
            return state

        try:
            target_type = decision.get("target_type")
            target_details = decision.get("target_details", {})
            target_value = target_details.get("value")
            
            new_iteration_data = None
            iteration_title = f"Focused Analysis on {target_details.get('description', 'target')}"
            new_system_info = state["system_info"] # Default to old one

            if target_type == "spatial":
                self.logger.info(f"Applying SPATIAL refinement: {target_details.get('description')}")
                component_index = int(target_value)
                
                # --- THIS IS THE FIX ---
                # The LLM provides a 1-based index (e.g., "Component 3" -> 3)
                # We must convert it to a 0-based index for Python.
                if component_index > 0:
                    component_index = component_index - 1 # Convert 1-based to 0-based
                else:
                    component_index = 0 # Safety check, use first component
                self.logger.info(f"  (Tool Info: LLM 1-based index {target_value} converted to 0-based index {component_index})")
                # --- END FIX ---
                
                # Use the *current* iteration's data, not the original
                current_data = state.get("hspy_data") 
                if current_data is None:
                    raise ValueError("'hspy_data' (current iteration data) not found in state.")
                
                abundance_maps = state.get("final_abundance_maps")
                if abundance_maps is None:
                    raise ValueError("Cannot apply spatial mask: 'final_abundance_maps' not found.")
                
                # Check bounds *after* converting to 0-index
                if not (0 <= component_index < abundance_maps.shape[2]):
                    raise IndexError(f"LLM-provided component index {target_value} (corrected to {component_index}) is out of bounds for abundance maps with shape {abundance_maps.shape}")

                # Call the *tool* function
                new_iteration_data = tools.apply_spatial_mask(
                    current_data, abundance_maps, component_index
                )

            elif target_type == "spectral":
                self.logger.info(f"Applying SPECTRAL refinement: {target_details.get('description')}")
                
                # Use the *original* data for a spectral slice
                original_data = state.get("original_hspy_data")
                if original_data is None:
                    raise ValueError("'original_hspy_data' not found in state.")

                # Ensure target_value is a list, as expected by the tool
                if not isinstance(target_value, list):
                    raise ValueError(f"Spectral target value must be a list [start, end], but got: {target_value}")
                
                energy_range = list(target_value)
                
                # Call the *tool* function
                new_iteration_data, new_system_info = tools.apply_spectral_slice(
                    original_data, state["system_info"], energy_range
                )

            else:
                raise ValueError(f"Unknown target_type: {target_type}")

            self.logger.info(f"✅ Tool Complete: New data shape for next iteration: {new_iteration_data.shape}")
            state["data_for_this_iteration"] = new_iteration_data
            state["iteration_title"] = iteration_title
            state["system_info"] = new_system_info # Update system info for the *next* loop
            state["continue_loop"] = True # Explicitly continue

        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Applying refinement failed: {e}", exc_info=True)
            # Don't set error_dict, just stop the loop
            state["continue_loop"] = False
            state["refinement_decision"]["reasoning"] = f"Loop stopped. Error applying target: {e}"
            
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

        # Add system info first
        if state.get("system_info"):
            sys_info_str = json.dumps(state["system_info"], indent=2)
            prompt_parts.append(f"\n\n--- System Information ---\n{sys_info_str}")

        # Loop through each iteration's results
        for i, iter_result in enumerate(all_results):
            title = iter_result.get('iteration_title', f'Iteration {i}')
            prompt_parts.append(f"\n\n--- {title} ---")
            
            # Add text summary for this iteration
            iter_analysis = iter_result.get('iteration_analysis_text', 'No text summary.')
            prompt_parts.append(f"**Analysis Summary:**\n{iter_analysis}")
            
            # Add plots for this iteration
            iter_images = iter_result.get('analysis_images', [])
            if iter_images:
                prompt_parts.append("\n**Analysis Plots:**")
                for img in iter_images:
                    # Robustly get bytes, as in the other controller
                    image_bytes = img.get('data') or img.get('bytes') 
                    if image_bytes:
                        prompt_parts.append(f"\n{img['label']}:")
                        prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
            
            # Add the refinement decision that *followed* this iteration
            ref_decision = iter_result.get('refinement_decision', {})
            if ref_decision:
                prompt_parts.append(f"\n**Refinement Decision from this Step:**\n{ref_decision.get('reasoning')}")

        prompt_parts.append("\n\nProvide your final, synthesized analysis in the requested JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        # Store all iteration images for the final feedback step
        all_images = []
        for r in all_results:
            all_images.extend(r.get('analysis_images', []))
        state["analysis_images"] = all_images # Overwrite with the full list
        
        self.logger.info("✅ Prep Step Complete: Final synthesis prompt is ready.")
        return state