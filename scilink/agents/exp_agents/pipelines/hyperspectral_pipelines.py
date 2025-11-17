import logging
from typing import Callable, List
from ..controllers.hyperspectral_controllers import (
    RunPreprocessingController,
    GetInitialComponentParamsController,
    RunComponentTestLoopController,
    CreateElbowPlotController,
    GetFinalComponentSelectionController,
    RunFinalSpectralUnmixingController,
    CreateAnalysisPlotsController,
    BuildHyperspectralPromptController
)
from ..controllers.base_controllers import (
    RunFinalInterpretationController,
    StoreAnalysisResultsController
)
from ..preprocess import HyperspectralPreprocessingAgent

def create_hyperspectral_pipeline(
    model, 
    logger: logging.Logger, 
    generation_config, 
    safety_settings, 
    settings: dict,
    preprocessor: HyperspectralPreprocessingAgent,
    parse_fn: Callable,
    store_fn: Callable
) -> List:
    """
    Assembles the full, multi-step pipeline for the HyperspectralAnalysisAgent.
    """
    
    pipeline = []
    
    # --- 1. PREPROCESSING (Optional) ---
    if settings.get('run_preprocessing', True):
        pipeline.append(RunPreprocessingController(logger, preprocessor))
    
    # --- 2. SPECTRAL UNMIXING (Optional) ---
    if settings.get('enabled', True):
        
        # 2a. Auto-component workflow
        if settings.get('auto_components', True):
            # [🧠 LLM] Get initial component guess
            pipeline.append(GetInitialComponentParamsController(
                model, logger, generation_config, safety_settings, parse_fn
            ))
            # [🛠️ Tool] Run NMF loop to get errors
            pipeline.append(RunComponentTestLoopController(logger, settings))
            # [🛠️ Tool] Create elbow plot from errors
            pipeline.append(CreateElbowPlotController(logger, settings))
            # [🛠️ LLM] Select final n_components from elbow plot
            pipeline.append(GetFinalComponentSelectionController(
                model, logger, generation_config, safety_settings, parse_fn
            ))
        
        # 2b. [🛠️ Tool] Run final NMF (either with selected or fixed components)
        pipeline.append(RunFinalSpectralUnmixingController(logger, settings))
        
        # 2c. [🛠️ Tool] Create all plots for analysis
        pipeline.append(CreateAnalysisPlotsController(logger, settings))

    # --- 3. FINAL INTERPRETATION ---
    
    # 3a. [📝 Prep] Build the final prompt
    pipeline.append(BuildHyperspectralPromptController(logger))

    # 3b. [🧠 LLM] Run final interpretation (from base_controllers)
    pipeline.append(RunFinalInterpretationController(
        model, logger, generation_config, safety_settings, parse_fn
    ))

    # 3c. [🛠️ Tool] Store images for feedback (from base_controllers)
    pipeline.append(StoreAnalysisResultsController(logger, store_fn))
    
    logger.info(f"Hyperspectral pipeline created with {len(pipeline)} steps.")
    return pipeline