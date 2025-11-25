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
    BuildHyperspectralPromptController,
    SelectRefinementTargetController,
    GenerateRefinementTasksController,
    BuildHolisticSynthesisPromptController
)
from ..controllers.base_controllers import (
    RunFinalInterpretationController,
    StoreAnalysisResultsController,
    # Import the new controller
    IterativeFeedbackController
)
from ..preprocess import HyperspectralPreprocessingAgent

def create_hyperspectral_iteration_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    preprocessor: HyperspectralPreprocessingAgent,
    parse_fn: Callable
) -> List:
    """
    Assembles the pipeline that runs *per-iteration* of the recursive analysis.
    This includes: NMF -> Plotting -> Refinement Decision -> Data Slicing.
    """

    pipeline = []

    # --- 1. PREPROCESSING (Only on first iteration) ---
    if settings.get('run_preprocessing', True):
        pipeline.append(RunPreprocessingController(logger, preprocessor))

    # --- 2. SPECTRAL UNMIXING ---
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

        # 2b. [🛠️ Tool] Run final NMF
        pipeline.append(RunFinalSpectralUnmixingController(logger, settings))

        # 2c. [🛠️ Tool] Create all plots for analysis
        pipeline.append(CreateAnalysisPlotsController(logger, settings))

    # --- 3. ITERATION ANALYSIS & REFINEMENT DECISION ---

    # 3a. [📝 Prep] Build the prompt for *this iteration*
    pipeline.append(BuildHyperspectralPromptController(logger))

    # 3b. [🧠 LLM] Run interpretation for *this iteration*
    pipeline.append(RunFinalInterpretationController(
        model, logger, generation_config, safety_settings, parse_fn
    ))

    # 3c. [🧠 LLM] Decide if we need to zoom
    pipeline.append(SelectRefinementTargetController(
        model, logger, generation_config, safety_settings, parse_fn
    ))
    
    # 3d. [🧠/👤 User] Immediate Feedback step
    pipeline.append(IterativeFeedbackController(
        model, logger, generation_config, safety_settings, parse_fn
    ))
    
    # 3e. [🛠️ Tool] Prepare data for next loop (if needed)
    pipeline.append(GenerateRefinementTasksController(logger))

    logger.info(f"Hyperspectral *iteration* pipeline created with {len(pipeline)} steps.")
    return pipeline

def create_hyperspectral_synthesis_pipeline(
    model,  
    logger: logging.Logger,  
    generation_config,  
    safety_settings,  
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable
) -> List:
    # ... (Implementation of synthesis pipeline remains unchanged)
    pipeline = []

    # 1. [📝 Prep] Build the holistic synthesis prompt
    pipeline.append(BuildHolisticSynthesisPromptController(logger))

    # 2. [🧠 LLM] Run final synthesis interpretation
    pipeline.append(RunFinalInterpretationController(
        model, logger, generation_config, safety_settings, parse_fn
    ))

    # 3. [🛠️ Tool] Store all images from all iterations
    pipeline.append(StoreAnalysisResultsController(logger, store_fn))
    
    logger.info(f"Hyperspectral *synthesis* pipeline created with {len(pipeline)} steps.")
    return pipeline