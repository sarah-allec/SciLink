from ..controllers.microscopy_controllers import (
    GetFFTParamsController,
    RunFFTNMFController,
    BuildFFTNMFPromptController,
    RunGlobalFFTController
)
from ..controllers.base_controllers import (
    RunFinalInterpretationController,
    StoreAnalysisResultsController
)
from typing import Callable, List

def create_fftnmf_pipeline(
    model, 
    logger, 
    generation_config, 
    safety_settings, 
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable
) -> List:
    """
    Assembles and returns a list of controller instances 
    for the standard FFT/NMF analysis pipeline.
    """
    
    pipeline = []
    
    # --- 1. Global FFT (runs for every image) ---
    pipeline.append(
        RunGlobalFFTController(logger)
    )

    # --- 2. Sliding FFT/NMF Specific Workflow ---
    if settings.get('FFT_NMF_ENABLED', False):
        
        # 2a. 🧠 LLM Step (Reason)
        pipeline.append(
            GetFFTParamsController(model, logger, generation_config, safety_settings)
        )
        
        # 2b. 🛠️ Tool Step (Act)
        pipeline.append(
            RunFFTNMFController(logger, settings)
        )
    
    # --- 3. Generic Workflow (Final Steps) ---
    
    # 3a. 📝 Prep Step (Builds prompt)
    pipeline.append(
        BuildFFTNMFPromptController(logger)
    )
    
    # 3b. 🧠 LLM Step (Interpret)
    pipeline.append(
        RunFinalInterpretationController(
            model, logger, generation_config, safety_settings, parse_fn
        )
    )
    
    # 3c. 🛠️ Tool Step (Store results)
    pipeline.append(
        StoreAnalysisResultsController(logger, store_fn)
    )
    
    return pipeline