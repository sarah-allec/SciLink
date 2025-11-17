import logging
from typing import Callable, List
from ..controllers.curve_fitting_controllers import (
    RunCurvePreprocessingController,
    CreateInitialPlotController,
    GetLiteratureQueryController,
    RunLiteratureSearchController,
    RunFittingLoopController,
    BuildCurveFittingPromptController
)
from ..controllers.base_controllers import (
    RunFinalInterpretationController,
    StoreAnalysisResultsController
)
from ..preprocess import CurvePreprocessingAgent
from ....executors import ScriptExecutor
from ...lit_agents.literature_agent import FittingModelLiteratureAgent

def create_curve_fitting_pipeline(
    model, 
    logger: logging.Logger, 
    generation_config, 
    safety_settings, 
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable,
    preprocessor: CurvePreprocessingAgent,
    literature_agent: FittingModelLiteratureAgent,
    executor: ScriptExecutor,
    output_dir: str # Pass the agent's output_dir
) -> List:
    """
    Assembles the full, multi-step pipeline for the CurveFittingAgent.
    """
    pipeline = []

    # 1. 🛠️ Tool: Run preprocessor (which has its own LLM/scripting logic)
    pipeline.append(RunCurvePreprocessingController(logger, preprocessor, output_dir))

    # 2. 🛠️ Tool: Plot the (potentially processed) curve for context
    pipeline.append(CreateInitialPlotController(logger))

    # 3. 🧠 LLM: Generate literature search query
    pipeline.append(GetLiteratureQueryController(
        model, logger, generation_config, safety_settings, parse_fn
    ))

    # 4. 🛠️ Tool: Run literature search
    pipeline.append(RunLiteratureSearchController(logger, literature_agent, output_dir))

    # 5. 🛠️/🧠 Meta-Controller: Run the complex fitting/correction/validation loop
    pipeline.append(RunFittingLoopController(
        model, logger, generation_config, safety_settings, parse_fn, executor
    ))
    
    # 6. 📝 Prep: Build prompt for final interpretation
    pipeline.append(BuildCurveFittingPromptController(logger))

    # 7. 🧠 LLM: Run final interpretation
    pipeline.append(RunFinalInterpretationController(
        model, logger, generation_config, safety_settings, parse_fn
    ))

    # 8. 🛠️ Tool: Store final images for feedback
    pipeline.append(StoreAnalysisResultsController(logger, store_fn))
    
    logger.info(f"Curve fitting pipeline created with {len(pipeline)} steps.")
    return pipeline