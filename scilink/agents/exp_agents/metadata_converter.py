import google.generativeai as genai
import json
import logging
import os
from google.generativeai.types import GenerationConfig
from scilink.auth import get_api_key, APIKeyNotFoundError
from scilink.wrappers.openai_wrapper import OpenAIAsGenerativeModel


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the JSON Schema as a Dictionary
METADATA_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        # --- Core Identification ---
        "experiment_type": {
            "type": "string",
            "description": "General type of experiment (e.g., Microscopy, Spectroscopy, Diffraction, Curve Analysis)."
        },
        # --- Detailed Experiment Info ---
        "experiment": {
            "type": "object",
            "description": "Details about the specific experimental setup.",
            "properties": {
                "technique": {
                    "type": "string",
                    "description": "Specific experimental technique used (e.g., HAADF-STEM, TEM, Photoluminescence Spectroscopy, XRD)."
                },
                "date": {
                    "type": ["string", "null"], # Optional
                    "description": "Date of the experiment (if available, YYYY-MM-DD)."
                },
                "instrument": {
                    "type": ["string", "null"], # Optional
                    "description": "Instrument used for the experiment."
                },
                "details": {
                    "type": ["string", "null"], # Optional
                    "description": "Other relevant conditions or parameters (e.g., voltage, temperature, excitation source, probe current)."
                }
            },
            "required": ["technique"] # Technique is crucial
        },
        # --- Sample Info ---
        "sample": {
            "type": "object",
            "description": "Details about the sample.",
            "properties": {
                "material": {
                    "type": "string",
                    "description": "Specific material name, composition, or formula (e.g., MoS2, Gallium Nitride (GaN))."
                },
                "description": {
                    "type": ["string", "null"], # Optional
                    "description": "Additional description (e.g., form, substrate, synthesis method)."
                }
            },
            "required": ["material"] # Specific material is crucial
        },
        # --- Microscopy Specific (Conditional) ---
        "spatial_info": {
            "type": ["object", "null"], # Optional object
            "description": "Spatial dimensions/scale (Primarily for Microscopy).",
            "properties": {
                "field_of_view_x": {"type": ["number", "null"]},
                "field_of_view_y": {"type": ["number", "null"]},
                "field_of_view_units": {"type": ["string", "null"], "description": "Units like 'nm', 'um', 'pixels'."}
            },
        },
        # --- Spectroscopy/Hyperspectral Specific (Conditional) ---
        "energy_range": {
            "type": ["object", "null"], # Optional object
            "description": "Spectral or energy range covered (Primarily for Spectroscopy/Hyperspectral).",
            "properties": {
                "start": {"type": ["number", "null"]},
                "end": {"type": ["number", "null"]},
                "units": {"type": ["string", "null"], "description": "Units like 'nm', 'eV', 'cm^-1'."}
            },
        },
        # --- 1D Curve Specific (Conditional) ---
        "title": {
            "type": ["string", "null"], # Optional
            "description": "A descriptive title for the data/plot (Primarily for 1D Curves)."
        },
        "data_columns": {
            "type": ["array", "null"], # Optional array
            "description": "Description of data columns, typically X and Y (Primarily for 1D Curves).",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name like 'Wavelength', 'Angle', 'Energy', 'Intensity', 'Counts'."},
                    "units": {"type": "string"}
                },
                "required": ["name", "units"]
            }
        },
        "xlabel": {
            "type": ["string", "null"], # Optional
            "description": "Suggested X-axis label, including units (Primarily for 1D Curves)."
        },
        "ylabel": {
            "type": ["string", "null"], # Optional
            "description": "Suggested Y-axis label, including units (Primarily for 1D Curves)."
        }
    },
    # Universally required fields at the top level
    "required": ["experiment_type", "experiment", "sample"]
}

# Convert schema dictionary to a JSON string for embedding in the prompt
schema_json_string_for_prompt = json.dumps(METADATA_SCHEMA_DICT, indent=2)

METADATA_GENERATION_PROMPT = f"""
You are an expert scientific assistant. Your task is to read the provided plain text description of a scientific experiment (which could be microscopy, spectroscopy/hyperspectral, or 1D curve data like PL/XRD) and extract key metadata into a structured JSON format.

Format your output STRICTLY as a valid JSON object conforming *exactly* to the following structure:

```json
{schema_json_string_for_prompt}
```

Instructions:

Identify Experiment Type: First, determine the general experiment_type (e.g., Microscopy, Spectroscopy, Diffraction, Curve Analysis) and the specific experiment.technique (e.g., HAADF-STEM, TEM, PL, XRD, Absorption). Also identify sample.material.

Fill Core Fields: Populate the required fields: experiment_type, experiment (including technique), and sample (including material). Extract other details like instrument or conditions into experiment.details or sample.description if available.

Fill Conditional Fields based on Type:

If Microscopy: Focus on extracting spatial_info (field of view and units). Omit or use null for spectroscopy/curve fields if not relevant.

If Spectroscopy/Hyperspectral: Focus on extracting energy_range (start, end, units). Omit or use null for microscopy/curve fields if not relevant.

If 1D Curve Data (PL, XRD, etc.): Focus on extracting title, data_columns (determining X and Y column names/units), xlabel, and ylabel. energy_range might also apply if the x-axis represents energy. Omit or use null for microscopy fields.

Handle Missing Info: Use null for any optional fields (like experiment.date) or omit entire optional objects (spatial_info, energy_range, data_columns) if the information is missing or clearly not applicable to the described experiment type.

Required Fields: For universally required fields (experiment_type, experiment.technique, sample.material) that are truly missing even after careful reading, use the string "N/A".

Strict Formatting: Only include fields defined in the schema. Output ONLY the valid JSON object without markdown formatting.

Ensure the output JSON accurately reflects the information present in the text description and adheres to the conditional logic based on the experiment type.
"""


def generate_metadata_json_from_text(
    input_text_filepath: str,          
    google_api_key: str | None = None,
    model_name: str = "gemini-2.5-flash-preview-05-20",
    local_model: str | None = None
) -> dict | None:
    """
    Reads a plain text experimental description from a file, uses an LLM model
    (Gemini or OpenAI-compatible) to convert it into a structured JSON metadata
    dictionary, saves the JSON to a file, and returns the dictionary.

    Args:
        input_text_filepath: Path to the input text file containing the description.
        google_api_key: Your Google AI API key (used for Gemini or as the key for OpenAI-compatible endpoint).
                          If None, attempts to retrieve via scilink.auth.get_api_key('google').
        model_name: The Gemini model to use (if local_model is None).
                    For OpenAI-compatible endpoints, this might specify the model served at the endpoint.
        local_model: Optional. The base URL of the OpenAI-compatible API endpoint.
                     If provided, this function will use the OpenAI wrapper.

    Returns:
        A dictionary containing the extracted metadata, or None if an error occurs.
        Saves the output JSON to a file named based on the input file.
    """
    # --- Input File Validation and Reading ---
    if not input_text_filepath or not os.path.exists(input_text_filepath):
        logger.error(f"Input text file not found or path is invalid: {input_text_filepath}")
        return None
    try:
        with open(input_text_filepath, 'r', encoding='utf-8') as f:
            text_description = f.read()
        if not text_description.strip():
            logger.error(f"Input text file is empty: {input_text_filepath}")
            return None
        logger.info(f"Read description from: {input_text_filepath}")
    except Exception as e:
        logger.error(f"Error reading input file {input_text_filepath}: {e}", exc_info=True)
        return None

    # --- Determine Output File Path ---
    base_name = os.path.splitext(input_text_filepath)[0]
    output_json_filepath = f"{base_name}.json"

    # --- API Key Configuration ---
    api_key_to_use = google_api_key
    if api_key_to_use is None:
        try:
            api_key_to_use = get_api_key('google')
            if not api_key_to_use:
                raise ValueError("API Key not found via scilink.auth.get_api_key('google').")
        except (ImportError, ValueError, ModuleNotFoundError, APIKeyNotFoundError) as e:
            logger.error(f"Failed to get Google API key via scilink.auth: {e}. Provide key or configure scilink.auth.")
            return None

    model = None
    use_openai_wrapper = False
    generation_config_dict = None # Use dict for Gemini

    try:
        # --- Conditional Model Initialization ---
        if local_model and 'ai-incubator' in local_model: # Using trigger string convention
            logger.info(f"🏛️ Using network agent (OpenAI wrapper) via endpoint: {local_model}")
            model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key_to_use,
                base_url=local_model
            )
            use_openai_wrapper = True
        else:
            logger.info(f"☁️ Using Google Generative AI model: {model_name}")
            genai.configure(api_key=api_key_to_use)
            model = genai.GenerativeModel(model_name)
            use_openai_wrapper = False
            generation_config_dict = {
                "response_mime_type": "application/json",
                "response_schema": METADATA_SCHEMA_DICT
            }

        # --- Prepare Prompt ---
        prompt_parts = [
            METADATA_GENERATION_PROMPT,
            "\n--- Plain Text Description ---",
            text_description, # Use content read from file
            "\n--- Extracted JSON Metadata ---"
        ]

        # --- API Call ---
        logger.info(f"Sending request to {'OpenAI wrapper' if use_openai_wrapper else model_name} for metadata extraction...")
        if use_openai_wrapper:
            response = model.generate_content(contents=prompt_parts)
        else:
            response = model.generate_content(
                contents=prompt_parts,
                generation_config=generation_config_dict,
            )

        # --- Response Processing ---
        json_text = ""
        if hasattr(response, 'candidates') and not response.candidates:
             logger.error("LLM response was empty or blocked.")
             if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                  logger.error(f"Request blocked due to: {response.prompt_feedback.block_reason}")
             return None
        elif not hasattr(response, 'text'):
             logger.error("LLM response object does not contain expected 'text' attribute.")
             try: logger.error(f"Raw response: {response}")
             except: pass
             return None

        json_text = response.text
        if json_text.strip().startswith("```json"):
            json_text = json_text.strip()[7:]
        if json_text.strip().endswith("```"):
            json_text = json_text.strip()[:-3]
        logger.debug(f"Raw LLM JSON response text: {json_text}")

        metadata_dict = json.loads(json_text.strip())
        logger.info("Successfully extracted and parsed metadata JSON.")

        # --- Save Output JSON File ---
        try:
            with open(output_json_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=4)
            logger.info(f"Successfully saved metadata to: {output_json_filepath}")
        except Exception as e:
            logger.error(f"Error saving metadata JSON to {output_json_filepath}: {e}", exc_info=True)
            # Continue to return the dict even if saving fails

        return metadata_dict

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        logger.error(f"Raw text received: {json_text}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during API call or processing: {e}", exc_info=True)
        return None