import csv
import json
from pathlib import Path
from typing import Dict, Any

from .parser_utils import table_to_markdown

def parse_csv_with_context(csv_path: str, context_path: str) -> Dict[str, Any]:
    """
    Reads a CSV file and a corresponding JSON context file with a rich schema,
    then combines them into a single, detailed text chunk for a RAG pipeline.
    """
    print(f"  - Processing CSV '{Path(csv_path).name}' with rich context from '{Path(context_path).name}'")
    try:
        # --- 1. Read and validate the structured context from the JSON file ---
        with open(context_path, 'r', encoding='utf-8') as f:
            context = json.load(f)

        # Validate that essential keys exist
        if not all(k in context for k in ["objective", "column_definitions"]):
            print(f"    - ⚠️  Skipping pair: JSON '{context_path}' is missing required keys ('objective', 'column_definitions').")
            return None

        # --- 2. Build a descriptive text block from the JSON context ---
        description_parts = []
        if context.get("experiment_title"):
            description_parts.append(f"### Experiment: {context['experiment_title']}")
        
        description_parts.append(f"#### Objective\n{context['objective']}")

        if context.get("instrumentation"):
            instruments = "\n".join([f"- {item}" for item in context['instrumentation']])
            description_parts.append(f"#### Instrumentation\n{instruments}")

        # Format column definitions for clarity
        col_defs = "\n".join([f"- `{col}`: {desc}" for col, desc in context['column_definitions'].items()])
        description_parts.append(f"#### Data Column Definitions\n{col_defs}")

        # Combine all descriptive parts into a single string
        description_text = "\n\n".join(description_parts)

        # --- 3. Read the CSV and convert it to a Markdown table ---
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            data_list = list(reader)
        markdown_table = table_to_markdown(data_list)
        
        # --- 4. Combine the rich description and raw data into the final chunk text ---
        final_text = f"""
{description_text}

#### Raw Experimental Data
{markdown_table}
        """.strip()

        chunk = {
            'text': final_text,
            'metadata': {
                'source': csv_path,
                'context_source': context_path,
                'content_type': 'experimental_data_package',
                'page': 1 # Placeholder
            }
        }
        print(f"    - ✅ Successfully created rich chunk for {Path(csv_path).name}")
        return chunk

    except FileNotFoundError as e:
        print(f"    - ❌ Error processing data pair: File not found - {e}")
        return None
    except json.JSONDecodeError:
        print(f"    - ❌ Error processing data pair: Invalid JSON in file '{context_path}'")
        return None
    except Exception as e:
        print(f"    - ❌ Error processing data pair for '{csv_path}': {e}")
        return None
