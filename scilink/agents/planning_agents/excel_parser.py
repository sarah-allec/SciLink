# planning_agents/excel_parser.py
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List

# If a file has this many rows or fewer, we embed it all in one chunk.
SMALL_FILE_THRESHOLD = 150

def parse_adaptive_excel(excel_path: str, context_path: str, row_chunk_size: int = 500) -> List[Dict[str, Any]]:
    """
    Reads an Excel file and a JSON context file with an adaptive strategy.    
    - If rows <= SMALL_FILE_THRESHOLD:
      Creates ONE chunk containing the summary, definitions, AND the full data table.
    - If rows > SMALL_FILE_THRESHOLD:
      Creates TWO types of chunks:
      1. A single "summary chunk" with statistical info.
      2. Multiple "data chunks" by batching the rows.
    """
    print(f"  - Processing Excel '{Path(excel_path).name}' with adaptive strategy...")
    all_chunks = []

    try:
        # --- 1. Read and validate the structured context ---
        with open(context_path, 'r', encoding='utf-8') as f:
            context = json.load(f)
        
        # Validate that at least one of 'objective' or 'title' exists
        if "objective" not in context and "title" not in context:
            print(f"    - ⚠️  Skipping: JSON '{context_path}' must contain at least one of 'objective' or 'title'.")
            return []

        # --- 2. Load the Excel file ---
        try:
            df = pd.read_excel(excel_path)
        except ImportError:
            print("    - ❌ Error: 'pandas' or 'openpyxl' not installed. Please run: pip install pandas openpyxl")
            return []
        
        total_rows = len(df)
        print(f"    - Loaded {total_rows} rows from Excel.")

        # --- 3. Base Content (common to all strategies) ---
        
        description_parts = []
        
        # Get title: Use 'title' if present, else fallback to filename
        title = context.get('title', Path(excel_path).stem)
        description_parts.append(f"### Experiment: {title}")
        
        # Get objective: Only add if present
        if context.get("objective"):
            description_parts.append(f"#### Objective\n{context['objective']}")

        # Get or create column definitions
        column_defs_dict = context.get('column_definitions')
        if not column_defs_dict:
            print(f"     - ℹ️  'column_definitions' not found in JSON. Using headers from '{Path(excel_path).name}'.")
            # Create definitions from DataFrame column headers
            column_defs_dict = {str(header): "No definition provided." for header in df.columns}

        col_defs = "\n".join([f"- `{col}`: {desc}" for col, desc in column_defs_dict.items()])
        description_parts.append(f"#### Data Column Definitions\n{col_defs}")
        
        statistical_summary = df.describe().to_markdown() if not df.empty else "No statistical summary available."

        # --- 4. Adaptive Chunking Logic ---
        
        if total_rows <= SMALL_FILE_THRESHOLD:
            # --- STRATEGY A: Small File (One Rich Chunk) ---
            print(f"    - File is small ({total_rows} rows). Creating one single, comprehensive chunk.")
            
            full_data_table = df.to_markdown(index=False)
            
            # Create the base description from our parts
            base_description = "\n\n".join(description_parts)
            
            combined_text = f"""
{base_description}

#### Statistical Summary
{statistical_summary}

#### Full Experimental Data ({total_rows} rows)
{full_data_table}
            """.strip()

            single_chunk = {
                'text': combined_text,
                'metadata': {
                    'source': excel_path,
                    'context_source': context_path,
                    'content_type': 'dataset_package', 
                    'page': 1 
                }
            }
            all_chunks.append(single_chunk)
            print(f"    - ✅ Created 1 'dataset_package' chunk.")

        else:
            # --- STRATEGY B: Large File (Summary + Data Chunks) ---
            print(f"    - File is large ({total_rows} rows). Creating summary + batched data chunks.")
            
            # 4.1 Create the "Summary Chunk"
            
            # Create the base description from our parts
            base_description = "\n\n".join(description_parts)
            
            summary_text = f"""
{base_description}

#### Statistical Summary of {total_rows} Rows
{statistical_summary}
            """.strip()

            summary_chunk = {
                'text': summary_text,
                'metadata': {
                    'source': excel_path,
                    'context_source': context_path,
                    'content_type': 'dataset_summary',
                    'page': 1 
                }
            }
            all_chunks.append(summary_chunk)
            print(f"    - ✅ Created 1 'dataset_summary' chunk.")

            # 4.2 Create "Data Chunks" by batching rows
            num_batches = 0
            for i in range(0, total_rows, row_chunk_size):
                df_batch = df.iloc[i : i + row_chunk_size]
                markdown_table = df_batch.to_markdown(index=False)
                
                # We use the title (which has a fallback) for the header
                chunk_text = f"""
### {title}
#### Data Rows {i + 1} to {i + len(df_batch)}

{markdown_table}
                """.strip()
                
                data_chunk = {
                    'text': chunk_text,
                    'metadata': {
                        'source': excel_path,
                        'context_source': context_path,
                        'content_type': 'data_rows',
                        'start_row': i + 1,
                        'end_row': i + len(df_batch),
                        'page': 1 
                    }
                }
                all_chunks.append(data_chunk)
                num_batches += 1
            
            print(f"    - ✅ Created {num_batches} 'data_rows' chunks (batch size: {row_chunk_size}).")
        
        print(f"    - ✅ Successfully created {len(all_chunks)} total chunks for {Path(excel_path).name}")
        return all_chunks

    except FileNotFoundError as e:
        print(f"    - ❌ Error: File not found - {e}")
        return []
    except json.JSONDecodeError:
        print(f"    - ❌ Error: Invalid JSON in file '{context_path}'")
        return []
    except Exception as e:
        print(f"    - ❌ Error processing data pair for '{excel_path}': {e}")
        return []