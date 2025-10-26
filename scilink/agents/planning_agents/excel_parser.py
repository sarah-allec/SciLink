# planning_agents/excel_parser.py
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List

# If a file has this many rows or fewer, we embed it all in one chunk.
# This is more efficient for small, preliminary results.
SMALL_FILE_THRESHOLD = 150

def parse_adaptive_excel(excel_path: str, context_path: str, row_chunk_size: int = 100) -> List[Dict[str, Any]]:
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
        
        if not all(k in context for k in ["objective", "column_definitions"]):
            print(f"    - ⚠️  Skipping: JSON '{context_path}' is missing 'objective' or 'column_definitions'.")
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
        experiment_title = context.get('experiment_title', Path(excel_path).stem)
        col_defs = "\n".join([f"- `{col}`: {desc}" for col, desc in context['column_definitions'].items()])
        statistical_summary = df.describe().to_markdown() if not df.empty else "No statistical summary available."

        # --- 4. Adaptive Chunking Logic ---
        
        if total_rows <= SMALL_FILE_THRESHOLD:
            # --- STRATEGY A: Small File (One Rich Chunk) ---
            print(f"    - File is small ({total_rows} rows). Creating one single, comprehensive chunk.")
            
            full_data_table = df.to_markdown(index=False)
            
            combined_text = f"""
### Experiment: {experiment_title}
#### Objective
{context['objective']}

#### Data Column Definitions
{col_defs}

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
                    'content_type': 'dataset_package', # New type for small, whole datasets
                    'page': 1 
                }
            }
            all_chunks.append(single_chunk)
            print(f"    - ✅ Created 1 'dataset_package' chunk.")

        else:
            # --- STRATEGY B: Large File (Summary + Data Chunks) ---
            print(f"    - File is large ({total_rows} rows). Creating summary + batched data chunks.")
            
            # 4.1 Create the "Summary Chunk"
            summary_text = f"""
### Experiment: {experiment_title}
#### Objective
{context['objective']}

#### Data Column Definitions
{col_defs}

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
                
                chunk_text = f"""
### {experiment_title}
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