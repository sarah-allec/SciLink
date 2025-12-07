import logging

import os
from typing import List, Dict, Any
from pathlib import Path

# Match these to the extensions you check in planning_agent.py
SUPPORTED_EXTENSIONS = {
    '.py', '.java', '.r', '.cpp', '.h', '.js', '.json', 
    '.csv', '.txt', '.md', '.pdf'
}

def get_files_from_directory(directory_path: str) -> List[str]:
    """
    Recursively finds all supported files in a directory, ignoring hidden files.
    """
    found_files = []
    path = Path(directory_path)
    
    if not path.exists():
        print(f"  - ⚠️ Directory not found: {directory_path}")
        return []

    print(f"  - 📂 Scanning directory: {path.name}...")

    for root, dirs, files in os.walk(path):
        # In-place modification to skip hidden dirs and common junk
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'venv', 'env', 'node_modules', '.git')]
        
        for file in files:
            if file.startswith('.'): continue
            
            file_path = Path(root) / file
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                found_files.append(str(file_path))
                
    print(f"    -> Found {len(found_files)} files in directory.")
    return found_files

def generate_repo_map(root_dir: str) -> str:
    """
    Generates a visual tree structure of the repository.
    Useful for giving the LLM context on where files live for imports.
    """
    root = Path(root_dir)
    if not root.exists(): return ""

    tree_lines = [f"{root.name}/"]
    
    for path in sorted(root.rglob('*')):
        # Skip hidden files/dirs
        if any(part.startswith('.') or part in ('__pycache__', 'venv', 'env') for part in path.parts):
            continue
        
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            rel_path = path.relative_to(root)
            depth = len(rel_path.parts)
            indent = '    ' * (depth - 1)
            tree_lines.append(f"{indent}├── {path.name}")
            
    return "\n".join(tree_lines)

def table_to_markdown(table: List[List[str]]) -> str:
    """Converts a 2D list representation of a table into Markdown format."""
    if not table or not table[0]: return ""
    # Ensure all cells are strings before joining
    cleaned_table = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]
    header, *rows = cleaned_table
    md = f"| {' | '.join(header)} |\n| {' | '.join(['---'] * len(header))} |\n"
    for row in rows:
        # Pad rows that are shorter than the header
        while len(row) < len(header): row.append("")
        # Truncate rows that are longer than the header
        md += f"| {' | '.join(row[:len(header)])} |\n"
    return md


def write_experiments_to_disk(result_json: Dict[str, Any], target_dir: str) -> List[str]:
    """
    Parses the result JSON and writes 'implementation_code' to .py files in the target directory.
    Returns a list of filenames that were successfully saved.
    """
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    experiments = result_json.get("proposed_experiments", [])
    saved_files = []
    
    if not experiments:
        logging.warning(f"No experiments found to save in {target_dir}")
        return []
    
    for i, exp in enumerate(experiments):
        code_content = exp.get("implementation_code")
        exp_name = exp.get("experiment_name", f"Experiment_{i+1}")
        
        # 1. Clean filename
        # Replace spaces with underscores and remove non-alphanumeric chars (except _ and .)
        safe_name = "".join(c for c in exp_name if c.isalnum() or c in (' ', '_', '.')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        # Fallback if name becomes empty after cleaning
        if not safe_name: 
            safe_name = f"experiment_code_{i+1}"
            
        filename = f"{safe_name}.py"
        file_path = path / filename

        # 2. Extract and Write
        if code_content and "No relevant code found" not in code_content:
            try:
                # Strip markdown code blocks (```python ... ```)
                code_lines = code_content.splitlines()
                
                # Logic to find the content between the backticks
                start_index = next((j for j, line in enumerate(code_lines) if line.strip().startswith('```')), -1)
                end_index = next((j for j, line in enumerate(code_lines[start_index+1:]) if line.strip().endswith('```')), -1)
                
                if start_index != -1 and end_index != -1:
                    # Adjust end_index because we sliced the list
                    actual_end = start_index + 1 + end_index
                    extracted_code = "\n".join(code_lines[start_index + 1 : actual_end]).strip()
                else:
                    # Fallback: assume the whole string is code if no backticks found
                    extracted_code = code_content.strip()

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_code)
                
                saved_files.append(filename)
                
            except Exception as e:
                logging.error(f"Failed to write {filename}: {e}")
        else:
            logging.info(f"Experiment {i+1} ('{exp_name}') has no executable code.")

    return saved_files