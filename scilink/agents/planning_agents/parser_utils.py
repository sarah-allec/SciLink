from typing import List

import os
from pathlib import Path
from typing import List

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