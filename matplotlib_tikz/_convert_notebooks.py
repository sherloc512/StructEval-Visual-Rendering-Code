#!/usr/bin/env python3
import json
import os
import sys
import glob

def convert_notebook_to_py(notebook_path):
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Create Python file path
        py_file_path = os.path.splitext(notebook_path)[0] + '.py'
        
        # Create backup of original notebook
        backup_path = notebook_path + '.bak'
        os.rename(notebook_path, backup_path)
        
        # Extract code cells and write to Python file
        with open(py_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Converted from: {os.path.basename(notebook_path)}\n\n")
            
            for i, cell in enumerate(notebook.get('cells', [])):
                if cell.get('cell_type') == 'code':
                    # Get source code from the cell
                    source = ''.join(cell.get('source', []))
                    
                    # If not empty, write to file with cell markers
                    if source.strip():
                        f.write(f"# Cell {i+1}\n")
                        f.write(source)
                        # Add newline if needed
                        if not source.endswith('\n'):
                            f.write('\n')
                        f.write('\n')
                elif cell.get('cell_type') == 'markdown':
                    # Include markdown as comments
                    source = ''.join(cell.get('source', []))
                    if source.strip():
                        f.write(f"# Markdown Cell {i+1}\n")
                        for line in source.splitlines():
                            f.write(f"# {line}\n")
                        f.write('\n')
        
        print(f"Converted {notebook_path} to {py_file_path}")
        return True
    except Exception as e:
        print(f"Error converting {notebook_path}: {e}")
        return False

def main():
    # Get all ipynb files
    ipynb_files = glob.glob('/mnt/ubuntu_hdd/open_source/code/struct_eval/llm/matplotlib_tikz/**/*.ipynb', recursive=True)
    
    success_count = 0
    failure_count = 0
    
    for notebook_path in ipynb_files:
        if convert_notebook_to_py(notebook_path):
            success_count += 1
        else:
            failure_count += 1
    
    print(f"\nConversion complete. Successfully converted: {success_count}, Failed: {failure_count}")

if __name__ == "__main__":
    main()