#!/usr/bin/env python
import os
import re
import glob

def extract_python_code_from_markdown(md_path, output_path=None):
    """Extract Python code blocks from Markdown file and save to a Python file"""
    if output_path is None:
        # Replace .md extension with .py
        output_path = os.path.splitext(md_path)[0] + '_extracted.py'
    
    # Read Markdown content
    with open(md_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    
    # Extract Python code blocks
    # Look for ```python ... ``` blocks
    code_blocks = re.findall(r'```python(.*?)```', md_content, re.DOTALL)
    
    if not code_blocks:
        print(f"No Python code blocks found in {md_path}")
        return None
    
    # Prepare the Python file with header comment and imports
    python_content = f"""#!/usr/bin/env python
# Extracted from {os.path.basename(md_path)}

import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for saved plots
os.makedirs('plots', exist_ok=True)

"""
    
    # Add each code block with a separator and modify to save figures
    for i, block in enumerate(code_blocks):
        code = block.strip()
        
        # Check if the code block includes plt.show() or plt.savefig()
        if 'plt.show()' in code and 'plt.savefig(' not in code:
            # Replace plt.show() with plt.savefig() and plt.close()
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            plot_name = f"plots/{base_name}_plot_{i+1}.png"
            code = code.replace('plt.show()', f'plt.savefig("{plot_name}", dpi=300, bbox_inches="tight")\nplt.close()')
        
        # Add the modified code block
        python_content += f"\n# Code Block {i+1}\n"
        python_content += code
        python_content += "\n\n"
    
    # Add a list of generated plot files at the end
    python_content += """
print("\\nPlots generated:")
for i, plot_file in enumerate(sorted(glob.glob('plots/*.png'))):
    print(f"{i+1}. {plot_file}")
"""
    
    # Write Python content
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(python_content)
    
    print(f"Extracted Python code from {md_path} to {output_path}")
    return output_path

def main():
    # Find all Markdown files
    md_files = glob.glob('/mnt/ubuntu_hdd/open_source/code/struct_eval/llm/matplotlib_tikz/*.md')
    
    # Extract Python code from each Markdown file
    for md_file in md_files:
        extract_python_code_from_markdown(md_file)

if __name__ == '__main__':
    main()