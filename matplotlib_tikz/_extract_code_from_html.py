#!/usr/bin/env python
import os
import re
import glob
from bs4 import BeautifulSoup

def extract_python_code_from_html(html_path, output_path=None):
    """Extract Python code blocks from HTML file and save to a Python file"""
    if output_path is None:
        # Replace .html extension with .py
        output_path = os.path.splitext(html_path)[0] + '_extracted.py'
    
    # Read HTML content
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as file:
        html_content = file.read()
    
    # Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all <pre> or <code> tags that might contain Python code
    code_elements = []
    for element in soup.find_all(['pre', 'code']):
        # Check if contains Python-like code
        if element.text and ('import' in element.text or 'def ' in element.text or 'plt.' in element.text):
            code_elements.append(element.text)
    
    if not code_elements:
        print(f"No Python code blocks found in {html_path}")
        return None
    
    # Prepare the Python file with header comment and imports
    python_content = f"""#!/usr/bin/env python
# Extracted from {os.path.basename(html_path)}

import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for saved plots
os.makedirs('plots', exist_ok=True)

"""
    
    # Add each code block with a separator and modify to save figures
    for i, code in enumerate(code_elements):
        clean_code = code.strip()
        
        # Check if the code block includes plt.show() or plt.savefig()
        if 'plt.show()' in clean_code and 'plt.savefig(' not in clean_code:
            # Replace plt.show() with plt.savefig() and plt.close()
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            plot_name = f"plots/{base_name}_plot_{i+1}.png"
            clean_code = clean_code.replace('plt.show()', f'plt.savefig("{plot_name}", dpi=300, bbox_inches="tight")\nplt.close()')
        
        # Add the modified code block
        python_content += f"\n# Code Block {i+1}\n"
        python_content += clean_code
        python_content += "\n\n"
    
    # Add a list of generated plot files at the end
    python_content += """
import glob
print("\\nPlots generated:")
plot_files = glob.glob('plots/*.png')
for i, plot_file in enumerate(sorted(plot_files)):
    print(f"{i+1}. {plot_file}")
"""
    
    # Write Python content
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(python_content)
    
    print(f"Extracted Python code from {html_path} to {output_path}")
    return output_path

def main():
    # Find all HTML files
    html_files = glob.glob('/mnt/ubuntu_hdd/open_source/code/struct_eval/llm/matplotlib_tikz/*.html')
    
    # Extract Python code from each HTML file
    for html_file in html_files:
        extract_python_code_from_html(html_file)

if __name__ == '__main__':
    main()