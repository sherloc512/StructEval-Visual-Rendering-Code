#!/usr/bin/env python
import os
import sys
import glob
from html2markdown import convert

def html_to_markdown(html_path, output_path=None):
    """Convert HTML file to Markdown and save to output_path"""
    if output_path is None:
        # Replace .html extension with .md
        output_path = os.path.splitext(html_path)[0] + '.md'
    
    # Read HTML content
    with open(html_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Convert HTML to Markdown
    md_content = convert(html_content)
    
    # Write Markdown content
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(md_content)
    
    print(f"Converted {html_path} to {output_path}")
    return output_path

def main():
    # Find all HTML files
    html_files = glob.glob('/mnt/ubuntu_hdd/open_source/code/struct_eval/llm/matplotlib_tikz/*.html')
    
    # Convert each HTML file to Markdown
    for html_file in html_files:
        html_to_markdown(html_file)

if __name__ == '__main__':
    main()