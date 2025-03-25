#!/usr/bin/env python3
"""
Stem-Leaf Plots in Python: A Visual Representation of Data
Converted from Markdown file
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import glob

# Create directory for plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Basic Stem-Leaf Plot
def stem_leaf_plot(data):
    # Sort the data
    sorted_data = sorted(data)
    
    # Extract stems and leaves
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    # Create a dictionary to store leaves for each stem
    stem_leaf_dict = {}
    for stem, leaf in zip(stems, leaves):
        if stem in stem_leaf_dict:
            stem_leaf_dict[stem].append(leaf)
        else:
            stem_leaf_dict[stem] = [leaf]
    
    # Plot the stem-leaf plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for stem, leaves in stem_leaf_dict.items():
        ax.text(0, int(stem), f"{stem} | {''.join(leaves)}")
    
    ax.set_ylim(min(map(int, stems))-1, max(map(int, stems))+1)
    ax.set_axis_off()
    plt.title("Stem-Leaf Plot")
    plt.savefig('plots/stemleaf_basic.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30]
stem_leaf_plot(data)

# Create and print stem-leaf plot
def create_stem_leaf(data):
    # Sort the data
    sorted_data = sorted(data)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    # Create a dictionary to store leaves for each stem
    stem_leaf_dict = {}
    for stem, leaf in zip(stems, leaves):
        if stem in stem_leaf_dict:
            stem_leaf_dict[stem].append(leaf)
        else:
            stem_leaf_dict[stem] = [leaf]
    
    # Print the stem-leaf plot
    for stem, leaves in stem_leaf_dict.items():
        print(f"{stem} | {''.join(leaves)}")

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30]
print("Basic Printed Stem-Leaf Plot:")
create_stem_leaf(data)

# Basic stem-leaf with sorted stems
def basic_stem_leaf(data):
    # Sort the data
    sorted_data = sorted(data)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    # Print the stem-leaf plot
    unique_stems = sorted(set(stems))
    for stem in unique_stems:
        print(f"{stem} | ", end="")
        for i, s in enumerate(stems):
            if s == stem:
                print(leaves[i], end="")
        print()

# Example data
print("\nSorted Stem-Leaf Plot:")
basic_stem_leaf(data)

# Interpreting a Stem-Leaf Plot
def interpret_stem_leaf(data):
    # Calculate basic statistics
    median = np.median(data)
    mode = max(set(data), key=data.count)
    data_range = max(data) - min(data)
    
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Range: {data_range}")
    
    # Create and print the stem-leaf plot
    basic_stem_leaf(data)

# Example data
data_with_duplicates = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30, 20, 28]
print("\nStem-Leaf Plot with Statistics:")
interpret_stem_leaf(data_with_duplicates)

# Back-to-Back Stem-Leaf Plots
def back_to_back_stem_leaf(data1, data2):
    # Combine and sort both datasets
    all_data = sorted(data1 + data2)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in all_data]
    leaves1 = [str(x)[-1] if x in data1 else '' for x in all_data]
    leaves2 = [str(x)[-1] if x in data2 else '' for x in all_data]
    
    # Print the back-to-back stem-leaf plot
    unique_stems = sorted(set(stems), reverse=True)
    for stem in unique_stems:
        left_leaves = ''.join(leaves1[i] for i, s in enumerate(stems) if s == stem)
        right_leaves = ''.join(leaves2[i] for i, s in enumerate(stems) if s == stem)
        print(f"{left_leaves:>10} | {stem} | {right_leaves}")
    
    # Plot version
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = 0
    for stem in unique_stems:
        left_leaves = ''.join(leaves1[i] for i, s in enumerate(stems) if s == stem)
        right_leaves = ''.join(leaves2[i] for i, s in enumerate(stems) if s == stem)
        ax.text(0.4, y_pos, f"{left_leaves:>10} | {stem} | {right_leaves}")
        y_pos += 1
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, y_pos)
    ax.set_axis_off()
    plt.title("Back-to-Back Stem-Leaf Plot")
    plt.savefig('plots/back_to_back_stemleaf.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example data
data1 = [23, 29, 20, 32, 27, 25, 28]
data2 = [31, 24, 26, 22, 30, 33, 35]
print("\nBack-to-Back Stem-Leaf Plot:")
back_to_back_stem_leaf(data1, data2)

# Handling Large Datasets
def large_dataset_stem_leaf(data, stem_interval=10):
    # Sort the data
    sorted_data = sorted(data)
    
    # Create stems and leaves
    stems = [str(x // stem_interval) for x in sorted_data]
    leaves = [str(x % stem_interval) for x in sorted_data]
    
    # Print the stem-leaf plot
    unique_stems = sorted(set(stems))
    for stem in unique_stems:
        print(f"{int(stem) * stem_interval:3d} | ", end="")
        for i, s in enumerate(stems):
            if s == stem:
                print(f"{leaves[i]:2s}", end=" ")
        print()
    
    # Plot version
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = 0
    for stem in unique_stems:
        stem_text = f"{int(stem) * stem_interval:3d} | "
        for i, s in enumerate(stems):
            if s == stem:
                stem_text += f"{leaves[i]:2s} "
        ax.text(0, y_pos, stem_text)
        y_pos += 1
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, y_pos)
    ax.set_axis_off()
    plt.title("Large Dataset Stem-Leaf Plot")
    plt.savefig('plots/large_dataset_stemleaf.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example data (larger dataset)
print("\nLarge Dataset Stem-Leaf Plot:")
random.seed(42)  # For reproducibility
data_large = [random.randint(0, 199) for _ in range(50)]
large_dataset_stem_leaf(data_large)

# Stem-Leaf Plot with Outlier Detection
def stem_leaf_with_outliers(data):
    # Calculate Q1, Q3, and IQR
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Sort the data
    sorted_data = sorted(data)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    # Print the stem-leaf plot with outlier detection
    unique_stems = sorted(set(stems))
    for stem in unique_stems:
        print(f"{stem} | ", end="")
        for i, s in enumerate(stems):
            if s == stem:
                if lower_bound <= sorted_data[i] <= upper_bound:
                    print(leaves[i], end="")
                else:
                    print(f"*{leaves[i]}*", end="")  # Mark outliers with asterisks
        print()
    
    # Plot version
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = 0
    for stem in unique_stems:
        stem_text = f"{stem} | "
        for i, s in enumerate(stems):
            if s == stem:
                if lower_bound <= sorted_data[i] <= upper_bound:
                    stem_text += leaves[i]
                else:
                    stem_text += f"*{leaves[i]}*"  # Mark outliers with asterisks
        ax.text(0, y_pos, stem_text)
        y_pos += 1
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, y_pos)
    ax.set_axis_off()
    plt.title("Stem-Leaf Plot with Outlier Detection")
    plt.savefig('plots/stemleaf_outliers.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example data with outliers
print("\nStem-Leaf Plot with Outlier Detection:")
data_with_outliers = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30, 15, 40]
stem_leaf_with_outliers(data_with_outliers)

# Colored Stem-Leaf Plot
def colored_stem_leaf(data):
    # Sort the data
    sorted_data = sorted(data)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    # Create a dictionary to store leaves for each stem
    stem_leaf_dict = {}
    for stem, leaf in zip(stems, leaves):
        if stem in stem_leaf_dict:
            stem_leaf_dict[stem].append(leaf)
        else:
            stem_leaf_dict[stem] = [leaf]
    
    # Plot the colored stem-leaf plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(stem_leaf_dict)))
    
    for i, (stem, leaves) in enumerate(stem_leaf_dict.items()):
        ax.text(0, int(stem), f"{stem} | {''.join(leaves)}", color=colors[i])
    
    ax.set_ylim(min(map(int, stems))-1, max(map(int, stems))+1)
    ax.set_axis_off()
    plt.title("Colored Stem-Leaf Plot")
    plt.savefig('plots/stemleaf_colored.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30]
colored_stem_leaf(data)

# Student Heights Stem-Leaf Plot
def student_heights_stem_leaf(heights):
    # Sort the heights
    sorted_heights = sorted(heights)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_heights]
    leaves = [str(x)[-1] for x in sorted_heights]
    
    # Plot the stem-leaf plot
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_stems = sorted(set(stems))
    
    for i, stem in enumerate(unique_stems):
        stem_leaves = ''.join([l for s, l in zip(stems, leaves) if s == stem])
        ax.text(0, i, f"{stem} | {stem_leaves}")
    
    ax.set_yticks(range(len(unique_stems)))
    ax.set_yticklabels(unique_stems)
    ax.set_axis_off()
    plt.title("Student Heights (cm) Stem-Leaf Plot")
    plt.savefig('plots/stemleaf_heights.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also print it
    print("\nStudent Heights (cm) Stem-Leaf Plot:")
    for stem in unique_stems:
        print(f"{stem} | ", end="")
        for i, s in enumerate(stems):
            if s == stem:
                print(leaves[i], end="")
        print()

# Example student heights (in cm)
heights = [158, 162, 165, 170, 171, 173, 175, 176, 178, 180, 182, 185]
student_heights_stem_leaf(heights)

# Stem-Leaf Plot vs. Histogram Comparison
def compare_stem_leaf_histogram(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Stem-leaf plot
    sorted_data = sorted(data)
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    unique_stems = sorted(set(stems))
    for i, stem in enumerate(unique_stems):
        stem_leaves = ''.join([l for s, l in zip(stems, leaves) if s == stem])
        ax1.text(0, i, f"{stem} | {stem_leaves}")
    
    ax1.set_yticks(range(len(unique_stems)))
    ax1.set_yticklabels(unique_stems)
    ax1.set_title("Stem-Leaf Plot")
    ax1.set_axis_off()
    
    # Histogram
    ax2.hist(data, bins='auto', edgecolor='black')
    ax2.set_title("Histogram")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig('plots/stemleaf_vs_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30, 20, 28]
compare_stem_leaf_histogram(data)

print("\nPlots generated in 'plots' directory:")
plot_files = glob.glob('plots/stemleaf*.png') + glob.glob('plots/back_to_back*.png') + glob.glob('plots/large_dataset*.png')
for i, plot_file in enumerate(sorted(plot_files)):
    print(f"{i+1}. {plot_file}")