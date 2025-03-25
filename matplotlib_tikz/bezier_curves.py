#!/usr/bin/env python3
"""
Bezier Curves Implementation
Based on extracted code from Bezier.html
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import os

# Create directory for plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

def B(i, N, t):
    """Bernstein polynomial basis function."""
    # Ensure t is array-like for vectorized operations
    t = np.asarray(t)
    val = comb(N, i) * t**i * (1.-t)**(N-i)
    return val

def plot_bernstein_polynomials(N=7):
    """Plot Bernstein polynomials of degree N."""
    tt = np.linspace(0, 1, 200)
    plt.figure(figsize=(10, 6))
    
    for i in range(N+1):
        plt.plot(tt, B(i, N, tt), label=f'B({i},{N})')
    
    plt.title(f'Bernstein Polynomials of Degree {N}')
    plt.xlabel('t')
    plt.ylabel('B(i,N,t)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('plots/bernstein_polynomials.png', dpi=300, bbox_inches='tight')
    plt.close()

def P(t, X):
    """
    Evaluates a Bezier curve for the points in X.
    
    Inputs:
      X is a list (or array) or 2D coords
      t is a number (or list of numbers) in [0,1] where you want to
        evaluate the Bezier curve
      
    Output:
      xx is the set of 2D points along the Bezier curve
    """
    X = np.array(X)
    N, d = np.shape(X)   # Number of points, Dimension of points
    N = N - 1
    
    # Ensure t is array-like for vectorized operations
    t = np.asarray(t)
    if t.ndim == 0:
        t = np.array([t])
    
    xx = np.zeros((len(t), d))
    
    for i in range(N+1):
        xx += np.outer(B(i, N, t), X[i])
    
    return xx

def plot_bezier_curve(control_points=None):
    """Plot a Bezier curve with given control points."""
    if control_points is None:
        # Default control points if none provided
        control_points = [
            (0.09375, 0.15297),
            (0.54911, 0.16488),
            (0.70833, 0.61429),
            (0.52827, 0.89405),
            (0.24405, 0.87768),
            (0.15327, 0.63214),
            (0.58036, 0.08304),
            (0.88393, 0.28988)
        ]
    
    X = np.array(control_points)
    tt = np.linspace(0, 1, 200)
    xx = P(tt, X)
    
    plt.figure(figsize=(8, 8))
    plt.plot(xx[:, 0], xx[:, 1], 'b-', linewidth=2, label='Bezier Curve')
    plt.plot(X[:, 0], X[:, 1], 'ro-', alpha=0.5, label='Control Points')
    
    # Add annotations for control points
    for i, point in enumerate(X):
        plt.annotate(f'P{i}', (point[0], point[1]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title('Bezier Curve with Control Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.savefig('plots/bezier_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_bezier_construction(t_values=None):
    """Plot the construction of a Bezier curve at specific parameter values."""
    if t_values is None:
        t_values = [0.5]  # Default to showing construction at t=0.5
        
    control_points = [
        (0, 0),
        (0.3, 0.8),
        (0.7, 0.8),
        (1, 0)
    ]
    X = np.array(control_points)
    
    # Plot each value of t separately
    for t_idx, t in enumerate(t_values):
        plt.figure(figsize=(8, 8))
        
        # Draw control polygon
        plt.plot(X[:, 0], X[:, 1], 'ro-', alpha=0.5, label='Control Polygon')
        
        # Evaluate curve for all t values
        curve_t = np.linspace(0, 1, 100)
        curve_points = P(curve_t, X)
        plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=1, label='Bezier Curve')
        
        # Highlight the point at parameter t
        point_at_t = P(np.array([t]), X)[0]
        plt.plot(point_at_t[0], point_at_t[1], 'go', markersize=8, label=f'Point at t={t}')
        
        # De Casteljau's algorithm visualization (recursive blending)
        points = X.copy()
        levels = len(X) - 1
        colors = plt.cm.viridis(np.linspace(0, 1, levels))
        
        for level in range(levels):
            new_points = []
            for j in range(len(points) - 1):
                p1 = points[j]
                p2 = points[j + 1]
                # Linear interpolation
                new_point = (1 - t) * p1 + t * p2
                new_points.append(new_point)
                
                # Draw line between points being interpolated
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=colors[level], alpha=0.7, linewidth=1)
            
            # Plot the intermediate points
            new_points = np.array(new_points)
            plt.plot(new_points[:, 0], new_points[:, 1], 'o', color=colors[level], 
                   alpha=0.7, markersize=5, label=f'Level {level+1}' if level == 0 else "")
            
            points = new_points
        
        plt.title(f'Bezier Curve Construction at t = {t}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        
        plt.savefig(f'plots/bezier_construction_t{t:.1f}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Run the examples
plot_bernstein_polynomials()
plot_bezier_curve()
plot_bezier_construction([0.0, 0.3, 0.5, 0.7, 1.0])

print("Plots generated in 'plots' directory:")
for i, plot_file in enumerate(sorted(os.listdir('plots'))):
    if plot_file.startswith('bezier') or plot_file.startswith('bernstein'):
        print(f"{i+1}. plots/{plot_file}")