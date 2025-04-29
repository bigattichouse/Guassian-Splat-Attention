"""
HSA Splat Monitor Tool

This script monitors HSA splat counts during training by analyzing checkpoint files.
It can be used to track adaptation behavior and diagnose issues with splat births and deaths.

Save as monitor_splats.py and run:
python monitor_splats.py
"""

import os
import sys
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
OUTPUT_DIR = "outputs"
MODEL_NAME = "hsa-tinystories"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, MODEL_NAME, "checkpoints")

def find_checkpoints():
    """Find all checkpoint files in order."""
    checkpoints = []
    
    # Check if directory exists
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return []
    
    # Find step checkpoints
    step_checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "step_*.pt"))
    for path in step_checkpoints:
        # Only include checkpoints that were successfully saved
        if os.path.exists(f"{path}.success"):
            try:
                step = int(os.path.basename(path).split("_")[1].split(".")[0])
                checkpoints.append((step, "step", path))
            except:
                continue
    
    # Find epoch checkpoints
    epoch_checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt"))
    for path in epoch_checkpoints:
        if os.path.exists(f"{path}.success"):
            try:
                epoch = int(os.path.basename(path).split("_")[1].split(".")[0])
                # Multiply by large number to sort after steps
                checkpoints.append((epoch * 10000, "epoch", path))
            except:
                continue
    
    # Sort by step/epoch number
    checkpoints.sort()
    return [(type, path) for _, type, path in checkpoints]

def extract_hsa_data(checkpoint_path):
    """Extract HSA data from a checkpoint file."""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract metadata
        step = checkpoint.get("step", 0)
        epoch = checkpoint.get("epoch", 0)
        loss = checkpoint.get("loss", float('inf'))
        
        # Extract HSA data if available
        if "hsa_data" not in checkpoint:
            print(f"No HSA data found in checkpoint: {checkpoint_path}")
            return None
        
        hsa_data = checkpoint["hsa_data"]
        
        # Extract splat counts by level
        if "splats" in hsa_data and "hierarchy" in hsa_data:
            splats = hsa_data["splats"]
            hierarchy = hsa_data["hierarchy"]
            
            # Count splats by level
            counts_by_level = defaultdict(int)
            for splat in splats:
                level = splat["level"]
                counts_by_level[level] += 1
            
            # Extract amplitude statistics by level
            amplitude_stats = {}
            for level in hierarchy["levels"]:
                level_splats = [s for s in splats if s["level"] == level]
                if level_splats:
                    amplitudes = [s["amplitude"] for s in level_splats]
                    amplitude_stats[level] = {
                        "min": min(amplitudes),
                        "max": max(amplitudes),
                        "mean": sum(amplitudes) / len(amplitudes),
                        "at_risk": sum(1 for a in amplitudes if a < 0.01)
                    }
            
            return {
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "type": os.path.basename(checkpoint_path).split("_")[0],
                "total_splats": len(splats),
                "counts_by_level": dict(counts_by_level),
                "init_counts": hierarchy["init_splats_per_level"],
                "hierarchy_levels": hierarchy["levels"],
                "amplitude_stats": amplitude_stats
            }
            
        return None
    except Exception as e:
        print(f"Error extracting HSA data from {checkpoint_path}: {e}")
        return None

def analyze_checkpoints():
    """Analyze all checkpoints and collect HSA data."""
    print("Analyzing HSA splats across checkpoints...")
    
    # Find all checkpoints
    checkpoints = find_checkpoints()
    if not checkpoints:
        print("No checkpoints found.")
        return None
    
    print(f"Found {len(checkpoints)} checkpoints.")
    
    # Analyze each checkpoint
    data_points = []
    for checkpoint_type, path in checkpoints:
        print(f"Analyzing {os.path.basename(path)}...")
        data = extract_hsa_data(path)
        if data:
            data_points.append(data)
    
    print(f"Successfully analyzed {len(data_points)} checkpoints with HSA data.")
    return data_points

def plot_splat_trends(data_points):
    """Plot trends in splat counts over time."""
    if not data_points:
        print("No data points to plot.")
        return
    
    # Make sure data is sorted by step
    data_points.sort(key=lambda x: x["step"])
    
    # Extract steps and total counts
    steps = [d["step"] for d in data_points]
    total_counts = [d["total_splats"] for d in data_points]
    
    # Get all levels from hierarchy
    levels = data_points[0]["hierarchy_levels"]
    initial_counts = data_points[0]["init_counts"]
    level_counts = {level: [] for level in levels}
    
    # Extract counts by level
    for d in data_points:
        for level in levels:
            level_counts[level].append(d["counts_by_level"].get(level, 0))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot total splats
    plt.subplot(2, 1, 1)
    plt.plot(steps, total_counts, 'k-', label='Total Splats', linewidth=2)
    plt.axhline(y=sum(initial_counts), color='r', linestyle='--', label='Initial Total')
    plt.xlabel('Training Steps')
    plt.ylabel('Number of Splats')
    plt.title('Total Splat Count Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot counts by level
    plt.subplot(2, 1, 2)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i, level in enumerate(levels):
        plt.plot(steps, level_counts[level], f'{colors[i % len(colors)]}-', 
                 label=f'{level} Level', linewidth=2)
        plt.axhline(y=initial_counts[i], color=colors[i % len(colors)], 
                    linestyle='--', alpha=0.5)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Number of Splats')
    plt.title('Splat Counts by Hierarchy Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, MODEL_NAME, "splat_trends.png"))
    print(f"Saved plot to {os.path.join(OUTPUT_DIR, MODEL_NAME, 'splat_trends.png')}")
    
    # Show plot
    plt.show()

def analyze_birth_death_rates(data_points):
    """Analyze birth and death rates between checkpoints."""
    if len(data_points) < 2:
        print("Need at least 2 checkpoints to analyze birth/death rates.")
        return
    
    # Make sure data is sorted by step
    data_points.sort(key=lambda x: x["step"])
    
    # Create arrays to store changes
    steps = []
    net_changes = []
    changes_by_level = {level: [] for level in data_points[0]["hierarchy_levels"]}
    
    # Calculate changes between consecutive checkpoints
    for i in range(1, len(data_points)):
        prev = data_points[i-1]
        curr = data_points[i]
        
        steps.append(curr["step"])
        
        # Calculate net change
        prev_total = prev["total_splats"]
        curr_total = curr["total_splats"]
        net_change = curr_total - prev_total
        net_changes.append(net_change)
        
        # Calculate changes by level
        for level in changes_by_level.keys():
            prev_count = prev["counts_by_level"].get(level, 0)
            curr_count = curr["counts_by_level"].get(level, 0)
            level_change = curr_count - prev_count
            changes_by_level[level].append(level_change)
    
    # Create summary of changes
    print("\nSplat Change Analysis:")
    print("=====================")
    
    total_increase = sum(max(0, change) for change in net_changes)
    total_decrease = abs(sum(min(0, change) for change in net_changes))
    
    print(f"Total splats created: {total_increase}")
    print(f"Total splats removed: {total_decrease}")
    print(f"Net change: {total_increase - total_decrease}")
    
    # Show level breakdown
    for level in changes_by_level.keys():
        level_increase = sum(max(0, change) for change in changes_by_level[level])
        level_decrease = abs(sum(min(0, change) for change in changes_by_level[level]))
        print(f"\nLevel {level}:")
        print(f"  Created: {level_increase}")
        print(f"  Removed: {level_decrease}")
        print(f"  Net change: {level_increase - level_decrease}")
    
    # Calculate birth and death rates over time
    print("\nChange rates (positive = births, negative = deaths):")
    for i, step in enumerate(steps):
        print(f"Step {step}: Net change = {net_changes[i]}", end="")
        if net_changes[i] > 0:
            print(f" (Birth event: +{net_changes[i]})")
        elif net_changes[i] < 0:
            print(f" (Death event: {net_changes[i]})")
        else:
            print(" (No change)")
    
    # Plot changes over time
    plt.figure(figsize=(12, 8))
    
    # Plot net changes
    plt.subplot(2, 1, 1)
    plt.bar(steps, net_changes, color=['g' if x > 0 else 'r' for x in net_changes])
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Net Change in Splat Count')
    plt.title('Net Splat Changes Between Checkpoints')
    plt.grid(True, alpha=0.3)
    
    # Plot changes by level
    plt.subplot(2, 1, 2)
    width = 0.2
    x = np.arange(len(steps))
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for i, (level, changes) in enumerate(changes_by_level.items()):
        plt.bar(x + width*i - width*len(changes_by_level)/2, changes, 
                width=width, label=level, color=colors[i % len(colors)])
    
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.xticks(x, steps)
    plt.xlabel('Training Steps')
    plt.ylabel('Change in Splat Count')
    plt.title('Splat Changes by Level Between Checkpoints')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(OUTPUT_DIR, MODEL_NAME, "splat_changes.png"))
    print(f"Saved plot to {os.path.join(OUTPUT_DIR, MODEL_NAME, 'splat_changes.png')}")
    
    # Show plot
    plt.show()

def analyze_current_state(data_points):
    """Analyze the current state of HSA based on the latest checkpoint."""
    if not data_points:
        print("No data points to analyze.")
        return
    
    # Get the latest data point
    latest = max(data_points, key=lambda x: x["step"])
    
    print("\nCurrent HSA State Analysis (Latest Checkpoint):")
    print("==============================================")
    print(f"Step: {latest['step']}, Epoch: {latest['epoch']}")
    print(f"Total splats: {latest['total_splats']}")
    
    # Compare with initial configuration
    initial_total = sum(latest["init_counts"])
    ratio = latest["total_splats"] / initial_total
    print(f"Ratio to initial count: {ratio:.2f}")
    
    if ratio < 0.5:
        print("WARNING: Total splat count is less than 50% of initial count!")
    
    # Analyze by level
    print("\nBreakdown by Level:")
    for i, level in enumerate(latest["hierarchy_levels"]):
        init_count = latest["init_counts"][i]
        current_count = latest["counts_by_level"].get(level, 0)
        level_ratio = current_count / init_count
        
        status = "OK"
        if level_ratio < 0.3:
            status = "CRITICAL - Severe underpopulation"
        elif level_ratio < 0.5:
            status = "WARNING - Underpopulation"
        elif level_ratio < 0.7:
            status = "CAUTION - Birth needed"
        
        print(f"Level {level}: {current_count}/{init_count} splats ({level_ratio:.2f}) - {status}")
        
        # Show amplitude stats if available
        if "amplitude_stats" in latest and level in latest["amplitude_stats"]:
            amp_stats = latest["amplitude_stats"][level]
            print(f"  Amplitudes: min={amp_stats['min']:.4f}, max={amp_stats['max']:.4f}, " +
                  f"mean={amp_stats['mean']:.4f}")
            
            if amp_stats["at_risk"] > 0:
                print(f"  WARNING: {amp_stats['at_risk']} splats at risk of death (low amplitude)")
    
    # Generate recommendations
    print("\nRecommendations:")
    if ratio < 0.7:
        print("- Reduce death threshold significantly (try 0.0005 or lower)")
        print("- Increase adaptation frequency to create more opportunities for births")
        print("- Modify birth mechanism to be more aggressive (see fix_birth_mechanism.py)")
    
    level_problems = []
    for i, level in enumerate(latest["hierarchy_levels"]):
        init_count = latest["init_counts"][i]
        current_count = latest["counts_by_level"].get(level, 0)
        level_ratio = current_count / init_count
        
        if level_ratio < 0.5:
            level_problems.append(level)
    
    if level_problems:
        print(f"- Focus on fixing birth for these levels: {', '.join(level_problems)}")
        print("- Increase initial splat counts for these levels")
        print("- Review level-specific parameters like min_distance_threshold")
    
    # Check for amplitude issues
    has_amplitude_risk = False
    for level in latest["hierarchy_levels"]:
        if "amplitude_stats" in latest and level in latest["amplitude_stats"]:
            if latest["amplitude_stats"][level]["at_risk"] > 0:
                has_amplitude_risk = True
    
    if has_amplitude_risk:
        print("- Increase minimum amplitude threshold to prevent deaths")
        print("- Implement progressive amplitude reduction instead of immediate death")

def main():
    """Main entry point for the script."""
    print("HSA Splat Monitor Tool")
    print("=====================")
    
    # Analyze checkpoints
    data_points = analyze_checkpoints()
    if not data_points:
        print("No data available for analysis.")
        return
    
    # Plot trends
    plot_splat_trends(data_points)
    
    # Analyze birth/death rates
    analyze_birth_death_rates(data_points)
    
    # Analyze current state
    analyze_current_state(data_points)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
