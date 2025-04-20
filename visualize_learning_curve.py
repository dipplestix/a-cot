#!/usr/bin/env python3
"""
Visualization script for RL controller learning curves.
Plots accuracy, average steps, rewards, and thinking tokens across epochs.
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize RL controller learning curves")
    parser.add_argument("--summary_file", type=str, required=True,
                      help="Path to summary JSON file")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Directory to save visualizations (default: same directory as summary file)")
    parser.add_argument("--dpi", type=int, default=300,
                      help="DPI for saved figures (default: 300)")
    return parser.parse_args()


def load_summary(summary_file):
    """Load the summary JSON file."""
    with open(summary_file, 'r') as f:
        return json.load(f)


def plot_learning_curve(summary, output_dir, dpi=300):
    """Plot learning curves from the epoch history."""
    epoch_history = summary.get("epoch_history", [])
    
    if not epoch_history:
        print("No epoch history found in summary file.")
        return
    
    # Extract data
    epochs = [e["epoch"] + 1 for e in epoch_history]  # 1-indexed for display
    accuracy = [e["accuracy"] for e in epoch_history]
    avg_steps = [e["avg_steps"] for e in epoch_history]
    avg_reward = [e["avg_reward"] for e in epoch_history]
    avg_tokens = [e.get("avg_thinking_tokens", 0) for e in epoch_history]
    
    # Set up plotting style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"RL Controller Learning Curves - {summary.get('model', 'Unknown Model')}", fontsize=16)
    
    # Plot 1: Accuracy
    ax1 = axes[0, 0]
    ax1.plot(epochs, accuracy, marker='o', linestyle='-', linewidth=2, markersize=8)
    ax1.set_title('Accuracy by Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, max(1.0, max(accuracy) * 1.1)])
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Steps
    ax2 = axes[0, 1]
    ax2.plot(epochs, avg_steps, marker='s', linestyle='-', color='green', linewidth=2, markersize=8)
    ax2.set_title('Average Steps by Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Avg Steps')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average Reward
    ax3 = axes[1, 0]
    ax3.plot(epochs, avg_reward, marker='^', linestyle='-', color='red', linewidth=2, markersize=8)
    ax3.set_title('Average Reward by Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Avg Reward')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average Thinking Tokens
    ax4 = axes[1, 1]
    ax4.plot(epochs, avg_tokens, marker='D', linestyle='-', color='purple', linewidth=2, markersize=8)
    ax4.set_title('Average Thinking Tokens by Epoch')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Avg Tokens')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save the figure
    output_file = Path(output_dir) / "learning_curves.png"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved learning curves to {output_file}")
    
    # Also create individual plots
    metrics = [
        {"name": "accuracy", "values": accuracy, "title": "Accuracy", "color": "blue", "marker": "o"},
        {"name": "avg_steps", "values": avg_steps, "title": "Average Steps", "color": "green", "marker": "s"},
        {"name": "avg_reward", "values": avg_reward, "title": "Average Reward", "color": "red", "marker": "^"},
        {"name": "avg_tokens", "values": avg_tokens, "title": "Average Thinking Tokens", "color": "purple", "marker": "D"}
    ]
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metric["values"], marker=metric["marker"], linestyle='-', 
                 color=metric["color"], linewidth=2, markersize=8)
        plt.title(f'{metric["title"]} by Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric["title"])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = Path(output_dir) / f"{metric['name']}_curve.png"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()


def main():
    args = parse_args()
    
    # Determine output directory
    summary_path = Path(args.summary_file)
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load summary
    try:
        summary = load_summary(args.summary_file)
        print(f"Loaded summary from {args.summary_file}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Model: {summary.get('model', 'Unknown')}")
        print(f"Epochs: {summary.get('epochs', 0)}")
        print(f"Final accuracy: {summary.get('final_accuracy', 0):.4f}")
        print(f"Average steps: {summary.get('avg_steps', 0):.2f}")
        print(f"Average reward: {summary.get('avg_reward', 0):.2f}")
        print(f"Average thinking tokens: {summary.get('avg_thinking_tokens', 0):.2f}")
        
        # Plot learning curves
        plot_learning_curve(summary, output_dir, args.dpi)
        
    except Exception as e:
        print(f"Error processing summary file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 