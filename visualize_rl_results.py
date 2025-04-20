#!/usr/bin/env python3
"""
Visualization script for RL controller results.
Generates plots of rewards, accuracy, and decision points.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import jsonlines
from collections import defaultdict


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize RL controller results")
    parser.add_argument("--results_dir", type=str, default="rl_cot_results",
                        help="Directory containing results (default: rl_cot_results)")
    parser.add_argument("--output_dir", type=str, default="rl_cot_visualizations",
                        help="Directory to save visualizations (default: rl_cot_visualizations)")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter results by model name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to visualize (default: test)")
    return parser.parse_args()


def load_results(results_dir, model=None, split="test"):
    """Load results from JSONL files in the results directory.
    
    Args:
        results_dir: Directory containing results
        model: Filter results by model name
        split: Dataset split to load
        
    Returns:
        DataFrame containing all results
    """
    results_dir = Path(results_dir)
    all_results = []
    
    # Find all result files
    for file in results_dir.glob(f"results_*.jsonl"):
        if split not in file.name:
            continue
        if model and model.replace('/', '_').replace('-', '_') not in file.name:
            continue
        
        # Load results from file
        with jsonlines.open(file) as reader:
            for result in reader:
                all_results.append(result)
    
    return pd.DataFrame(all_results)


def plot_rewards_over_time(df, output_dir):
    """Plot rewards over time.
    
    Args:
        df: DataFrame containing results
        output_dir: Directory to save visualizations
    """
    plt.figure(figsize=(10, 6))
    
    # Only include numeric columns for mean calculation
    numeric_cols = ['total_reward']
    # Group by epoch and calculate mean reward
    grouped = df.groupby(['epoch', 'id'])[numeric_cols].mean().reset_index()
    
    # Plot average reward per epoch
    sns.lineplot(data=grouped, x='epoch', y='total_reward', estimator='mean', 
                 errorbar=('ci', 95), label='Mean Reward')
    
    plt.title('Average Reward per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'rewards_over_time.png', dpi=300)
    plt.close()


def plot_accuracy_over_time(df, output_dir):
    """Plot accuracy over time.
    
    Args:
        df: DataFrame containing results
        output_dir: Directory to save visualizations
    """
    plt.figure(figsize=(10, 6))
    
    # Only include numeric columns for mean calculation
    numeric_cols = ['correct']
    # Group by epoch and calculate accuracy
    grouped = df.groupby(['epoch', 'id'])[numeric_cols].mean().reset_index()
    
    # Plot accuracy per epoch
    sns.lineplot(data=grouped, x='epoch', y='correct', estimator='mean', 
                 errorbar=('ci', 95), label='Accuracy')
    
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'accuracy_over_time.png', dpi=300)
    plt.close()


def plot_decision_points(df, output_dir):
    """Plot distribution of decision points (when to force an answer).
    
    Args:
        df: DataFrame containing results
        output_dir: Directory to save visualizations
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of steps
    sns.histplot(data=df, x='steps', hue='correct', multiple='stack', bins=20)
    
    plt.title('Distribution of Decision Points')
    plt.xlabel('Number of Steps Before Decision')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'decision_points.png', dpi=300)
    plt.close()


def plot_token_usage(df, output_dir):
    """Plot token usage distribution.
    
    Args:
        df: DataFrame containing results
        output_dir: Directory to save visualizations
    """
    plt.figure(figsize=(10, 6))
    
    # Plot token count vs correctness
    sns.boxplot(data=df, x='correct', y='thinking_token_count')
    
    plt.title('Token Usage vs Correctness')
    plt.xlabel('Correct Answer')
    plt.ylabel('Number of Thinking Tokens')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'token_usage.png', dpi=300)
    plt.close()


def plot_model_comparison(results_dir, output_dir, split="test"):
    """Plot performance comparison across different models.
    
    Args:
        results_dir: Directory containing results
        output_dir: Directory to save visualizations
        split: Dataset split to visualize
    """
    results_dir = Path(results_dir)
    summary_data = []
    
    # Find all summary files
    for file in results_dir.glob(f"summary_*.json"):
        if split not in file.name:
            continue
        
        # Load summary data
        with open(file) as f:
            summary = json.load(f)
            summary_data.append(summary)
    
    if not summary_data:
        print("No summary data found for model comparison")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(summary_data)
    
    # Plot model comparison
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy by model
    acc_plot = sns.barplot(data=df, x='model', y='final_accuracy')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'model_comparison.png', dpi=300)
    plt.close()


def main():
    args = parse_args()
    
    # Load results
    df = load_results(args.results_dir, args.model, args.split)
    
    if df.empty:
        print(f"No results found in {args.results_dir} for split {args.split}")
        return
    
    # Create visualization directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate visualizations
    plot_rewards_over_time(df, output_dir)
    plot_accuracy_over_time(df, output_dir)
    plot_decision_points(df, output_dir)
    plot_token_usage(df, output_dir)
    
    # Model comparison if available
    plot_model_comparison(args.results_dir, output_dir, args.split)
    
    print(f"Visualizations saved to {args.output_dir}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total samples: {len(df)}")
    
    # Only compute means for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if 'total_reward' in numeric_cols:
        print(f"Average reward: {df['total_reward'].mean():.2f}")
    if 'correct' in numeric_cols:
        print(f"Accuracy: {df['correct'].mean():.4f}")
    if 'steps' in numeric_cols:
        print(f"Average steps before decision: {df['steps'].mean():.2f}")
    if 'thinking_token_count' in numeric_cols:
        print(f"Average thinking tokens: {df['thinking_token_count'].mean():.2f}")


if __name__ == "__main__":
    main() 