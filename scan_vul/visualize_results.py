import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from pathlib import Path

def load_statistics(scan_ids, results_dir="./scan_results"):
    """
    Load statistics data from multiple scan IDs
    
    Args:
        scan_ids (list): List of scan IDs
        results_dir (str): Results directory path
        
    Returns:
        list: List of statistics data
    """
    all_stats = []
    
    for scan_id in scan_ids:
        stats_path = os.path.join(results_dir, scan_id, "statistics.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                # Ensure model_id exists
                if 'model_id' not in stats:
                    stats['model_id'] = scan_id
                all_stats.append(stats)
        else:
            print(f"Warning: Statistics not found for scan ID {scan_id}")
    
    return all_stats

def plot_vulnerability_severity(all_stats):
    """Plot vulnerability severity distribution comparison"""
    models = [stat['model_id'] for stat in all_stats]
    low_counts = [stat['severity_counts']['LOW'] for stat in all_stats]
    medium_counts = [stat['severity_counts']['MEDIUM'] for stat in all_stats]
    high_counts = [stat['severity_counts']['HIGH'] for stat in all_stats]
    
    x = np.arange(len(models))  # positions
    width = 0.25  # bar width
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width, low_counts, width, label='LOW', color='#FFD700')
    rects2 = ax.bar(x, medium_counts, width, label='MEDIUM', color='#FFA500')
    rects3 = ax.bar(x + width, high_counts, width, label='HIGH', color='#FF4500')
    
    # Add data labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    
    ax.set_ylabel('Vulnerability Count')
    ax.set_title('Vulnerability Severity Distribution by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    fig.tight_layout()
    return fig

def plot_issue_distribution(all_stats):
    """Plot issue type distribution comparison"""
    models = [stat['model_id'] for stat in all_stats]
    
    # Calculate the percentage of each issue type
    data = []
    for stat in all_stats:
        total = stat['total_samples']
        if total == 0:
            continue
            
        vuln_pct = stat['vulnerability_count'] / total * 100
        invalid_pct = stat['invalid_code_count'] / total * 100
        hallu_pct = stat['hallucination_count'] / total * 100
        normal_pct = 100 - vuln_pct - invalid_pct - hallu_pct
        
        data.append([normal_pct, vuln_pct, invalid_pct, hallu_pct])
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bottoms = np.zeros(len(models))
    categories = ['Normal Code', 'Vulnerable Code', 'Invalid Code', 'Hallucination']
    colors = ['#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    
    for i, cat in enumerate(categories):
        values = data[:, i]
        ax.bar(models, values, bottom=bottoms, label=cat, color=colors[i])
        # Add percentage labels
        for j, v in enumerate(values):
            if v > 5:  # Only display labels for segments > 5%
                ax.text(j, bottoms[j] + v/2, f"{v:.1f}%", ha='center', va='center')
        bottoms += values
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Code Quality Distribution by Model')
    ax.legend(loc='upper right')
    
    # Set y-axis to percentage format
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    fig.tight_layout()
    return fig

def plot_vulnerability_count_comparison(all_stats):
    """Plot vulnerability count and rate comparison"""
    models = [stat['model_id'] for stat in all_stats]
    vuln_counts = [stat['vulnerability_count'] for stat in all_stats]
    
    # Calculate vulnerability percentage
    total_samples = [stat['total_samples'] for stat in all_stats]
    vuln_rates = [count/total*100 if total > 0 else 0 for count, total in zip(vuln_counts, total_samples)]
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot bar chart - Vulnerability count
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Vulnerability Count', color=color)
    bars = ax1.bar(models, vuln_counts, color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', color=color)
    
    # Create second Y-axis - Vulnerability rate
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Vulnerability Rate (%)', color=color)
    line = ax2.plot(models, vuln_rates, color=color, marker='o', linestyle='-', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add rate labels
    for i, rate in enumerate(vuln_rates):
        ax2.annotate(f'{rate:.1f}%',
                    xy=(i, rate),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', color=color)
    
    ax1.set_title('Vulnerability Count and Rate by Model')
    fig.tight_layout()
    
    return fig

def plot_issue_counts(all_stats):
    """Plot issue count comparison by type"""
    models = [stat['model_id'] for stat in all_stats]
    vuln_counts = [stat['vulnerability_count'] for stat in all_stats]
    invalid_counts = [stat['invalid_code_count'] for stat in all_stats]
    hallu_counts = [stat['hallucination_count'] for stat in all_stats]
    
    # Calculate normal code count
    normal_counts = [stat['total_samples'] - stat['vulnerability_count'] - 
                    stat['invalid_code_count'] - stat['hallucination_count'] 
                    for stat in all_stats]
    
    x = np.arange(len(models))  # positions
    width = 0.2  # bar width
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot four categories as bars
    rects1 = ax.bar(x - width*1.5, normal_counts, width, label='Normal Code', color='#4CAF50')
    rects2 = ax.bar(x - width/2, vuln_counts, width, label='Vulnerable Code', color='#FF9800')
    rects3 = ax.bar(x + width/2, invalid_counts, width, label='Invalid Code', color='#F44336')
    rects4 = ax.bar(x + width*1.5, hallu_counts, width, label='Hallucination', color='#9C27B0')
    
    # Add data labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    add_labels(rects4)
    
    ax.set_ylabel('Sample Count')
    ax.set_title('Code Quality Statistics by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    fig.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize code scan results')
    parser.add_argument('--scan_ids', nargs='+', required=True, help='List of scan IDs to compare')
    parser.add_argument('--results_dir', type=str, default='./scan_results', help='Results directory path')
    parser.add_argument('--output_dir', type=str, default='./img', help='Chart output directory')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load statistics data
    all_stats = load_statistics(args.scan_ids, args.results_dir)
    
    if not all_stats:
        print("Error: No statistics data found")
        return
    
    # 1. Plot vulnerability severity distribution
    fig1 = plot_vulnerability_severity(all_stats)
    fig1.savefig(os.path.join(args.output_dir, 'vulnerability_severity.png'), dpi=300)
    
    # 2. Plot issue distribution
    fig2 = plot_issue_distribution(all_stats)
    fig2.savefig(os.path.join(args.output_dir, 'issue_distribution.png'), dpi=300)
    
    # 3. Plot vulnerability count and rate
    fig3 = plot_vulnerability_count_comparison(all_stats)
    fig3.savefig(os.path.join(args.output_dir, 'vulnerability_count_comparison.png'), dpi=300)
    
    # 4. Plot issue counts by type
    fig4 = plot_issue_counts(all_stats)
    fig4.savefig(os.path.join(args.output_dir, 'issue_counts.png'), dpi=300)
    
    print(f"Charts saved to {args.output_dir} directory")
    
    # Create a summary JSON with all model metrics
    summary = {}
    for stat in all_stats:
        model_id = stat['model_id']
        summary[model_id] = {
            "total_samples": stat['total_samples'],
            "vulnerability_count": stat['vulnerability_count'],
            "vulnerability_rate": f"{stat['vulnerability_count']/stat['total_samples']*100:.2f}%" if stat['total_samples'] > 0 else "0%",
            "severity_breakdown": stat['severity_counts'],
            "invalid_code_count": stat['invalid_code_count'],
            "hallucination_count": stat['hallucination_count']
        }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("Generated charts:")
    print("1. vulnerability_severity.png - Vulnerability severity distribution")
    print("2. issue_distribution.png - Issue type distribution")
    print("3. vulnerability_count_comparison.png - Vulnerability count and rate")
    print("4. issue_counts.png - Issue counts by type")
    print("5. summary.json - Summary of all model metrics")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}") 