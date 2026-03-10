import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from pathlib import Path

def create_performance_figure(results_data, output_path="performance_figure.png"):
    """
    Create a comprehensive performance visualization figure.
    
    Args:
        results_data: Dictionary containing evaluation results
        output_path: Path to save the figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Chapter Book QA System Performance', fontsize=16, fontweight='bold')
    
    # Extract metrics
    bert_scores = [r['bert_score'] for r in results_data['individual_results']]
    spoiler_scores = [r['spoiler_score'] for r in results_data['individual_results']]
    answer_correct = [r['answer_equivalent'] for r in results_data['individual_results']]
    spoiler_violations = [r['spoiler_violation'] for r in results_data['individual_results']]
    
    aggregate = results_data['aggregate_metrics']
    
    # Plot 1: BERT Score Distribution
    ax1 = axes[0, 0]
    ax1.hist(bert_scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(aggregate['avg_bert_score'], color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {aggregate["avg_bert_score"]:.3f}')
    ax1.set_xlabel('BERT Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Answer Quality: BERT Score Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Answer Accuracy
    ax2 = axes[0, 1]
    correct_count = sum(answer_correct)
    incorrect_count = len(answer_correct) - correct_count
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax2.pie(
        [correct_count, incorrect_count],
        labels=['Correct', 'Incorrect'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )
    ax2.set_title(f'Answer Accuracy\n({correct_count}/{len(answer_correct)} correct)', 
                  fontsize=12, fontweight='bold')
    
    # Plot 3: Spoiler Detection
    ax3 = axes[1, 0]
    spoiler_count = sum(spoiler_violations)
    safe_count = len(spoiler_violations) - spoiler_count
    colors_spoiler = ['#3498db', '#e67e22']
    wedges, texts, autotexts = ax3.pie(
        [safe_count, spoiler_count],
        labels=['Spoiler-Free', 'Spoiler Detected'],
        autopct='%1.1f%%',
        colors=colors_spoiler,
        startangle=90,
        textprops={'fontsize': 11}
    )
    ax3.set_title(f'Spoiler Prevention\n(Rate: {aggregate["spoiler_free_rate"]:.1%})', 
                  fontsize=12, fontweight='bold')

    # Plot 4: Summary Metrics Bar Chart
    ax4 = axes[1, 1]
    metrics = ['BERT Score', 'Answer\nAccuracy', 'Spoiler-Free\nRate']
    values = [
        aggregate['avg_bert_score'],
        aggregate['answer_accuracy'],
        aggregate['spoiler_free_rate']
    ]
    colors_bar = ['#3498db', '#2ecc71', '#9b59b6']
    bars = ax4.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Aggregate Performance Metrics', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance figure saved to: {output_path}")
    plt.close()


def load_results_from_json(json_path):
    """Load results from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Visualize Book QA performance results")
    parser.add_argument(
        "--results_json",
        type=str,
        required=True,
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="performance_figure.png",
        help="Output path for the figure (default: performance_figure.png)"
    )
    
    args = parser.parse_args()
    
    # Load results
    results_data = load_results_from_json(args.results_json)
    
    # Create figure
    create_performance_figure(results_data, args.output)

if __name__ == "__main__":
    main()
