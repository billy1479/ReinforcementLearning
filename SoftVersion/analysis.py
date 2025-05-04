import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.gridspec import GridSpec

def parse_log_file(file_path):
    """Parse the log file and return a pandas DataFrame."""
    try:
        # Read the log file
        df = pd.read_csv(file_path, sep='\t')
        return df
    except Exception as e:
        print(f"Error parsing log file: {e}")
        exit(1)

def calculate_moving_average(data, window_size=3):
    """Calculate the moving average of rewards."""
    return data.rolling(window=window_size, min_periods=1).mean()

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def analyze_training_data(df):
    """Analyze the training data and return key metrics."""
    # Basic statistics
    max_reward = df['reward_sum'].max()
    min_reward = df['reward_sum'].min()
    max_episodes = len(df)
    
    # Success rate (episodes that reached max length)
    max_length_episodes = len(df[df['length'] == 2000])
    success_rate = max_length_episodes / max_episodes
    
    # Check for improvement
    first_half = df.iloc[:len(df)//2]
    second_half = df.iloc[len(df)//2:]
    first_half_avg = first_half['reward_sum'].mean()
    second_half_avg = second_half['reward_sum'].mean()
    is_improving = second_half_avg > first_half_avg
    
    # Identify optimal policy cases (early terminations with good rewards in later episodes)
    avg_reward = df['reward_sum'].mean()
    midpoint = len(df) // 2
    optimal_policy_count = sum(1 for i in range(midpoint, len(df)) 
                              if (df.iloc[i]['length'] < 2000 and 
                                  df.iloc[i]['reward_sum'] > avg_reward))
    
    return {
        'max_reward': max_reward,
        'min_reward': min_reward,
        'max_episodes': max_episodes,
        'max_length_episodes': max_length_episodes,
        'success_rate': success_rate,
        'optimal_policy_count': optimal_policy_count,
        'is_improving': is_improving,
        'improvement_amount': second_half_avg - first_half_avg
    }

def plot_reward_progression(df, output_dir, window_size=3):
    """Plot the reward progression over episodes."""
    plt.figure(figsize=(12, 6))
    
    # Calculate moving average
    df['moving_avg'] = calculate_moving_average(df['reward_sum'], window_size)
    
    # Plot reward progression
    plt.plot(df['count'], df['reward_sum'], 'o-', label='Reward')
    plt.plot(df['count'], df['moving_avg'], 'r-', label=f'Moving Avg (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Progression During Training')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_progression.png'), dpi=300)
    plt.close()

def plot_episode_duration(df, output_dir):
    """Plot the episode duration over episodes."""
    plt.figure(figsize=(12, 6))
    
    # Create bar plot for episode lengths
    bars = plt.bar(df['count'], df['length'], alpha=0.7)
    
    # Determine the midpoint episode for color coding logic
    midpoint = len(df) // 2
    
    # Color bars differently based on episode length and training phase
    for i, bar in enumerate(bars):
        if df.iloc[i]['length'] == 2000:
            bar.set_color('green')
        elif i >= midpoint and df.iloc[i]['reward_sum'] > df['reward_sum'].mean():
            # Early terminations in later episodes with above-average rewards are likely optimal policies
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Length (steps)')
    plt.title('Episode Duration')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Completed (2000 steps)'),
        Patch(facecolor='orange', label='Optimal Policy (early termination with good reward)'),
        Patch(facecolor='red', label='Failed (early termination with poor reward)')
    ]
    plt.legend(handles=legend_elements)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'episode_duration.png'), dpi=300)
    plt.close()

def plot_reward_vs_length(df, output_dir):
    """Plot the reward vs episode length."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(df['length'], df['reward_sum'], alpha=0.7, s=80, c=df['count'], cmap='viridis')
    
    plt.xlabel('Episode Length (steps)')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode Length')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(label='Episode Number')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_vs_length.png'), dpi=300)
    plt.close()

def plot_training_summary(df, metrics, output_dir):
    """Create a summary plot with key metrics."""
    # Create a figure with a specific layout
    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Reward progression
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['count'], df['reward_sum'], 'o-', label='Reward')
    ax1.plot(df['count'], calculate_moving_average(df['reward_sum'], 3), 'r-', label='Moving Avg (window=3)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Progression')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Episode length
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(df['count'], df['length'], alpha=0.7)
    
    # Determine the midpoint episode for color coding logic
    midpoint = len(df) // 2
    
    # Color bars differently based on episode length and training phase
    for i, bar in enumerate(bars):
        if df.iloc[i]['length'] == 2000:
            bar.set_color('green')
        elif i >= midpoint and df.iloc[i]['reward_sum'] > df['reward_sum'].mean():
            # Early terminations in later episodes with above-average rewards are likely optimal policies
            bar.set_color('orange')
        else:
            bar.set_color('red')
            
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Duration')
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add a legend to episode duration
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Completed (2000 steps)'),
        Patch(facecolor='orange', label='Optimal Policy (early term.)'),
        Patch(facecolor='red', label='Failed (early term.)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize='x-small')
    
    # Reward vs Length scatter
    ax3 = fig.add_subplot(gs[1, 1])
    scatter = ax3.scatter(df['length'], df['reward_sum'], alpha=0.7, s=80, c=df['count'], cmap='viridis')
    ax3.set_xlabel('Episode Length')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward vs Episode Length')
    ax3.grid(True, linestyle='--', alpha=0.7)
    fig.colorbar(scatter, ax=ax3, label='Episode Number')
    
    # Metrics text box
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Calculate optimal policy cases (early terminations with good rewards in later episodes)
    optimal_policy_count = sum(1 for i in range(midpoint, len(df)) 
                              if (df.iloc[i]['length'] < 2000 and 
                                  df.iloc[i]['reward_sum'] > df['reward_sum'].mean()))
    
    metrics_text = (
        f"Training Summary:\n"
        f"Total Episodes: {metrics['max_episodes']}\n"
        f"Episodes Reaching Max Length: {metrics['max_length_episodes']} ({metrics['success_rate']*100:.1f}%)\n"
        f"Optimal Policy Early Terminations: {optimal_policy_count}\n"
        f"Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]\n"
        f"Is Agent Improving: {'Yes' if metrics['is_improving'] else 'No'} "
        f"(Difference: {metrics['improvement_amount']:.2f})\n\n"
        f"Observations:\n"
        f"- Episodes with high durations (2000 steps) generally have better rewards\n"
        f"- Short episodes in later training may represent optimal policies\n"
        f"- Early terminations can indicate both failures and efficient solutions\n"
        f"- {'The agent is showing improvement over time' if metrics['is_improving'] else 'The agent is not showing significant improvement'}"
    )
    
    ax4.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Save the summary plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_summary.png'), dpi=300)
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze bipedal walker training log file')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('--output', type=str, default='output', help='Output directory for plots')
    parser.add_argument('--window', type=int, default=3, help='Window size for moving average')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory
    output_dir = create_output_directory(args.output)
    
    # Parse log file
    df = parse_log_file(args.log_file)
    
    # Analyze training data
    metrics = analyze_training_data(df)
    
    # Generate individual plots
    plot_reward_progression(df, output_dir, args.window)
    plot_episode_duration(df, output_dir)
    plot_reward_vs_length(df, output_dir)
    
    # Generate summary plot
    plot_training_summary(df, metrics, output_dir)
    
    print(f"Analysis complete. Plots saved to {output_dir}")
    print(f"Total episodes analyzed: {metrics['max_episodes']}")
    print(f"Success rate: {metrics['success_rate']*100:.1f}%")
    print(f"Is agent improving: {'Yes' if metrics['is_improving'] else 'No'}")

if __name__ == "__main__":
    main()