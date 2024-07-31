import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_multiple_bars(data, labels, group_labels, sub_labels, title='Bar Plot', ylabel='Y-axis', bar_width=0.2, colors=None, success=False, rearanged=False, mapped_color=False):
    """
    Plots a bar chart with multiple bars, each labeled accordingly.

    Parameters:
    - data: list of lists or 2D array, where each inner list represents the data for each group of bars.
    - labels: list of strings, labels for each group of bars.
    - group_labels: list of strings, labels for the groups.
    - sub_labels: list of lists, subcategory labels for each bar in each group.
    - title: string, title of the plot (default 'Bar Plot').
    - ylabel: string, label for the y-axis (default 'Y-axis').
    - bar_width: float, width of each bar (default 0.2).
    - colors: list of strings, colors for each group of bars (default None, which uses default colors).
    """
    # Set the style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    num_groups = len(data)
    num_bars = len(data[0])
    gap_between_sets = bar_width * 0.1  # Define the gap between the second and third bars within each group
    gap_between_finetune = bar_width * 0.2  # Define the larger gap between the fourth and fifth bars within each group

    # Generate bar positions with padding between groups and an additional gap within each group
    indices = np.arange(num_bars) * (num_groups + 1) * bar_width + gap_between_sets + gap_between_finetune
    bar_positions = [indices + bar_width * i for i in range(num_groups)]

    # Check if colors are provided, otherwise use default colors
    if colors is None:
        if mapped_color:
            if rearanged:
                colors = ["#80b5ed", "#f57878", "#7ade86", "#265a91", "#801d1d", "#046b11"]
            else:
                colors = ["#80b5ed", "#f57878", "#3e74ad", "#ad3434", "#0e3d6e", "#780b0b"]
        else:
            # Generate colors using the specified colormap
            cmap = plt.get_cmap("Blues")
            colors = cmap(np.linspace(0, 1, num_groups))

    # Plot each group of bars
    if rearanged:
        labels = [labels[0], labels[2], labels[4], labels[1], labels[3], labels[5]]

    for i, (bars, label) in enumerate(zip(data, labels)):
        adjusted_positions = bar_positions[i].copy()
        if not rearanged:
            if i == 2 or i == 3:  # Shift the third and fourth bars to create the gap
                adjusted_positions += gap_between_sets
            if i == 4 or i == 5:  # Shift the fifth and sixth bars to create the larger gap
                adjusted_positions += gap_between_finetune
        else:
            if i in [3, 4, 5]:
                adjusted_positions += gap_between_finetune
        plt.bar(adjusted_positions, bars, bar_width, label=label, color=colors[i], edgecolor='black')
        # Add annotations
        for x, y in zip(adjusted_positions, bars):
            if success:
                plt.text(x, y + 0.01, f'{y:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                plt.text(x, y + 0.001, f'{y:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add subcategory labels
    for pos, sub_label_set in zip(bar_positions, sub_labels):
        for x, sub_label in zip(pos, sub_label_set):
            if success:
                plt.text(x+0.12, -0.05, sub_label, ha='center', va='top', fontsize=10, fontweight='bold')
            else:
                plt.text(x + 0.12, -0.005, sub_label, ha='center', va='top', fontsize=10, fontweight='bold')

    # Add labels, title, and legend
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(indices + bar_width * (num_groups - 1) / 2, group_labels, fontsize=12, fontweight='bold')
    if rearanged:
        plt.legend(["Original", "Finetuned", "Combined"], loc='upper right', fontsize=12)
    else:
        plt.legend(["Initial Faucet(s)", "Transfer Faucet"], loc='upper right', fontsize=12)

    # Add grid lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Remove top and right spines
    sns.despine()

    # Set x and y limits
    if success:
        plt.ylim(0, 1)
    plt.xlim(-bar_width, indices[-1] + bar_width * (num_groups + 1))

    plt.tight_layout()
    plt.show()


# Example usage:
accuracy_data = [
    [0.85, np.mean([0, 1, 1]), 0.88],  # Init (Orig)
    [0.65, np.mean([0, 0, 0]), 0.68],  # Transfer (Orig)
    [0.65, np.mean([0, 0, 0]), 0.3],  # Init (Fine)
    [0.62, np.mean([0, 1, 0]), 0.80],  # Transfer (Fine)
    [0.35, np.mean([0, 1, 1]), 0.2],  # Init (Comb)
    [0.32, np.mean([1, 0, 0]), 0.2],  # Transfer (Comb)
]

rewards_data = [
    [0, np.mean([0.031, 0.086, 0.096]), 0],  # Init (Orig)
    [0, np.mean([0.029, 0.059, 0.029]), 0],  # Transfer (Orig)
    [0, np.mean([0.071, 0.022, 0.092]), 0],  # Init (Fine)
    [0, np.mean([0.040, 0.079, 0.025]), 0],  # Transfer (Fine)
    [0, np.mean([0.026, 0.087, 0.092]), 0],  # Init (Comb)
    [0, np.mean([0.095, 0.048, 0.076]), 0],  # Transfer (Comb)
]

rearanged_success = [accuracy_data[0], accuracy_data[2], accuracy_data[4], accuracy_data[1], accuracy_data[3], accuracy_data[5]]
rearanged_rewards = [rewards_data[0], rewards_data[2], rewards_data[4], rewards_data[1], rewards_data[3], rewards_data[5]]

labels = ['Initial Faucet(s) (Original)', 'Transfer Faucet (Original)', 'Initial Faucet(s) (Finetuned)', 'Transfer Faucet (Finetuned)', 'Initial Faucet(s) (Combined)', 'Transfer Faucet (Combined)']
group_labels = ['Single Faucet', 'Mult. Faucets (Same Cat.)', 'Mult. Faucets (Different Cat.)']

# Subcategory labels for each group
sub_labels = [
    ['Original', 'Original', 'Original'],
    ['', '', ''],
    ['Finetuned', 'Finetuned', 'Finetuned'],
    ['', '', ''],
    ['Combined', 'Combined', 'Combined'],
    ['', '', ''],
]

plot_multiple_bars(accuracy_data, labels, group_labels, sub_labels, title='Initial/Finetune/Transfer Success Comparison', ylabel='Mean Success Rate', colors=None, success=True, mapped_color=True)
plot_multiple_bars(rewards_data, labels, group_labels, sub_labels, title='Initial/Finetune/Transfer Reward Comparison', ylabel='Mean Reward', colors=None, rearanged=False, mapped_color=True)
