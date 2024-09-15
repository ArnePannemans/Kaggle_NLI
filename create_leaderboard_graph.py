import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

models = ['Phi-3 Mini 4k', 'Phi-3.5 Mini', 'Phi-3 Small']
out_of_the_box = [67.3, 74.1, 76.6]
fine_tuned = [81.1, 81.9, 0]  # Replace None with 0 for plotting purposes

bar_width = 0.35

r = np.arange(len(models))
fig, ax = plt.subplots(figsize=(8, 6))

# Plot zero-shot results
bars1 = ax.bar(r, out_of_the_box, color='#4c72b0', width=bar_width, label='Zero-Shot', edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Leaderboard: Zero-Shot', fontsize=16)
ax.set_xticks(r)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=12)


ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

zero_shot_image_path = os.path.join(output_dir, "zero_shot_results.png")
plt.savefig(zero_shot_image_path)

# Create plot for Fine-Tuned results
fig, ax = plt.subplots(figsize=(8, 6))

bars2 = ax.bar(r, fine_tuned, color='#55a868', width=bar_width, label='Fine-Tuned', edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Leaderboard: Fine-Tuned', fontsize=16)
ax.set_xticks(r)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=12)

ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

fine_tuned_image_path = os.path.join(output_dir, "fine_tuned_results.png")
plt.savefig(fine_tuned_image_path)
