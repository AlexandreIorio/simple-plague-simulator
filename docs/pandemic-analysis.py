import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('detail.csv')

# Create a figure with 4 subplots arranged in 2x2
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
plt.style.use('default')

# Plot for Healthy Population
ax1.plot(df['round'], df['healthy'], color='green', linewidth=2)
ax1.set_title('Healthy Population Over Time', fontsize=12, pad=10)
ax1.set_xlabel('Round')
ax1.set_ylabel('Number of People')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.fill_between(df['round'], df['healthy'], alpha=0.2, color='green')

# Plot for Infected Population
ax2.plot(df['round'], df['infected'], color='red', linewidth=2)
ax2.set_title('Infected Population Over Time', fontsize=12, pad=10)
ax2.set_xlabel('Round')
ax2.set_ylabel('Number of People')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.fill_between(df['round'], df['infected'], alpha=0.2, color='red')

# Add annotation for peak infections
max_infected = df['infected'].max()
max_infected_round = df.loc[df['infected'].idxmax(), 'round']
ax2.annotate(f'Peak: {max_infected}',
            xy=(max_infected_round, max_infected),
            xytext=(10, 10), textcoords='offset points',
            ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Plot for Immune Population
ax3.plot(df['round'], df['immune'], color='blue', linewidth=2)
ax3.set_title('Immune Population Over Time', fontsize=12, pad=10)
ax3.set_xlabel('Round')
ax3.set_ylabel('Number of People')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.fill_between(df['round'], df['immune'], alpha=0.2, color='blue')

# Plot for Dead Population
ax4.plot(df['round'], df['dead'], color='black', linewidth=2)
ax4.set_title('Deaths Over Time', fontsize=12, pad=10)
ax4.set_xlabel('Round')
ax4.set_ylabel('Number of People')
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.fill_between(df['round'], df['dead'], alpha=0.2, color='gray')

# Adjust layout
plt.suptitle('Pandemic Evolution Analysis', fontsize=16, y=1.02)
plt.tight_layout()

# Save the plot in SVG format
plt.savefig('pandemic_evolution.svg', format='svg', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print some statistics
print("\nKey Statistics:")
print(f"Maximum number of infected: {max_infected} at round {max_infected_round}")
print(f"Final death toll: {df['dead'].iloc[-1]}")
print(f"Total immune at end: {df['immune'].iloc[-1]}")
print(f"Healthy population remaining: {df['healthy'].iloc[-1]}")