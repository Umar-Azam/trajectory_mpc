
# %%
import numpy as np
import matplotlib.pyplot as plt

# Data points for alpha and magnitude (|a|)
alpha_values = np.linspace(-1, 1, 11)
magnitude_values = np.linspace(0, 1, 6)

# Creating a grid of points for the scatter plot
alpha, magnitude = np.meshgrid(alpha_values, magnitude_values)
alpha = alpha.flatten()
magnitude = magnitude.flatten()

# Plotting the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(alpha, magnitude, color='red')
plt.title('Scatter Plot of α vs |a|')
plt.xlabel('α')
plt.ylabel('|a|')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()

# %%
