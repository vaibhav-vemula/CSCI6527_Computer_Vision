import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('preds.csv')

noise_levels = df['noise'].tolist()
confidence_scores = df['confidence'].tolist()

plt.figure(figsize=(10, 6))
plt.plot(noise_levels, confidence_scores, marker='o', linestyle='-', color='blue')
plt.title('Noise Level vs Confidence Score')
plt.xlabel('Noise Level')
plt.ylabel('Confidence Score')
plt.xticks(noise_levels)
plt.grid(True)
plt.savefig('noise_confidence_plot.png', format='png')
print("Plot saved as 'noise_confidence_plot.png'")