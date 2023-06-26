import pandas as pd
import matplotlib.pyplot as plt

metric_history = pd.read_csv('/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Depression_HAN_2023-06-23 22:08:52.141772metricHistory.csv')
metric_history['f1'] = 2 * (metric_history['precision'] * metric_history['recall']) / (metric_history['precision'] + metric_history['recall'])
metric_history['val_f1'] = 2 * (metric_history['val_precision'] * metric_history['val_recall']) / (metric_history['val_precision'] + metric_history['val_recall'])

# Assuming your DataFrame is called 'metric_history'
# Replace 'metric_history' with your actual DataFrame variable name

# Create the upper plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Plot auc and val_auc in the upper plot
ax1.set_ylabel('prec')
ax1.plot(metric_history['epoch'], metric_history['precision'], label='precision', color='blue')
ax1.plot(metric_history['epoch'], metric_history['val_precision'], label='val_precision', color='purple')
ax1.legend(loc='upper right')

# Plot f1 and val_f1 in the middle plot
ax2.set_ylabel('recall')
ax2.plot(metric_history['epoch'], metric_history['recall'], label='recall', color='green')
ax2.plot(metric_history['epoch'], metric_history['val_recall'], label='val_recall', color='orange')
ax2.legend(loc='upper right')


# Adjust spacing between plots
plt.subplots_adjust(hspace=0.3)

# Show the plot
plt.show()