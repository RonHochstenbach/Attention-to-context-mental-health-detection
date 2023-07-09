import pandas as pd
import matplotlib.pyplot as plt

metric_history = pd.read_csv("/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Self-Harm_HAN_2023-07-06 21:56:12.679465metricHistory.csv")
metric_history['f1'] = 2 * (metric_history['precision'] * metric_history['recall']) / (metric_history['precision'] + metric_history['recall'])
#metric_history['val_f1'] = 2 * (metric_history['val_precision'] * metric_history['val_recall']) / (metric_history['val_precision'] + metric_history['val_recall'])


# Create the upper plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Plot auc and val_auc in the upper plot
ax1.set_ylabel('auc/val_auc')
ax1.plot(metric_history['epoch'], metric_history['auc'], label='auc', color='blue')
#ax1.plot(metric_history['epoch'], metric_history['val_auc'], label='val_auc', color='purple')
ax1.legend(loc='upper right')

# Plot f1 and val_f1 in the middle plot
ax2.set_ylabel('f1/val_f1')
ax2.plot(metric_history['epoch'], metric_history['f1'], label='f1', color='green')
#ax2.plot(metric_history['epoch'], metric_history['val_f1'], label='val_f1', color='orange')
ax2.legend(loc='upper right')

# Plot loss and val_loss in the bottom plot
ax3.set_xlabel('epoch')
ax3.set_ylabel('loss/val_loss')
ax3.plot(metric_history['epoch'], metric_history['loss'], label='loss', color='red')
#ax3.plot(metric_history['epoch'], metric_history['val_loss'], label='val_loss', color='brown')
ax3.legend(loc='upper right')

# Adjust spacing between plots
plt.subplots_adjust(hspace=0.3)

# Show the plot
plt.show()
