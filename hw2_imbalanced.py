import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Import dataset and convert to numpy arrays
df = pd.read_csv('hw4_data.csv')
model_output = df['model_output'].to_numpy()
prediction = df['prediction'].to_numpy()
true_class = df['true_class'].to_numpy()

# Finding TP, FP, TN, FN according to confusion matrix definitions
TP = int(((prediction==1) & (true_class==1)).sum())
FP = int(((prediction==1) & (true_class==0)).sum())
TN = int(((prediction==0) & (true_class==0)).sum())
FN = int(((prediction==0) & (true_class==1)).sum())

print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

precision = TP / (TP + FP) # Compute precision
recall = TP / (TP + FN) # Compute recall

print(f'Precision: {precision:0.2f}, Recall: {recall:0.2f}')

fpr, tpr, thresholds = roc_curve(true_class, model_output) # Compute false positive and true positive rates
roc_auc = auc(fpr, tpr) # Compute the area under the ROC curve (AUC)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr) # ROC Curve
plt.plot([0, 1], [0, 1], linestyle='--') # Classifier line
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)

# Find min FPR for TPR of 0.9
min_fpr = np.interp(0.9, tpr, fpr)
plt.axvline(x=min_fpr, linestyle=':')
print(f'Min FPR: {min_fpr:0.2f}')

plt.show()