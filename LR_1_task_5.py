import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')

thresh = 0.5
df['predicted_RF'] = (df['model_RF'] >= thresh).astype('int')
df['predicted_LR'] = (df['model_LR'] >= thresh).astype('int')

def meshkalov_find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def meshkalov_find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def meshkalov_find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def meshkalov_find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))

def meshkalov_find_conf_matrix_values(y_true, y_pred):
    TP = meshkalov_find_TP(y_true, y_pred)
    FN = meshkalov_find_FN(y_true, y_pred)
    FP = meshkalov_find_FP(y_true, y_pred)
    TN = meshkalov_find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def meshkalov_my_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = meshkalov_find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def meshkalov_my_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = meshkalov_find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def meshkalov_my_recall_score(y_true, y_pred):
    TP, FN, _, _ = meshkalov_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

def meshkalov_my_precision_score(y_true, y_pred):
    TP, _, FP, _ = meshkalov_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

def meshkalov_my_f1_score(y_true, y_pred):
    recall = meshkalov_my_recall_score(y_true, y_pred)
    precision = meshkalov_my_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

actual = df['actual_label'].values
predicted_RF = df['predicted_RF'].values
predicted_LR = df['predicted_LR'].values

print("Confusion Matrix (RF):\n", meshkalov_my_confusion_matrix(actual, predicted_RF))
print("Accuracy (RF):", meshkalov_my_accuracy_score(actual, predicted_RF))
print("Recall (RF):", meshkalov_my_recall_score(actual, predicted_RF))
print("Precision (RF):", meshkalov_my_precision_score(actual, predicted_RF))
print("F1 Score (RF):", meshkalov_my_f1_score(actual, predicted_RF))

fpr_RF, tpr_RF, _ = roc_curve(df['actual_label'], df['model_RF'])
fpr_LR, tpr_LR, _ = roc_curve(df['actual_label'], df['model_LR'])
auc_RF = roc_auc_score(df['actual_label'], df['model_RF'])
auc_LR = roc_auc_score(df['actual_label'], df['model_LR'])

plt.plot(fpr_RF, tpr_RF, 'r-', label=f'RF AUC: {auc_RF:.3f}')
plt.plot(fpr_LR, tpr_LR, 'b-', label=f'LR AUC: {auc_LR:.3f}')
plt.plot([0, 1], [0, 1], 'k-', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.savefig('roc_curve.png')  # Додаємо цей рядок для збереження графіка
plt.show()
