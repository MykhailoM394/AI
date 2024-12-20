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
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, normalize
import matplotlib.pyplot as plt

data = pd.read_csv('data_multivar_nb.txt', sep=',')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

classes = np.unique(y)
y_bin = label_binarize(y, classes=classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_decision = svm_model.decision_function(X_test)
svm_prob = normalize(svm_decision, norm='l1', axis=1)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_prob = nb_model.predict_proba(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred, average='macro')
svm_precision = precision_score(y_test, svm_pred, average='macro')
svm_f1 = f1_score(y_test, svm_pred, average='macro')

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred, average='macro')
nb_precision = precision_score(y_test, nb_pred, average='macro')
nb_f1 = f1_score(y_test, nb_pred, average='macro')

print("SVM Metrics:")
print(f"Accuracy: {svm_accuracy:.3f}")
print(f"Recall: {svm_recall:.3f}")
print(f"Precision: {svm_precision:.3f}")
print(f"F1 Score: {svm_f1:.3f}")

print("\nNaive Bayes Metrics:")
print(f"Accuracy: {nb_accuracy:.3f}")
print(f"Recall: {nb_recall:.3f}")
print(f"Precision: {nb_precision:.3f}")
print(f"F1 Score: {nb_f1:.3f}")

svm_auc = roc_auc_score(y_test, svm_prob, multi_class='ovr')
nb_auc = roc_auc_score(y_test, nb_prob, multi_class='ovr')

print(f"SVM AUC: {svm_auc:.3f}")
print(f"Naive Bayes AUC: {nb_auc:.3f}")

for i, cls in enumerate(classes):
    svm_fpr, svm_tpr, _ = roc_curve(y_bin[:, i], svm_prob[:, i])
    nb_fpr, nb_tpr, _ = roc_curve(y_bin[:, i], nb_prob[:, i])
    plt.plot(svm_fpr, svm_tpr, label=f'SVM (Class {cls})')
    plt.plot(nb_fpr, nb_tpr, label=f'NB (Class {cls})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve for Multiclass Classification')
plt.show()
