import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from scipy.stats import ttest_rel
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]

# Load data
data = pd.read_csv("processed.cleveland.data", names=column_names, na_values="?")
data.dropna(inplace=True)

# Convert target to binary
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Encode categorical columns
for col in ['cp', 'slope', 'thal']:
    data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop('target', axis=1).values
y = data['target'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# --- Custom AdaBoost ---
class AdaBoost:
    def __init__(self, n_estimators=10, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learners = []
        self.alphas = []
        self.train_errors = []

    def fit(self, X, y):
        n_samples = len(y)
        w = np.full(n_samples, 1 / n_samples)

        for t in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
            clf.fit(X, y, sample_weight=w)
            y_pred = clf.predict(X)

            miss = (y_pred != y).astype(int)
            error = np.dot(w, miss)

            EPS = 1e-10
            if error > 0.5:
                continue
            elif error == 0:
                alpha = 1
            else:
                alpha = 0.5 * np.log((1 - error + EPS) / (error + EPS))

            w *= np.exp(-alpha * (2 * y - 1) * (2 * y_pred - 1))
            w /= w.sum()

            self.learners.append(clf)
            self.alphas.append(alpha)

            train_pred = self.predict(X)
            train_error = 1 - accuracy_score(y, train_pred)
            self.train_errors.append(train_error)

    def predict(self, X):
        learner_preds = np.array([alpha * (2 * clf.predict(X) - 1) for clf, alpha in zip(self.learners, self.alphas)])
        result = np.sign(learner_preds.sum(axis=0))
        return (result + 1) // 2

# --- Custom Gradient Boosting ---
class GBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.base_learners = []
        self.init_prediction = 0
        self.train_errors = []

    def fit(self, X, y):
        N = len(y)
        self.init_prediction = np.mean(y)
        current_preds = np.full(N, self.init_prediction)

        for _ in range(self.n_estimators):
            residuals = y - current_preds

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions = tree.predict(X)

            current_preds += self.learning_rate * predictions
            self.base_learners.append(tree)

            binary_preds = (current_preds >= 0.5).astype(int)
            error = 1 - accuracy_score(y, binary_preds)
            self.train_errors.append(error)

    def predict(self, X):
        preds = np.full(X.shape[0], self.init_prediction)
        for tree in self.base_learners:
            preds += self.learning_rate * tree.predict(X)
        return (preds >= 0.5).astype(int)

# Train and evaluate AdaBoost
ada = AdaBoost(n_estimators=100, max_depth=3)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
ada_acc = accuracy_score(y_test, ada_pred)

# Train and evaluate GBM
gbm = GBM(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X_train, y_train)
gbm_pred = gbm.predict(X_test)
gbm_acc = accuracy_score(y_test, gbm_pred)

print("AdaBoost Accuracy:", ada_acc)
print("Gradient Boost Accuracy:", gbm_acc)

# --- t-test ---
t_stat, p_val = ttest_rel(ada_pred, gbm_pred)
print(f"Paired t-test result: t = {t_stat:.4f}, p = {p_val:.4f}")

# Optional: Plot training errors
plt.plot(range(1, len(ada.train_errors) + 1), ada.train_errors, label="AdaBoost")
plt.plot(range(1, len(gbm.train_errors) + 1), gbm.train_errors, label="GBM")
plt.title("Training Error Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.legend()
plt.grid(True)
plt.show()



# Example: assuming you have these from your models
# y_test      → true labels
# ada_preds   → predictions from your AdaBoost
# gb_preds    → predictions from your Gradient Boost

# Accuracy
print("AdaBoost Accuracy:", accuracy_score(y_test, ada_pred))
print("Gradient Boost Accuracy:", accuracy_score(y_test, gbm_pred))

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_ada = confusion_matrix(y_test, ada_pred)
cm_gb = confusion_matrix(y_test, gbm_pred)

ConfusionMatrixDisplay(cm_ada).plot(ax=axes[0], values_format='d')
axes[0].set_title("AdaBoost Confusion Matrix")

ConfusionMatrixDisplay(cm_gb).plot(ax=axes[1], values_format='d')
axes[1].set_title("Gradient Boost Confusion Matrix")

plt.tight_layout()
plt.show()

# Compare predictions
diffs = (ada_pred != gbm_pred).astype(int)
print("Number of samples where predictions differ:", np.sum(diffs))

# Paired t-test
t_stat, p_value = ttest_rel(ada_pred, gbm_pred)
print(f"Paired t-test: t = {t_stat:.4f}, p = {p_value:.4f}")

 plot disagreement
plt.figure(figsize=(8, 4))
plt.plot(diffs, marker='o', linestyle='None', alpha=0.6)
plt.title("Disagreement Between AdaBoost and Gradient Boost Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Disagreement (1 = mismatch)")
plt.grid(True)
plt.show()
