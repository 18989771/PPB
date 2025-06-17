import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import ttest_rel

# Load dataset
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]
data = pd.read_csv("processed.cleveland.data", names=column_names, na_values="?")
data.dropna(inplace=True)

# Binary target
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Encode categoricals
for col in ['cp', 'slope', 'thal']:
    data[col] = LabelEncoder().fit_transform(data[col])

# Features and labels
X = data.drop('target', axis=1).values
y = data['target'].values

# Normalize features for SGD
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# --- AdaBoost using SGDClassifier ---
class AdaBoostSGD:
    def __init__(self, n_estimators=100, max_iter=1000):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.learners = []
        self.alphas = []
        self.train_errors = []
        self.learner_errors = []  # Track weak learner errors

    def fit(self, X, y):
        n = len(y)
        w = np.full(n, 1 / n)

        for i in range(self.n_estimators):
            clf = SGDClassifier(loss='log_loss', max_iter=self.max_iter, tol=None, random_state=42)
            clf.fit(X, y, sample_weight=w)
            y_pred = clf.predict(X)

            miss = (y_pred != y).astype(int)
            error = np.dot(w, miss)

            EPS = 1e-10
            alpha = 0.5 * np.log((1 - error + EPS) / (error + EPS))

            # Debug prints
            print(f"Iteration {i+1}: Weak learner error = {error:.4f}, alpha = {alpha:.4f}")

            # Stop early if error is too high or zero
            if error >= 0.5:
                print("Stopping early due to weak learner error >= 0.5")
                break
            if error == 0:
                print("Stopping early due to perfect learner")
                break

            w *= np.exp(-alpha * (2 * y - 1) * (2 * y_pred - 1))
            w /= np.sum(w)

            self.learners.append(clf)
            self.alphas.append(alpha)
            self.learner_errors.append(error)

            train_pred = self.predict(X)
            err = 1 - accuracy_score(y, train_pred)
            self.train_errors.append(err)

    def predict(self, X):
        preds = np.array([alpha * (2 * clf.predict(X) - 1) for clf, alpha in zip(self.learners, self.alphas)])
        return ((np.sign(preds.sum(axis=0)) + 1) // 2).astype(int)

# --- Gradient Boosting using SGDRegressor ---
class GBM_SGD:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learners = []
        self.init_prediction = 0
        self.train_errors = []

    def fit(self, X, y):
        n = len(y)
        self.init_prediction = np.mean(y)
        current_preds = np.full(n, self.init_prediction)

        for _ in range(self.n_estimators):
            residuals = y - current_preds

            model = SGDRegressor(max_iter=100, tol=1e-3, random_state=42)
            model.fit(X, residuals)
            preds = model.predict(X)

            current_preds += self.learning_rate * preds
            self.base_learners.append(model)

            binary_preds = (current_preds >= 0.5).astype(int)
            err = 1 - accuracy_score(y, binary_preds)
            self.train_errors.append(err)

    def predict(self, X):
        preds = np.full(X.shape[0], self.init_prediction)
        for model in self.base_learners:
            preds += self.learning_rate * model.predict(X)
        return (preds >= 0.5).astype(int)

# Train models
ada = AdaBoostSGD(n_estimators=100, max_iter=10)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
ada_acc = accuracy_score(y_test, ada_pred)

gbm = GBM_SGD(n_estimators=100, learning_rate=0.1)
gbm.fit(X_train, y_train)
gbm_pred = gbm.predict(X_test)
gbm_acc = accuracy_score(y_test, gbm_pred)

# Print metrics
print("AdaBoost (SGD) Accuracy:", ada_acc)
print("Gradient Boost (SGD) Accuracy:", gbm_acc)

# Paired t-test
t_stat, p_val = ttest_rel(ada_pred, gbm_pred)
print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.4f}")

# Plot training errors
plt.plot(range(1, len(ada.train_errors)+1), ada.train_errors, label="AdaBoost-SGD")
plt.plot(range(1, len(gbm.train_errors)+1), gbm.train_errors, label="GBM-SGD")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.title("Training Error over Iterations")
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_ada = confusion_matrix(y_test, ada_pred)
cm_gbm = confusion_matrix(y_test, gbm_pred)

ConfusionMatrixDisplay(cm_ada).plot(ax=axes[0], values_format='d')
axes[0].set_title("AdaBoost-SGD Confusion Matrix")

ConfusionMatrixDisplay(cm_gbm).plot(ax=axes[1], values_format='d')
axes[1].set_title("GBM-SGD Confusion Matrix")

plt.tight_layout()
plt.show()

# Compare predictions
disagreement = (ada_pred != gbm_pred).astype(int)
print("Disagreements between AdaBoost and GBM predictions:", np.sum(disagreement))

# Plot disagreement
plt.figure(figsize=(8, 4))
plt.plot(disagreement, marker='o', linestyle='None', alpha=0.6)
plt.title("Disagreement Between AdaBoost-SGD and GBM-SGD")
plt.xlabel("Sample Index")
plt.ylabel("Mismatch (1 = different)")
plt.grid(True)
plt.show()
