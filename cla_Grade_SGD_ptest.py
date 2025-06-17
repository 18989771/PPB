import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

# Load and preprocess dataset
df = pd.read_csv('grade.csv', sep=';')
df = df[df['Target'].isin(['Graduate', 'Dropout'])]
df['Target'] = df['Target'].map({'Graduate': 1, 'Dropout': 0})

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features and labels
X = df.drop('Target', axis=1).values
y = df['Target'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# --- AdaBoost using SGDClassifier ---
class AdaBoost:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []
        self.train_errors = []

    def fit(self, X, y):
        n_samples = len(y)
        w = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_estimators):
            clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
            clf.fit(X, y, sample_weight=w)
            y_pred = clf.predict(X)

            miss = (y_pred != y).astype(int)
            error = np.dot(w, miss)
            EPS = 1e-10

            if error == 0:
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
        preds = np.array([alpha * (2 * clf.predict(X) - 1) for clf, alpha in zip(self.learners, self.alphas)])
        result = np.sign(preds.sum(axis=0))
        return ((result + 1) // 2).astype(int)

# --- Gradient Boosting using SGDRegressor ---
class GBM:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learners = []
        self.init_prediction = 0
        self.train_errors = []

    def fit(self, X, y):
        N = len(y)
        self.init_prediction = np.mean(y)
        current_preds = np.full(N, self.init_prediction)

        for _ in range(self.n_estimators):
            residuals = y - current_preds
            reg = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
            reg.fit(X, residuals)
            pred = reg.predict(X)

            current_preds += self.learning_rate * pred
            self.base_learners.append(reg)

            binary_preds = (current_preds >= 0.5).astype(int)
            error = 1 - accuracy_score(y, binary_preds)
            self.train_errors.append(error)

    def predict(self, X):
        preds = np.full(X.shape[0], self.init_prediction)
        for reg in self.base_learners:
            preds += self.learning_rate * reg.predict(X)
        return (preds >= 0.5).astype(int)

# Train and evaluate
ada = AdaBoost(n_estimators=100)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
ada_acc = accuracy_score(y_test, ada_pred)

gbm = GBM(n_estimators=100, learning_rate=0.1)
gbm.fit(X_train, y_train)
gbm_pred = gbm.predict(X_test)
gbm_acc = accuracy_score(y_test, gbm_pred)

print("AdaBoost Accuracy:", round(ada_acc * 100, 2), "%")
print("Gradient Boost Accuracy:", round(gbm_acc * 100, 2), "%")

# Paired t-test
t_stat, p_val = ttest_rel(ada_pred, gbm_pred)
print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.4f}")

# Plot training errors
plt.figure(figsize=(8, 5))
plt.plot(ada.train_errors, label="AdaBoost")
plt.plot(gbm.train_errors, label="Gradient Boost")
plt.title("Training Error Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, ada_pred)).plot(ax=axes[0], values_format='d')
axes[0].set_title("AdaBoost Confusion Matrix")

ConfusionMatrixDisplay(confusion_matrix(y_test, gbm_pred)).plot(ax=axes[1], values_format='d')
axes[1].set_title("Gradient Boost Confusion Matrix")
plt.tight_layout()
plt.show()

# Disagreement plot
diffs = (ada_pred != gbm_pred).astype(int)
print("Number of samples where predictions differ:", np.sum(diffs))

plt.figure(figsize=(8, 4))
plt.plot(diffs, marker='o', linestyle='None', alpha=0.6)
plt.title("Disagreement Between AdaBoost and Gradient Boost Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Disagreement (1 = mismatch)")
plt.grid(True)
plt.show()
