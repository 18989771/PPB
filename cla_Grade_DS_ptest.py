import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel

# Load dataset
data = pd.read_csv("grade.csv", sep=';')

# Filter for binary classification (Dropout=0, Graduate=1)
binary_data = data[data['Target'].isin(['Dropout', 'Graduate'])].copy()
binary_data['Target'] = binary_data['Target'].map({'Dropout': 0, 'Graduate': 1})

# Encode categorical features (except 'Target')
for col in binary_data.columns:
    if binary_data[col].dtype == 'object' and col != 'Target':
        binary_data[col] = LabelEncoder().fit_transform(binary_data[col])

X = binary_data.drop('Target', axis=1).values
y = binary_data['Target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# --- AdaBoost Implementation with max_depth placeholder ---
class Stump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        min_error = float('inf')
        for feature_i in range(n_features):
            thresholds = np.unique(X[:, feature_i])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    predictions[polarity * X[:, feature_i] < polarity * threshold] = 0
                    error = sum(sample_weights[i] for i in range(n_samples) if predictions[i] != y[i])
                    if error < min_error:
                        self.feature_index = feature_i
                        self.threshold = threshold
                        self.polarity = polarity
                        min_error = error
        EPS = 1e-10
        self.alpha = 0.5 * np.log((1 - min_error + EPS) / (min_error + EPS)) if min_error < 0.5 else 0

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_values = X[:, self.feature_index]
        predictions[self.polarity * feature_values < self.polarity * self.threshold] = 0
        return predictions

class AdaBoost:
    def __init__(self, n_estimators=10, max_depth=1):
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []
        self.train_errors = []
        self.max_depth = max_depth  # placeholder, currently unused

    def fit(self, X, y):
        n_samples = len(y)
        w = np.full(n_samples, 1 / n_samples)
        for _ in range(self.n_estimators):
            stump = Stump()
            stump.fit(X, y, w)
            predictions = stump.predict(X)
            error = sum(w[i] for i in range(n_samples) if predictions[i] != y[i])
            error = max(error, 1e-10)
            alpha = 0.5 * np.log((1 - error) / error)
            stump.alpha = alpha
            w = np.array([
                w[i] * np.exp(-alpha * (1 if y[i] == predictions[i] else -1))
                for i in range(n_samples)
            ])
            w /= w.sum()
            self.learners.append(stump)
            self.alphas.append(alpha)
            y_pred_train = self.predict(X)
            train_error = 1 - accuracy_score(y, y_pred_train)
            self.train_errors.append(train_error)

    def predict(self, X):
        learner_preds = np.array([learner.predict(X) for learner in self.learners])
        weighted_sum = np.dot(self.alphas, learner_preds)
        return np.where(weighted_sum >= 0.5 * sum(self.alphas), 1, 0)

# --- Gradient Boosting Machine with Stumps ---

class GBMStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        best_error = float('inf')
        n_samples, n_features = X.shape
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left_mask = feature_values < threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_mean = residuals[left_mask].mean()
                right_mean = residuals[right_mask].mean()
                preds = np.where(left_mask, left_mean, right_mean)
                error = np.sum((residuals - preds) ** 2)
                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_mean
                    self.right_value = right_mean

    def predict(self, X):
        feature_values = X[:, self.feature_index]
        preds = np.where(feature_values < self.threshold, self.left_value, self.right_value)
        return preds

class GBM:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learners = []
        self.init_prediction = 0
        self.train_errors = []
        self.max_depth = max_depth  # placeholder, currently unused

    def fit(self, X, y):
        N = len(y)
        self.init_prediction = y.mean()
        current_preds = np.full(N, self.init_prediction)
        for m in range(self.n_estimators):
            residuals = y - current_preds
            stump = GBMStump()
            stump.fit(X, residuals)
            preds = stump.predict(X)
            current_preds += self.learning_rate * preds
            self.base_learners.append(stump)
            binary_preds = (current_preds >= 0.5).astype(int)
            error = 1 - accuracy_score(y, binary_preds)
            self.train_errors.append(error)

    def predict(self, X):
        N = X.shape[0]
        preds = np.full(N, self.init_prediction)
        for stump in self.base_learners:
            preds += self.learning_rate * stump.predict(X)
        return (preds >= 0.5).astype(int)


# ---- Train and Evaluate AdaBoost ----
ada = AdaBoost(n_estimators=100, max_depth=3)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)
ada_acc = accuracy_score(y_test, ada_pred)

# ---- Train and Evaluate GBM ----
gbm = GBM(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X_train, y_train)
gbm_pred = gbm.predict(X_test)
gbm_acc = accuracy_score(y_test, gbm_pred)

print("AdaBoost Accuracy:", ada_acc)
print("Gradient Boost Accuracy:", gbm_acc)

# --- Paired t-test on predictions (note: this is unconventional for classification predictions)
t_stat, p_val = ttest_rel(ada_pred, gbm_pred)
print(f"Paired t-test result: t = {t_stat:.4f}, p = {p_val:.4f}")

# --- Plot training errors
plt.plot(range(1, len(ada.train_errors) + 1), ada.train_errors, label="AdaBoost")
plt.plot(range(1, len(gbm.train_errors) + 1), gbm.train_errors, label="GBM")
plt.title("Training Error Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.legend()
plt.grid(True)
plt.show()

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm_ada = confusion_matrix(y_test, ada_pred)
cm_gb = confusion_matrix(y_test, gbm_pred)

ConfusionMatrixDisplay(cm_ada).plot(ax=axes[0], values_format='d')
axes[0].set_title("AdaBoost Confusion Matrix")

ConfusionMatrixDisplay(cm_gb).plot(ax=axes[1], values_format='d')
axes[1].set_title("Gradient Boost Confusion Matrix")

plt.tight_layout()
plt.show()

# --- Compare predictions ---
diffs = (ada_pred != gbm_pred).astype(int)
print("Number of samples where predictions differ:", np.sum(diffs))

# Plot disagreement
plt.figure(figsize=(8, 4))
plt.plot(diffs, marker='o', linestyle='None', alpha=0.6)
plt.title("Disagreement Between AdaBoost and Gradient Boost Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Disagreement (1 = mismatch)")
plt.grid(True)
plt.show()
