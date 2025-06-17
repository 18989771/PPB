import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel

# --- Load and preprocess Heart Disease dataset ---
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]

data = pd.read_csv("processed.cleveland.data", names=column_names, na_values="?")
data.dropna(inplace=True)
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification

for col in ['cp', 'slope', 'thal']:
    data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop('target', axis=1).values
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# --- AdaBoost Implementation ---

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
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []
        self.train_errors = []

    def fit(self, X, y):
        n_samples = len(y)
        w = np.full(n_samples, 1 / n_samples)
        self.learners = []
        self.alphas = []
        self.train_errors = []

        for _ in range(self.n_estimators):
            stump = Stump()
            stump.fit(X, y, w)
            predictions = stump.predict(X)

            error = sum(w[i] for i in range(n_samples) if predictions[i] != y[i])
            if error == 0:
                error = 1e-10

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


# --- Gradient Boosting Machine Implementation ---

class GBMStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        best_error = float('inf')
        n_samples, n_features = len(X), len(X[0])

        for feature_index in range(n_features):
            feature_values = [x[feature_index] for x in X]
            for threshold in feature_values:
                left = [residuals[i] for i in range(n_samples) if X[i][feature_index] < threshold]
                right = [residuals[i] for i in range(n_samples) if X[i][feature_index] >= threshold]

                if not left or not right:
                    continue

                left_mean = sum(left) / len(left)
                right_mean = sum(right) / len(right)

                error = sum(
                    (residuals[i] - (left_mean if X[i][feature_index] < threshold else right_mean)) ** 2
                    for i in range(n_samples)
                )

                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_mean
                    self.right_value = right_mean

    def predict(self, X):
        return [
            self.left_value if x[self.feature_index] < self.threshold else self.right_value
            for x in X
        ]


class GBM:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learners = []
        self.init_prediction = 0
        self.train_errors = []

    def fit(self, X, y):
        N = len(y)
        self.init_prediction = sum(y) / N
        current_preds = [self.init_prediction] * N

        for m in range(self.n_estimators):
            residuals = [y[i] - current_preds[i] for i in range(N)]

            stump = GBMStump()
            stump.fit(X, residuals)
            predictions = stump.predict(X)

            current_preds = [
                current_preds[i] + self.learning_rate * predictions[i]
                for i in range(N)
            ]
            self.base_learners.append(stump)

            binary_preds = [1 if pred >= 0.5 else 0 for pred in current_preds]
            train_error = 1 - accuracy_score(y, binary_preds)
            self.train_errors.append(train_error)

    def predict(self, X):
        N = len(X)
        preds = [self.init_prediction] * N
        for stump in self.base_learners:
            stump_preds = stump.predict(X)
            preds = [preds[i] + self.learning_rate * stump_preds[i] for i in range(N)]
        return preds


# --- Train both models ---

ada = AdaBoost(n_estimators=100)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

gbm = GBM(n_estimators=100, learning_rate=0.1)
gbm.fit(X_train, y_train)
gbm_pred_prob = gbm.predict(X_test)
gbm_pred = [1 if p >= 0.5 else 0 for p in gbm_pred_prob]

# --- Evaluate and compare ---

ada_acc = accuracy_score(y_test, ada_pred)
gbm_acc = accuracy_score(y_test, gbm_pred)

print("AdaBoost Accuracy:", ada_acc)
print("Gradient Boost Accuracy:", gbm_acc)

# Paired t-test on predictions
t_stat, p_val = ttest_rel(ada_pred, gbm_pred)
print(f"Paired t-test result: t = {t_stat:.4f}, p = {p_val:.4f}")

# Plot training error over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ada.train_errors) + 1), ada.train_errors, label="AdaBoost")
plt.plot(range(1, len(gbm.train_errors) + 1), gbm.train_errors, label="GBM")
plt.title("Training Error Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_ada = confusion_matrix(y_test, ada_pred)
cm_gbm = confusion_matrix(y_test, gbm_pred)

ConfusionMatrixDisplay(cm_ada).plot(ax=axes[0], values_format='d')
axes[0].set_title("AdaBoost Confusion Matrix")

ConfusionMatrixDisplay(cm_gbm).plot(ax=axes[1], values_format='d')
axes[1].set_title("Gradient Boost Confusion Matrix")

plt.tight_layout()
plt.show()

# Disagreement analysis
diffs = (np.array(ada_pred) != np.array(gbm_pred)).astype(int)
print("Number of samples where predictions differ:", np.sum(diffs))

plt.figure(figsize=(10, 4))
plt.plot(diffs, marker='o', linestyle='None', alpha=0.7)
plt.title("Disagreement Between AdaBoost and Gradient Boost Predictions")
plt.xlabel("Test Sample Index")
plt.ylabel("Disagreement (1 = mismatch)")
plt.grid(True)
plt.show()
