import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("grade.csv", sep=';')  # Replace with your path if needed

# Filter for binary classification: Dropout vs Graduate
binary_data = data[data['Target'].isin(['Dropout', 'Graduate'])].copy()
binary_data['Target'] = binary_data['Target'].map({'Dropout': 0, 'Graduate': 1})

# Encode categorical columns
for col in binary_data.columns:
    if binary_data[col].dtype == 'object':
        binary_data[col] = LabelEncoder().fit_transform(binary_data[col])

# Feature matrix and target vector
X = binary_data.drop('Target', axis=1).values
y = binary_data['Target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Decision Stump
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

# AdaBoost Classifier
class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []
        self.train_errors = []

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

# Train model
model = AdaBoost(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("AdaBoost Accuracy on Student Dropout Dataset:", round(accuracy * 100, 2), "%")
print("True labels (sample):", y_test[:20])
print("Predictions (sample):", y_pred[:20])

# Plot training error over iterations
plt.figure(figsize=(8, 5))
plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o')
plt.title("AdaBoost Training Error Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.ylim(0, 0.5)
plt.grid(True)
plt.show()
