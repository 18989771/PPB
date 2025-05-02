import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Wine dataset
data = load_wine()
X = data.data
y = data.target

# Use only classes 0 and 1 for binary classification
X = X[y != 2]
y = y[y != 2]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Define regression stump
class Stump:
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

# Gradient Boosting Machine
class GBM:
    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learners = []
        self.init_prediction = 0
        self.train_errors = []  # For plotting

    def fit(self, X, y):
        N = len(y)
        self.init_prediction = sum(y) / N
        current_preds = [self.init_prediction] * N

        for m in range(self.n_estimators):
            residuals = [y[i] - current_preds[i] for i in range(N)]
            stump = Stump()
            stump.fit(X, residuals)
            predictions = stump.predict(X)

            current_preds = [
                current_preds[i] + self.learning_rate * predictions[i]
                for i in range(N)
            ]
            self.base_learners.append(stump)

            # Convert predictions to binary and record training error
            binary_preds = [1 if p >= 0.5 else 0 for p in current_preds]
            error = 1 - accuracy_score(y, binary_preds)
            self.train_errors.append(error)

    def predict(self, X):
        N = len(X)
        preds = [self.init_prediction] * N
        for stump in self.base_learners:
            stump_preds = stump.predict(X)
            preds = [preds[i] + self.learning_rate * stump_preds[i] for i in range(N)]
        return preds

# Train model
model = GBM(n_estimators=10, learning_rate=0.1)
model.fit(X_train.tolist(), y_train.tolist())

# Predict
y_pred = model.predict(X_test.tolist())
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy on test set:", accuracy)

# Plot training error over iterations
plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o')
plt.title("GBM Training Error Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.grid(True)
plt.show()
