import numpy as np
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

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Define Stump (Weak Learner) for regression tasks
class Stump:
    def __init__(self):
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        best_error = float('inf')
        for i in range(len(X)):
            threshold = X[i][0]
            left = [residuals[j] for j in range(len(X)) if X[j][0] < threshold]
            right = [residuals[j] for j in range(len(X)) if X[j][0] >= threshold]

            if not left or not right:
                continue

            left_mean = sum(left) / len(left)
            right_mean = sum(right) / len(right)

            error = sum((residuals[j] - (left_mean if X[j][0] < threshold else right_mean))**2 for j in range(len(X)))

            if error < best_error:
                best_error = error
                self.threshold = threshold
                self.left_value = left_mean
                self.right_value = right_mean

    def predict(self, X):
        return [self.left_value if x[0] < self.threshold else self.right_value for x in X]

# Gradient Boosting Machine (GBM) for regression tasks
class GBM:
    def __init__(self, n_estimators=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learners = []
        self.init_prediction = 0

    def fit(self, X, y):
        N = len(y)
        self.init_prediction = sum(y) / N
        current_preds = [self.init_prediction] * N

        for m in range(self.n_estimators):
            residuals = [y[i] - current_preds[i] for i in range(N)]  #loss function

            stump = Stump()
            stump.fit(X, residuals)

            predictions = stump.predict(X)
            current_preds = [
                current_preds[i] + self.learning_rate * predictions[i]
                for i in range(N)
            ]

            self.base_learners.append(stump)

    def predict(self, X):
        N = len(X)
        preds = [self.init_prediction] * N
        for stump in self.base_learners:
            stump_preds = stump.predict(X)
            preds = [preds[i] + self.learning_rate * stump_preds[i] for i in range(N)]
        return preds

# Train the GBM model on Wine dataset
model = GBM(n_estimators=10, learning_rate=0.1)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Convert predictions to binary classes (using threshold of 0.5)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy on test set:", accuracy)
print(y_test)
print(y_pred)