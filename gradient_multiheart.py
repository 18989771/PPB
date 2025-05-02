import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load and preprocess Heart Disease dataset
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]

# Load data 'processed.cleveland.data'
data = pd.read_csv("processed.cleveland.data", names=column_names, na_values="?")
data.dropna(inplace=True)

# Convert target to binary (0 = no disease, 1 = any heart disease)
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Encode categorical columns
for col in ['cp', 'slope', 'thal']:
    data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop('target', axis=1).values
y = data['target'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Decision Stump for GBM
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
import matplotlib.pyplot as plt

# Gradient Boosting Machine with training error tracking
class GBM:
    def __init__(self, n_estimators=10, learning_rate=0.1):
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
            # Compute residuals
            residuals = [y[i] - current_preds[i] for i in range(N)]

            # Fit stump
            stump = Stump()
            stump.fit(X, residuals)
            predictions = stump.predict(X)

            # Update predictions
            current_preds = [
                current_preds[i] + self.learning_rate * predictions[i]
                for i in range(N)
            ]
            self.base_learners.append(stump)

            # Compute binary predictions and training error
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

# Train GBM model on Heart Disease dataset
model = GBM(n_estimators=10, learning_rate=0.1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)

print("Accuracy on Heart Disease dataset:", accuracy)
print("True labels:", y_test)
print("Predictions:", y_pred_binary)


# Plot training error over iterations
plt.figure(figsize=(8, 5))
plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o', linestyle='-')
plt.title("GBM Training Error Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Training Error")
plt.ylim(0, 0.5)
plt.grid(True)
plt.show()
