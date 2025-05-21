import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor

# Load and preprocess Heart Disease dataset
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the weak learner using sklearn DecisionTreeRegressor
class TreeWeakLearner:
    def __init__(self, max_depth=2):
        self.tree = DecisionTreeRegressor(max_depth=max_depth)

    def fit(self, X, residuals):
        self.tree.fit(X, residuals)

    def predict(self, X):
        return self.tree.predict(X)

# Updated GBM class
class GBM:
    def __init__(self, n_estimators=50, learning_rate=0.05, max_depth=4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.base_learners = []
        self.init_prediction = 0
        self.train_errors = []

    def fit(self, X, y):
        N = len(y)
        self.init_prediction = sum(y) / N  # mean of target
        current_preds = [self.init_prediction] * N

        for m in range(self.n_estimators):
            # Compute residuals
            residuals = [y[i] - current_preds[i] for i in range(N)]

            # Fit a weak learner
            learner = TreeWeakLearner(max_depth=self.max_depth)
            learner.fit(X, residuals)
            predictions = learner.predict(X)

            # Update predictions
            current_preds = [
                current_preds[i] + self.learning_rate * predictions[i]
                for i in range(N)
            ]
            self.base_learners.append(learner)

            # Compute binary accuracy
            binary_preds = [1 if pred >= 0.5 else 0 for pred in current_preds]
            train_error = 1 - accuracy_score(y, binary_preds)
            self.train_errors.append(train_error)

    def predict(self, X):
        N = len(X)
        preds = [self.init_prediction] * N
        for learner in self.base_learners:
            stump_preds = learner.predict(X)
            preds = [preds[i] + self.learning_rate * stump_preds[i] for i in range(N)]
        return preds

# Train and evaluate the model
model = GBM(n_estimators=100, learning_rate=0.05, max_depth=2)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)

# Print results
print("Test Accuracy:", accuracy)
print("True labels:", y_test)
print("Predictions:", y_pred_binary)

# Plot training error
plt.figure(figsize=(8, 5))
plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o')
plt.title("Training Error over Iterations (Tree Depth=2)")
plt.xlabel("Iteration")
plt.ylabel("Training Error")
plt.ylim(0, 0.5)
plt.grid(True)
plt.show()

