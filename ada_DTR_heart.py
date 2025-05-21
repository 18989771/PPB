import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load and preprocess Heart Disease dataset
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]

data = pd.read_csv("processed.cleveland.data", names=column_names, na_values="?")
data.dropna(inplace=True)

# Binary classification: 0 = no disease, 1 = disease
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Encode categorical features
for col in ['cp', 'slope', 'thal']:
    data[col] = LabelEncoder().fit_transform(data[col])

X = data.drop('target', axis=1).values
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# AdaBoost with Decision Trees and fixed weight update
class AdaBoost:
    def __init__(self, n_estimators=10, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learners = []
        self.alphas = []
        self.train_errors = []

    def fit(self, X, y):
        n_samples = len(y)
        w = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(X, y, sample_weight=w)
            predictions = clf.predict(X)

            # Compute weighted error
            incorrect = (predictions != y)
            error = np.dot(w, incorrect)

            if error == 0:
                error = 1e-10

            # Compute alpha
            alpha = 0.5 * np.log((1 - error) / error)

            # Update sample weights (convert y to {-1, 1})
            y_converted = 2 * y - 1
            pred_converted = 2 * predictions - 1
            w *= np.exp(-alpha * y_converted * pred_converted)
            w /= w.sum()

            self.learners.append(clf)
            self.alphas.append(alpha)

            # Training error
            y_pred_train = self.predict(X)
            train_error = 1 - accuracy_score(y, y_pred_train)
            self.train_errors.append(train_error)

    def predict(self, X):
        learner_preds = np.array([
            (2 * clf.predict(X) - 1) for clf in self.learners  # Convert {0,1} → {-1,1}
        ])
        weighted_sum = np.dot(self.alphas, learner_preds)
        return np.where(weighted_sum >= 0, 1, 0)


# Train model
model = AdaBoost(n_estimators=50, max_depth=4)
model.fit(X_train, y_train)

# Test set predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Results
print("AdaBoost Accuracy on Heart Disease dataset:", accuracy)
print("True labels:", y_test)
print("Predictions:", y_pred)

# Plot training error
plt.figure(figsize=(8, 5))
plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o', linestyle='-')
plt.title("AdaBoost Training Error Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.ylim(0, 0.5)
plt.grid(True)
plt.show()
