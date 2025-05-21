import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load and filter Wine dataset for binary classification (class 0 vs 1)
data = load_wine()
X = data.data
y = data.target
X = X[y != 2]
y = y[y != 2]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# AdaBoost with Decision Trees
class AdaBoost:
    def __init__(self, n_estimators=10, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learners = []
        self.alphas = []
        self.train_errors = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.full(n_samples, 1 / n_samples)
        y_mod = 2 * y - 1  # Convert to {-1, 1}
        pred_sum = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            # Train decision tree with sample weights
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(X, y, sample_weight=w)
            pred = clf.predict(X)
            pred_mod = 2 * pred - 1

            # Compute weighted error
            error = np.sum(w[pred != y])
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Compute alpha
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            self.learners.append(clf)

            # Update weights
            w *= np.exp(-alpha * y_mod * pred_mod)
            w /= np.sum(w)

            # Compute training error
            pred_sum += alpha * pred_mod
            y_final = (np.sign(pred_sum) + 1) // 2
            train_error = 1 - accuracy_score(y, y_final)
            self.train_errors.append(train_error)

    def predict(self, X):
        pred_sum = np.zeros(X.shape[0])
        for clf, alpha in zip(self.learners, self.alphas):
            pred = clf.predict(X)
            pred_mod = 2 * pred - 1
            pred_sum += alpha * pred_mod
        return (np.sign(pred_sum) + 1) // 2

# Train and evaluate
model = AdaBoost(n_estimators=50, max_depth=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Predicted labels:", y_pred.astype(int))
print("True labels:", y_test)

# Plot learning curve
plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o')
plt.title("AdaBoost Training Error with Decision Trees")
plt.xlabel("Iteration")
plt.ylabel("Training Error")
plt.grid(True)
plt.show()
