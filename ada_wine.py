import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and filter Wine dataset for binary classification (class 0 vs 1)
data = load_wine()
X = data.data
y = data.target
X = X[y != 2]
y = y[y != 2]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Decision stump
class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, w):
        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_i in range(n_features):
            feature_values = X[:, feature_i]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = 0
                    else:
                        predictions[feature_values >= threshold] = 0

                    error = np.sum(w[predictions != y])
                    if error < min_error:
                        self.polarity = polarity
                        self.threshold = threshold
                        self.feature_index = feature_i
                        min_error = error

        EPS = 1e-10
        self.alpha = 0.5 * np.log((1 - min_error + EPS) / (min_error + EPS))

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_values = X[:, self.feature_index]
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = 0
        else:
            predictions[feature_values >= self.threshold] = 0
        return predictions

# AdaBoost
class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []
        self.train_errors = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.full(n_samples, (1 / n_samples))
        y_mod = 2 * y - 1  # Convert to {-1, 1}

        pred_sum = np.zeros(n_samples)

        for i in range(self.n_clf):
            clf = DecisionStump()
            clf.fit(X, y, w)
            predictions = clf.predict(X)
            pred_mod = 2 * predictions - 1

            # Update weights
            w *= np.exp(-clf.alpha * y_mod * pred_mod)
            w /= np.sum(w)

            self.clfs.append(clf)

            # Accumulate prediction and compute training error
            pred_sum += clf.alpha * pred_mod
            final_pred = np.sign(pred_sum)
            final_binary = (final_pred + 1) // 2
            error = 1 - accuracy_score(y, final_binary)
            self.train_errors.append(error)

    def predict(self, X):
        clf_preds = [clf.alpha * (2 * clf.predict(X) - 1) for clf in self.clfs]
        y_pred = np.sign(np.sum(clf_preds, axis=0))
        return (y_pred + 1) // 2

# Train and evaluate
model = AdaBoost(n_clf=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Predicted labels:", y_pred)
print("True labels:", y_test)

# Plot learning curve
plt.plot(range(1, model.n_clf + 1), model.train_errors, marker='o')
plt.title("AdaBoost Training Error Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Training Error")
plt.grid(True)
plt.show()