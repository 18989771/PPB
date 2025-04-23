import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and filter Wine dataset for binary classification (class 0 vs 1)
data = load_wine()
X = data.data
y = data.target

# Use only classes 0 and 1 (data set has three classes ie. three different types of wine
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

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_clf):
            clf = DecisionStump()
            clf.fit(X, y, w)
            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * (2 * y - 1) * (2 * predictions - 1))
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * (2 * clf.predict(X) - 1) for clf in self.clfs]
        y_pred = np.sign(np.sum(clf_preds, axis=0))
        return (y_pred + 1) // 2

# Train and test
model = AdaBoost(n_clf=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print(y_pred)
print(y_test)