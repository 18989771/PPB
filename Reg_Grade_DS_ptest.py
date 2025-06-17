import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel
from math import sqrt

# --- Load and preprocess the dataset ---
df = pd.read_csv('grade.csv', sep=';')
df = df.dropna(subset=["Admission grade"])

# Encode categorical columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(columns=["Admission grade"]).values
y = df["Admission grade"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Decision stump for regression ---
class RegressionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y, weights):
        m, n = X.shape
        min_error = float('inf')

        for feature_i in range(n):
            thresholds = np.unique(X[:, feature_i])
            for threshold in thresholds:
                left_mask = X[:, feature_i] < threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                if np.sum(weights[left_mask]) == 0 or np.sum(weights[right_mask]) == 0:
                    continue

                left_val = np.average(y[left_mask], weights=weights[left_mask])
                right_val = np.average(y[right_mask], weights=weights[right_mask])

                preds = np.where(left_mask, left_val, right_val)
                error = np.average((y - preds) ** 2, weights=weights)

                if error < min_error:
                    min_error = error
                    self.feature_index = feature_i
                    self.threshold = threshold
                    self.left_value = left_val
                    self.right_value = right_val

        # fallback if no valid split found
        if self.feature_index is None:
            self.feature_index = 0
            self.threshold = np.median(X[:, 0])
            self.left_value = np.average(y, weights=weights)
            self.right_value = self.left_value

    def predict(self, X):
        feature = X[:, self.feature_index]
        return np.where(feature < self.threshold, self.left_value, self.right_value)

# --- Custom AdaBoost Regressor ---
class AdaBoostRegressorCustom:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []
        self.train_errors = []

    def fit(self, X, y):
        n = len(y)
        weights = np.full(n, 1 / n)

        for t in range(self.n_estimators):
            stump = RegressionStump()
            stump.fit(X, y, weights)
            predictions = stump.predict(X)

            error = np.average((y - predictions) ** 2, weights=weights)
            error = np.clip(error, 1e-10, 1 - 1e-10)  # avoid log(0)

            alpha = 0.5 * np.log((1 - error) / error)

            exponent = -alpha * (y - predictions) ** 2
            exponent = np.clip(exponent, -50, 50)  # clip exponent to avoid overflow
            weights *= np.exp(exponent)

            weights_sum = np.sum(weights)
            if weights_sum == 0 or np.isnan(weights_sum):
                print("Terminating early due to invalid weights.")
                break
            weights /= weights_sum

            self.learners.append(stump)
            self.alphas.append(alpha)

            train_pred = self.predict(X)
            self.train_errors.append(mean_squared_error(y, train_pred))

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for alpha, learner in zip(self.alphas, self.learners):
            pred += alpha * learner.predict(X)
        return pred / (np.sum(self.alphas) + 1e-10)

# --- Custom Gradient Boosting Regressor ---
class GBMStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        m, n = X.shape
        best_error = float('inf')

        for feature_index in range(n):
            feature_values = X[:, feature_index]
            for threshold in np.unique(feature_values):
                left_mask = feature_values < threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_mean = np.mean(residuals[left_mask])
                right_mean = np.mean(residuals[right_mask])

                preds = np.where(left_mask, left_mean, right_mean)
                error = np.mean((residuals - preds) ** 2)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_mean
                    self.right_value = right_mean

    def predict(self, X):
        feature = X[:, self.feature_index]
        return np.where(feature < self.threshold, self.left_value, self.right_value)

class GradientBoostingRegressorCustom:
    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.init_value = None
        self.learners = []
        self.train_errors = []

    def fit(self, X, y):
        n = len(y)
        self.init_value = np.mean(y)
        pred = np.full(n, self.init_value)

        for _ in range(self.n_estimators):
            residuals = y - pred
            stump = GBMStump()
            stump.fit(X, residuals)
            update = stump.predict(X)
            pred += self.learning_rate * update

            self.learners.append(stump)
            self.train_errors.append(mean_squared_error(y, pred))

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_value)
        for stump in self.learners:
            pred += self.learning_rate * stump.predict(X)
        return pred

# --- Train and Evaluate ---
ada = AdaBoostRegressorCustom(n_estimators=50)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

gbm = GradientBoostingRegressorCustom(n_estimators=50, learning_rate=0.1)
gbm.fit(X_train, y_train)
gbm_pred = gbm.predict(X_test)

print("AdaBoost RMSE:", sqrt(mean_squared_error(y_test, ada_pred)))
print("GBM RMSE:", sqrt(mean_squared_error(y_test, gbm_pred)))

print("AdaBoost R2:", r2_score(y_test, ada_pred))
print("GBM R2:", r2_score(y_test, gbm_pred))

# Paired t-test
t_stat, p_val = ttest_rel(ada_pred, gbm_pred)
print(f"Paired t-test: t = {t_stat:.4f}, p = {p_val:.4f}")

# --- Plot training errors ---
plt.figure(figsize=(10, 5))
plt.plot(np.sqrt(ada.train_errors), label='AdaBoost RMSE')
plt.plot(np.sqrt(gbm.train_errors), label='GBM RMSE')
plt.xlabel("Iterations")
plt.ylabel("Training RMSE")
plt.title("Training Error Over Iterations")
plt.legend()
plt.grid(True)
plt.show()

# --- Prediction scatter ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, ada_pred, alpha=0.7)
plt.title("AdaBoost Predictions")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test, gbm_pred, alpha=0.7, color='orange')
plt.title("GBM Predictions")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()
