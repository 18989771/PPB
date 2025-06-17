import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel

# --- Load and preprocess dataset ---
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]

data = pd.read_csv("processed.cleveland.data", names=column_names, na_values="?")
data.dropna(inplace=True)

# Encode categorical columns
for col in ['cp', 'slope', 'thal']:
    data[col] = LabelEncoder().fit_transform(data[col])

# Use 'thalach' as the regression target
y = data['thalach'].values.astype(float)
X = data.drop(['thalach'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


class StumpRegressor:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        best_error = float('inf')
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_mask = feature_values < threshold
                right_mask = feature_values >= threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_mean = np.mean(residuals[left_mask])
                right_mean = np.mean(residuals[right_mask])

                error = np.sum((residuals[left_mask] - left_mean) ** 2) + np.sum((residuals[right_mask] - right_mean) ** 2)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_mean
                    self.right_value = right_mean

    def predict(self, X):
        feature_values = X[:, self.feature_index]
        return np.where(feature_values < self.threshold, self.left_value, self.right_value)

class AdaBoostRegressor:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.learners = []
        self.train_errors = []

    def fit(self, X, y):
        n_samples = len(y)
        current_preds = np.full(n_samples, np.mean(y))

        for _ in range(self.n_estimators):
            residuals = y - current_preds
            stump = StumpRegressor()
            stump.fit(X, residuals)
            preds = stump.predict(X)

            current_preds += preds  # <-- NO learning rate applied
            self.learners.append(stump)

            train_rmse = np.sqrt(mean_squared_error(y, current_preds))
            self.train_errors.append(train_rmse)

    def predict(self, X):
        n_samples = X.shape[0]
        preds = np.zeros(n_samples)
        for stump in self.learners:
            preds += stump.predict(X)
        preds += np.mean(y_train)
        return preds

# --- GBM Regressor stays unchanged ---
class GBMStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        best_error = float('inf')
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_mask = feature_values < threshold
                right_mask = feature_values >= threshold

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_mean = np.mean(residuals[left_mask])
                right_mean = np.mean(residuals[right_mask])

                error = np.sum((residuals[left_mask] - left_mean) ** 2) + np.sum((residuals[right_mask] - right_mean) ** 2)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_mean
                    self.right_value = right_mean

    def predict(self, X):
        feature_values = X[:, self.feature_index]
        return np.where(feature_values < self.threshold, self.left_value, self.right_value)

class GBMRegressor:
    def __init__(self, n_estimators=50, learning_rate=0.05):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.learners = []
        self.init_pred = 0
        self.train_errors = []

    def fit(self, X, y):
        n_samples = len(y)
        self.init_pred = np.mean(y)
        current_preds = np.full(n_samples, self.init_pred)

        for _ in range(self.n_estimators):
            residuals = y - current_preds
            stump = GBMStump()
            stump.fit(X, residuals)
            preds = stump.predict(X)
            current_preds += self.learning_rate * preds
            self.learners.append(stump)

            train_rmse = np.sqrt(mean_squared_error(y, current_preds))
            self.train_errors.append(train_rmse)

    def predict(self, X):
        n_samples = X.shape[0]
        preds = np.full(n_samples, self.init_pred)
        for stump in self.learners:
            preds += self.learning_rate * stump.predict(X)
        return preds

# --- Train and Evaluate ---
ada = AdaBoostRegressor(n_estimators=100)
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

gbm = GBMRegressor(n_estimators=100, learning_rate=0.1)
gbm.fit(X_train, y_train)
gbm_pred = gbm.predict(X_test)

# Metrics
ada_rmse = np.sqrt(mean_squared_error(y_test, ada_pred))
ada_r2 = r2_score(y_test, ada_pred)
gbm_rmse = np.sqrt(mean_squared_error(y_test, gbm_pred))
gbm_r2 = r2_score(y_test, gbm_pred)

print(f"AdaBoost Regressor RMSE: {ada_rmse:.4f}, R^2: {ada_r2:.4f}")
print(f"GBM Regressor RMSE: {gbm_rmse:.4f}, R^2: {gbm_r2:.4f}")

# t-test on predictions
t_stat, p_val = ttest_rel(ada_pred, gbm_pred)
print(f"Paired t-test on predictions: t = {t_stat:.4f}, p = {p_val:.4f}")

# Training error plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ada.train_errors) + 1), ada.train_errors, label='AdaBoost Regressor (no learning rate)')
plt.plot(range(1, len(gbm.train_errors) + 1), gbm.train_errors, label='GBM Regressor')
plt.xlabel('Iteration')
plt.ylabel('Training RMSE')
plt.title('Training RMSE Over Iterations')
plt.legend()
plt.grid(True)
plt.show()

# Plot prediction differences
diffs = np.abs(ada_pred - gbm_pred)
plt.figure(figsize=(10, 4))
plt.plot(diffs, marker='o', linestyle='None', alpha=0.7)
plt.title("Absolute Differences Between AdaBoost and GBM Predictions")
plt.xlabel("Test Sample Index")
plt.ylabel("Absolute Prediction Difference")
plt.grid(True)
plt.show()
