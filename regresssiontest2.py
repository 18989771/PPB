#uses adaboost R2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

class AdaBoostR2:
    def __init__(self, n_estimators=100, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learners = []
        self.betas = []  # beta values related to errors
        self.train_errors = []

    def fit(self, X, y):
        N = len(y)
        weights = np.full(N, 1 / N)  # initialize sample weights

        for t in range(self.n_estimators):
            # Fit regressor with current weights
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, y, sample_weight=weights)
            pred = tree.predict(X)

            # Compute absolute errors
            errors = np.abs(y - pred)

            # Normalize errors by max error (avoid division by zero)
            max_error = errors.max()
            if max_error == 0:
                max_error = 1e-10  # avoid division by zero
            normalized_errors = errors / max_error

            # Compute weighted error
            weighted_error = np.sum(weights * normalized_errors)

            # Compute beta_t (learner weight)
            beta = weighted_error / (1 - weighted_error)
            self.betas.append(beta)
            self.learners.append(tree)

            # Update weights for next iteration
            # Weight update rule: w_i = w_i * beta^(1 - normalized_error_i)
            weights *= beta ** (1 - normalized_errors)
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                weights = np.full(N, 1 / N)
            else:
                weights /= weights_sum

            # Calculate training MSE at current iteration
            train_pred = self.predict(X)
            train_error = mean_squared_error(y, train_pred)
            self.train_errors.append(train_error)

    def predict(self, X):
        # Aggregate predictions weighted by log(1 / beta)
        if not self.learners:
            return np.zeros(X.shape[0])

        weighted_preds = np.zeros(X.shape[0])
        total_weight = 0
        for tree, beta in zip(self.learners, self.betas):
            weight = np.log(1 / beta)
            weighted_preds += weight * tree.predict(X)
            total_weight += weight

        return weighted_preds / total_weight


df = pd.read_csv('grade.csv',sep=';')


# Encode categorical features
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Features and target
X = df.drop(columns=["Admission grade"])
y = df["Admission grade"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

class GradientBoostRegressorCustom:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.learners = []
        self.init_pred = 0
        self.train_errors = []

    def fit(self, X, y):
        self.init_pred = np.mean(y)
        pred = np.full_like(y, self.init_pred)
        for _ in range(self.n_estimators):
            residual = y - pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residual)
            update = tree.predict(X)
            pred += self.learning_rate * update
            self.learners.append(tree)
            self.train_errors.append(mean_squared_error(y, pred))

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_pred)
        for tree in self.learners:
            pred += self.learning_rate * tree.predict(X)
        return pred

# Train AdaBoost Regressor
ada = AdaBoostR2(n_estimators=100, max_depth=3)
ada.fit(X_train, y_train)
ada_preds = ada.predict(X_test)

ada_rmse = mean_squared_error(y_test, ada_preds, squared=False)
ada_r2 = r2_score(y_test, ada_preds)


# Train Gradient Boosting Regressor
gbm = GradientBoostRegressorCustom(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm.fit(X_train, y_train)
gbm_preds = gbm.predict(X_test)
gbm_rmse = mean_squared_error(y_test, gbm_preds, squared=False)
gbm_r2 = r2_score(y_test, gbm_preds)

# Print results
print(f"Gradient Boost RMSE: {gbm_rmse:.4f}, R2: {gbm_r2:.4f}")
print(f"ADA Boost RMSE: {ada_rmse:.4f}, R2: {ada_r2:.4f}")
# Plot training errors
plt.plot(range(1, len(ada.train_errors) + 1), ada.train_errors, label="AdaBoost")
plt.plot(range(1, len(gbm.train_errors) + 1), gbm.train_errors, label="Gradient Boosting")
plt.xlabel("Iteration")
plt.ylabel("Training MSE")
plt.title("Training Error Over Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()