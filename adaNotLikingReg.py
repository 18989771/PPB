from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal', 'target'
]
data = pd.read_csv("processed.cleveland.data", names=column_names, na_values="?")
data.dropna(inplace=True)

# **Do not binarize target** — use original values
y = data['target'].values.astype(float)
X = data.drop('target', axis=1).values

# Encode categoricals as before
for col in ['cp', 'slope', 'thal']:
    data[col] = LabelEncoder().fit_transform(data[col])

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
import numpy as np
from sklearn.linear_model import SGDRegressor

class AdaBoostRegressorSGD:
    def __init__(self, n_estimators=50, max_iter=500, tol=1e-4):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.tol = tol
        self.learners = []
        self.alphas = []
        self.train_mse = []

    def fit(self, X, y):
        n = len(y)
        w = np.full(n, 1 / n)
        y_pred_ensemble = np.zeros(n)

        for i in range(self.n_estimators):
            model = SGDRegressor(max_iter=self.max_iter, tol=self.tol, random_state=42)
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            residuals = np.abs(y - y_pred)
            max_residual = residuals.max()

            if max_residual == 0:
                print(f"Perfect fit at iteration {i+1}")
                break

            # Normalize residuals to [0,1]
            normalized_residuals = residuals / max_residual

            # Weighted error = sum of weights * normalized residuals
            weighted_error = np.dot(w, normalized_residuals)

            # Stop if error is too high
            if weighted_error >= 0.5:
                print(f"Stopping early at iteration {i+1} due to weighted error >= 0.5")
                break

            # Compute alpha
            alpha = np.log((1 - weighted_error) / (weighted_error + 1e-10))

            # Update weights: w_i = w_i * exp(alpha * (1 - normalized_residual_i))
            w *= np.exp(alpha * (1 - normalized_residuals))
            w /= w.sum()

            # Save model and alpha
            self.learners.append(model)
            self.alphas.append(alpha)

            # Update ensemble prediction
            y_pred_ensemble += alpha * y_pred
            ensemble_pred = y_pred_ensemble / sum(self.alphas)

            mse = np.mean((y - ensemble_pred) ** 2)
            self.train_mse.append(mse)

            print(f"Iteration {i+1}: Weighted error = {weighted_error:.4f}, alpha = {alpha:.4f}, train MSE = {mse:.4f}")

    def predict(self, X):
        pred_sum = np.zeros(X.shape[0])
        alpha_sum = sum(self.alphas)
        for alpha, learner in zip(self.alphas, self.learners):
            pred_sum += alpha * learner.predict(X)
        return pred_sum / alpha_sum
# --- Gradient Boosting Regression (same as before, works for regression) ---
class GBM_SGD:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learners = []
        self.init_prediction = 0
        self.train_errors = []

    def fit(self, X, y):
        n = len(y)
        self.init_prediction = np.mean(y)
        current_preds = np.full(n, self.init_prediction)

        for _ in range(self.n_estimators):
            residuals = y - current_preds

            model = SGDRegressor(max_iter=100, tol=1e-3, random_state=42)
            model.fit(X, residuals)
            preds = model.predict(X)

            current_preds += self.learning_rate * preds
            self.base_learners.append(model)

            mse = mean_squared_error(y, current_preds)
            self.train_errors.append(mse)

    def predict(self, X):
        preds = np.full(X.shape[0], self.init_prediction)
        for model in self.base_learners:
            preds += self.learning_rate * model.predict(X)
        return preds

# Train regression models
ada_reg = AdaBoostRegressorSGD(n_estimators=100, max_iter=100)
ada_reg.fit(X_train, y_train)
ada_reg_pred = ada_reg.predict(X_test)
ada_reg_mse = mean_squared_error(y_test, ada_reg_pred)

gbm_reg = GBM_SGD(n_estimators=50, learning_rate=0.1)
gbm_reg.fit(X_train, y_train)
gbm_reg_pred = gbm_reg.predict(X_test)
gbm_reg_mse = mean_squared_error(y_test, gbm_reg_pred)

print(f"AdaBoost Regression MSE: {ada_reg_mse:.4f}")
print(f"Gradient Boosting Regression MSE: {gbm_reg_mse:.4f}")

# Plot training error (MSE) over iterations
plt.plot(range(1, len(ada_reg.train_mse)+1), ada_reg.train_mse, label="AdaBoost Regression")
plt.plot(range(1, len(gbm_reg.train_errors)+1), gbm_reg.train_errors, label="GBM Regression")
plt.xlabel("Iterations")
plt.ylabel("Training MSE")
plt.title("Training Mean Squared Error over Iterations")
plt.legend()
plt.grid(True)
plt.show()
