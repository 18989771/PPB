import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel

class AdaBoostSGD:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_iter_sgd=1000):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_iter_sgd = max_iter_sgd
        self.learners = []
        self.alphas = []
        self.rmse_iter = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        combined_pred = np.zeros(n_samples)

        for m in range(self.n_estimators):
            learner = SGDRegressor(max_iter=self.max_iter_sgd, tol=1e-3, random_state=42)
            learner.fit(X, y, sample_weight=w)
            pred = learner.predict(X)

            err_m = np.sum(w * np.abs(y - pred)) / np.sum(w)
            err_m = np.clip(err_m, 1e-10, 1 - 1e-10)

            if err_m <= 1e-10:
                print(f"Stopping early at iteration {m+1} due to perfect learner.")
                break

            alpha_m = self.learning_rate * np.log((1 - err_m) / err_m)
            w = w * np.exp(alpha_m * np.abs(y - pred))

            if np.sum(w) == 0 or np.isnan(np.sum(w)):
                print(f"Stopping early at iteration {m+1} due to invalid weights.")
                break

            w /= np.sum(w)
            combined_pred += alpha_m * pred

            self.learners.append(learner)
            self.alphas.append(alpha_m)

            rmse = np.sqrt(mean_squared_error(y, combined_pred / np.sum(self.alphas)))
            self.rmse_iter.append(rmse)

    def predict(self, X):
        combined_pred = np.zeros(X.shape[0])
        for alpha, learner in zip(self.alphas, self.learners):
            combined_pred += alpha * learner.predict(X)
        return combined_pred / np.sum(self.alphas)


# Load and preprocess grade.csv
df = pd.read_csv("grade.csv", sep=";")

# Drop rows with missing target
df = df.dropna(subset=["Admission grade"])

# Encode categorical features
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("Admission grade", axis=1).values
y = df["Admission grade"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit AdaBoost with SGD weak learners
ada = AdaBoostSGD(n_estimators=100, learning_rate=0.1, max_iter_sgd=1000)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)

# Fit Gradient Boosting
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm.fit(X_train, y_train)

# Track GBM training RMSE
rmse_gbm = []
for y_pred_stage in gbm.staged_predict(X_train):
    rmse_gbm.append(np.sqrt(mean_squared_error(y_train, y_pred_stage)))

y_pred_gbm = gbm.predict(X_test)

# Print performance metrics
def print_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} RMSE: {rmse:.4f} | R²: {r2:.4f}")

print_metrics("AdaBoost", y_test, y_pred_ada)
print_metrics("Gradient Boosting", y_test, y_pred_gbm)

# Paired t-test on squared errors
squared_errors_ada = (y_test - y_pred_ada) ** 2
squared_errors_gbm = (y_test - y_pred_gbm) ** 2
t_stat, p_val = ttest_rel(squared_errors_ada, squared_errors_gbm)
print(f"Paired t-test on squared errors: t = {t_stat:.4f}, p = {p_val:.4f}")

# Plot RMSE over iterations
plt.plot(range(1, len(ada.rmse_iter) + 1), ada.rmse_iter, label="AdaBoost (train RMSE)")
plt.plot(range(1, len(rmse_gbm) + 1), rmse_gbm, label="Gradient Boosting (train RMSE)")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("Training RMSE vs Iterations (grade.csv)")
plt.legend()
plt.grid(True)
plt.show()
