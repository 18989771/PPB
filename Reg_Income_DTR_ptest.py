import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel

# Load Adult dataset
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

data = pd.read_csv("adult.data", names=column_names, na_values=" ?", skipinitialspace=True)
data.dropna(inplace=True)

# Encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# Features and regression target
X = data.drop(columns=['hours-per-week', 'income']).values  # drop 'hours-per-week' target and income label
y = data['hours-per-week'].values                           # regression target: hours worked per week

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Custom AdaBoost Regressor ---
class AdaBoostRegressorCustom:
    def __init__(self, n_estimators=100, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learners = []
        self.alphas = []
        self.train_errors = []

    def fit(self, X, y):
        N = len(y)
        weights = np.full(N, 1 / N)
        for t in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, y, sample_weight=weights)
            prediction = tree.predict(X)

            abs_error = np.abs(y - prediction)
            max_error = abs_error.max()
            if max_error == 0:
                max_error = 1e-10
            norm_error = abs_error / max_error

            error = np.sum(weights * norm_error)
            error = np.clip(error, 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - error) / error)

            self.learners.append(tree)
            self.alphas.append(alpha)

            weights *= np.exp(alpha * norm_error)
            weights /= np.sum(weights)

            train_pred = self.predict(X)
            train_error = mean_squared_error(y, train_pred)
            self.train_errors.append(train_error)

    def predict(self, X):
        pred_sum = np.zeros(X.shape[0])
        total_alpha = 0
        for tree, alpha in zip(self.learners, self.alphas):
            pred_sum += alpha * tree.predict(X)
            total_alpha += alpha
        return pred_sum / total_alpha if total_alpha > 0 else pred_sum

# --- Custom Gradient Boosting Regressor ---
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
        pred = np.full_like(y, self.init_pred, dtype=float)
        for _ in range(self.n_estimators):
            residual = y - pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residual)
            update = tree.predict(X)
            pred += self.learning_rate * update
            self.learners.append(tree)
            self.train_errors.append(mean_squared_error(y, pred))

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_pred, dtype=float)
        for tree in self.learners:
            pred += self.learning_rate * tree.predict(X)
        return pred

# Train AdaBoost Regressor
ada = AdaBoostRegressorCustom(n_estimators=100, max_depth=3)
ada.fit(X_train, y_train)
ada_preds = ada.predict(X_test)
ada_rmse = mean_squared_error(y_test, ada_preds, squared=False)
ada_r2 = r2_score(y_test, ada_preds)

# Train Gradient Boosting Regressor
gbm = GradientBoostRegressorCustom(n_estimators=100, learning_rate=0.3, max_depth=3)
gbm.fit(X_train, y_train)
gbm_preds = gbm.predict(X_test)
gbm_rmse = mean_squared_error(y_test, gbm_preds, squared=False)
gbm_r2 = r2_score(y_test, gbm_preds)

# Print results
print(f"AdaBoost RMSE: {ada_rmse:.4f}, R2: {ada_r2:.4f}")
print(f"Gradient Boost RMSE: {gbm_rmse:.4f}, R2: {gbm_r2:.4f}")

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

# Paired t-test on squared errors
errors_ada_sq = (y_test - ada_preds) ** 2
errors_gbm_sq = (y_test - gbm_preds) ** 2
t_stat, p_value = ttest_rel(errors_ada_sq, errors_gbm_sq)

print("\nPaired t-test on squared errors:")
print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")

if p_value < 0.05:
    print("→ Significant difference between AdaBoost and Gradient Boost (p < 0.05).")
    if t_stat < 0:
        print("→ AdaBoost had significantly lower squared error.")
    else:
        print("→ Gradient Boost had significantly lower squared error.")
else:
    print("→ No significant difference in squared error between models (p ≥ 0.05).")
