import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load grade dataset (make sure 'grade.csv' is in your working directory)
data = pd.read_csv("grade.csv", sep=';')

# Filter for binary classification (Dropout=0, Graduate=1)
binary_data = data[data['Target'].isin(['Dropout', 'Graduate'])].copy()
binary_data['Target'] = binary_data['Target'].map({'Dropout': 0, 'Graduate': 1})

# Encode categorical features (except 'Target')
for col in binary_data.columns:
    if binary_data[col].dtype == 'object' and col != 'Target':
        binary_data[col] = LabelEncoder().fit_transform(binary_data[col])

# Prepare features and labels
X = binary_data.drop('Target', axis=1).values
y = binary_data['Target'].values

# Split dataset into train and test (50% test size as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Regression Stump class (vectorized version)
class Stump:
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
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_mean = residuals[left_mask].mean()
                right_mean = residuals[right_mask].mean()

                preds = np.where(left_mask, left_mean, right_mean)
                error = np.sum((residuals - preds) ** 2)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_value = left_mean
                    self.right_value = right_mean

    def predict(self, X):
        feature_values = X[:, self.feature_index]
        preds = np.where(feature_values < self.threshold, self.left_value, self.right_value)
        return preds

# Gradient Boosting Machine for binary classification
class GBM:
    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learners = []
        self.init_prediction = 0
        self.train_errors = []

    def fit(self, X, y):
        N = len(y)
        self.init_prediction = y.mean()  # Initial prediction (mean of labels)
        current_preds = np.full(N, self.init_prediction)

        for m in range(self.n_estimators):
            residuals = y - current_preds
            stump = Stump()
            stump.fit(X, residuals)
            preds = stump.predict(X)

            current_preds += self.learning_rate * preds
            self.base_learners.append(stump)

            # Convert current predictions to binary class (threshold=0.5)
            binary_preds = (current_preds >= 0.5).astype(int)
            error = 1 - accuracy_score(y, binary_preds)
            self.train_errors.append(error)

    def predict(self, X):
        N = X.shape[0]
        preds = np.full(N, self.init_prediction)
        for stump in self.base_learners:
            preds += self.learning_rate * stump.predict(X)
        return preds

# Train the GBM on the grade dataset
model = GBM(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy on grade dataset test set: {accuracy:.4f}")

# Plot training error over iterations
plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o')
plt.title("GBM Training Error on Grade Dataset")
plt.xlabel("Iteration")
plt.ylabel("Training Error")
plt.grid(True)
plt.show()

#Loads the grade dataset.

#Filters for binary classes (Dropout vs Graduate).

#Label encodes categorical features.

#Implements a regression stump to fit residuals.

#Trains a GBM by iteratively fitting stumps on residuals.

#Tracks and plots training error over iterations.

#Evaluates accuracy on the test set.