import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset (adjust filename)
data = pd.read_csv('grade.csv',sep=';')

# Filter for binary classification: Dropout=0, Graduate=1
binary_data = data[data['Target'].isin(['Dropout', 'Graduate'])].copy()
binary_data['Target'] = binary_data['Target'].map({'Dropout': 0, 'Graduate': 1})

# Separate features and target
X = binary_data.drop(columns=['Target']).values
y = binary_data['Target'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Gradient Boosting Machine implementation
class GBM:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.base_learners = []
        self.init_prediction = 0
        self.train_errors = []

    def fit(self, X, y):
        N = len(y)
        self.init_prediction = np.mean(y)
        current_preds = np.full(N, self.init_prediction)

        for _ in range(self.n_estimators):
            residuals = y - current_preds

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions = tree.predict(X)

            current_preds += self.learning_rate * predictions
            self.base_learners.append(tree)

            binary_preds = (current_preds >= 0.5).astype(int)
            error = 1 - accuracy_score(y, binary_preds)
            self.train_errors.append(error)

    def predict(self, X):
        preds = np.full(X.shape[0], self.init_prediction)
        for tree in self.base_learners:
            preds += self.learning_rate * tree.predict(X)
        return preds

# Train GBM model
model = GBM(n_estimators=50, learning_rate=0.1, max_depth=1)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy on test set:", accuracy)

# Plot training error over iterations
plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o')
plt.title("GBM Training Error Over Iterations (Grade Dataset)")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.grid(True)
plt.show()
