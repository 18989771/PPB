import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                   names=column_names, na_values=" ?", skipinitialspace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Encode categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns.drop('income')
data = pd.get_dummies(data, columns=categorical_cols)

# Encode target variable
label_encoder = LabelEncoder()
data['income'] = label_encoder.fit_transform(data['income'])

# Split features and target
X = data.drop('income', axis=1).values
y = data['income'].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

class GBM:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=1):
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
# Train model
model = GBM(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy on test set:", accuracy)

# Plot training error over iterations
import matplotlib.pyplot as plt

plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o')
plt.title("GBM Training Error Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.grid(True)
plt.show()
