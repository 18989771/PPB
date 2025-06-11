import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load and preprocess Grade dataset
data = pd.read_csv("grade.csv", sep=';')

# Filter binary target: Dropout vs Graduate
binary_data = data[data['Target'].isin(['Dropout', 'Graduate'])].copy()
binary_data['Target'] = binary_data['Target'].map({'Dropout': 0, 'Graduate': 1})

# Encode categorical columns (except 'Target')
for col in binary_data.columns:
    if binary_data[col].dtype == 'object' and col != 'Target':
        binary_data[col] = LabelEncoder().fit_transform(binary_data[col])

# Features and target
X = binary_data.drop('Target', axis=1).values
y = binary_data['Target'].values

# Train-test split (50/50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# AdaBoost implementation with Decision Trees
class AdaBoost:
    def __init__(self, n_estimators=50, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learners = []
        self.alphas = []
        self.train_errors = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.full(n_samples, 1 / n_samples)
        y_mod = 2 * y - 1  # Convert to {-1, 1}
        pred_sum = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_depth=self.max_depth)
            clf.fit(X, y, sample_weight=w)
            pred = clf.predict(X)
            pred_mod = 2 * pred - 1

            error = np.sum(w[pred != y])
            error = np.clip(error, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)
            self.learners.append(clf)

            w *= np.exp(-alpha * y_mod * pred_mod)
            w /= np.sum(w)

            pred_sum += alpha * pred_mod
            y_final = (np.sign(pred_sum) + 1) // 2
            train_error = 1 - accuracy_score(y, y_final)
            self.train_errors.append(train_error)

    def predict(self, X):
        pred_sum = np.zeros(X.shape[0])
        for clf, alpha in zip(self.learners, self.alphas):
            pred = clf.predict(X)
            pred_mod = 2 * pred - 1
            pred_sum += alpha * pred_mod
        return (np.sign(pred_sum) + 1) // 2

# Train AdaBoost on Grade dataset
model = AdaBoost(n_estimators=100, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost Accuracy on Grade dataset: {accuracy:.4f}")
print("Sample predicted labels:", y_pred[:20])
print("Sample true labels:", y_test[:20])

# Plot training error over iterations
plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o')
plt.title("AdaBoost Training Error on Grade Dataset")
plt.xlabel("Iteration")
plt.ylabel("Training Error")
plt.grid(True)
plt.show()
