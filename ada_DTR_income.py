import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Load the UCI Adult Income dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Binary classification: 0 = <=50K, 1 = >50K
data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Encode categorical features
categorical_cols = data.select_dtypes(include=['object']).columns
if 'income' in categorical_cols:
    categorical_cols = categorical_cols.drop('income')

for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# Separate features and labels
X = data.drop('income', axis=1).values
y = data['income'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# AdaBoost with Decision Trees
class AdaBoost:
    def __init__(self, n_estimators=10, max_depth=2):
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []
        self.train_errors = []
        self.max_depth = max_depth

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y, sample_weight=w)
            predictions = tree.predict(X)

            error = np.sum(w[predictions != y])
            error = np.clip(error, 1e-10, 1 - 1e-10)  # Avoid log(0)

            alpha = 0.5 * np.log((1 - error) / error)

            w *= np.exp(-alpha * (2 * y - 1) * (2 * predictions - 1))
            w /= np.sum(w)

            self.learners.append(tree)
            self.alphas.append(alpha)

            # Track training error
            train_pred = self.predict(X)
            train_error = 1 - accuracy_score(y, train_pred)
            self.train_errors.append(train_error)

    def predict(self, X):
        learner_preds = np.array([alpha * learner.predict(X) for learner, alpha in zip(self.learners, self.alphas)])
        final_pred = np.sign(np.sum(learner_preds, axis=0))
        return (final_pred > 0).astype(int)


# Train AdaBoost model
model = AdaBoost(n_estimators=100, max_depth=3)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("AdaBoost Accuracy on Adult Income dataset:", accuracy)

# Plot training error
plt.figure(figsize=(8, 5))
plt.plot(range(1, model.n_estimators + 1), model.train_errors, marker='o')
plt.title("AdaBoost Training Error Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Training Error")
plt.grid(True)
plt.show()
