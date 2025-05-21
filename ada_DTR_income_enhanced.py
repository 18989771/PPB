import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load Adult dataset
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=column_names,
    na_values="?",
    skipinitialspace=True
)

# Drop rows with missing values
data.dropna(inplace=True)

# Encode target variable
data['income'] = data['income'].apply(lambda x: 1 if x == ">50K" else 0)

# Encode categorical variables
categorical_cols = [col for col in data.select_dtypes(include=['object']).columns if col != 'income']
data = pd.get_dummies(data, columns=categorical_cols) #one hot enconding

# Normalize continuous features
scaler = StandardScaler()
numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

X = data.drop('income', axis=1).values
y = data['income'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Manual AdaBoost implementation using decision trees
class AdaBoost:
    def __init__(self, n_estimators=10, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learners = []
        self.alphas = []
        self.train_errors = []

    def fit(self, X, y):
        n_samples = len(y)
        w = np.full(n_samples, 1 / n_samples)

        for t in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
            clf.fit(X, y, sample_weight=w)
            y_pred = clf.predict(X)

            # Compute weighted error
            miss = (y_pred != y).astype(int)
            error = np.dot(w, miss)

            # Avoid division by zero and overly confident learners
            EPS = 1e-10
            if error > 0.5: #skipping worse than random
                continue
            elif error == 0:
                alpha = 1
            else:
                alpha = 0.5 * np.log((1 - error + EPS) / (error + EPS))

            # Update weights
            w *= np.exp(-alpha * (2 * y - 1) * (2 * y_pred - 1))
            w /= w.sum()

            self.learners.append(clf)
            self.alphas.append(alpha)

            # Calculate training error after each iteration
            train_pred = self.predict(X)
            train_error = 1 - accuracy_score(y, train_pred)
            self.train_errors.append(train_error)

    def predict(self, X):
        learner_preds = np.array([alpha * (2 * clf.predict(X) - 1) for clf, alpha in zip(self.learners, self.alphas)])
        result = np.sign(learner_preds.sum(axis=0))
        return (result + 1) // 2  # Convert {-1, 1} to {0, 1}

# Train the model
model = AdaBoost(n_estimators=50, max_depth=1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Manual AdaBoost Accuracy on Adult Income dataset:", accuracy)

# Plot training error
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(model.train_errors) + 1), model.train_errors, marker='o')
plt.title("AdaBoost Training Error Over Iterations")
plt.xlabel("Number of Estimators")
plt.ylabel("Training Error")
plt.grid(True)
plt.show()
