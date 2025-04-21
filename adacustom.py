import math

# Simple binary classifier that outputs 0 or 1 (0 = correct, 1 = wrong)
class DecisionStump: #classifier with one feature and one threshold
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1 #less than or greater than, used to flip direction of comparison to see which gives smaller weighted error

    def fit(self, X, y, weights):
        n_samples, n_features = len(X), len(X[0])
        min_error = float('inf')

        for feature_i in range(n_features):
            thresholds = set([x[feature_i] for x in X])
            for threshold in thresholds:
                for polarity in [1, -1]:
                    errors = [0] * n_samples
                    for i in range(n_samples):
                        prediction = 1 if polarity * X[i][feature_i] < polarity * threshold else 0 #is prediction above or below threshold
                        errors[i] = 1 if prediction != y[i] else 0
                    weighted_error = sum([weights[i] * errors[i] for i in range(n_samples)])
                    if weighted_error < min_error:
                        min_error = weighted_error
                        self.polarity = polarity
                        self.threshold = threshold
                        self.feature_index = feature_i

    def predict(self, X):
        predictions = []
        for x in X:
            val = x[self.feature_index]
            prediction = 1 if self.polarity * val < self.polarity * self.threshold else 0
            predictions.append(prediction)
        return predictions

# AdaBoost that works with 0 (correct) and 1 (wrong)
class AdaBoost:
    def __init__(self, n_estimators=10): #no of boosting rounds
        self.n_estimators = n_estimators
        self.alphas = [] #wieght given to each classifier
        self.classifiers = [] #No of decision stumps

    def fit(self, X, y):
        n_samples = len(X)
        weights = [1 / n_samples] * n_samples

        for m in range(self.n_estimators): #no of classifiers
            stump = DecisionStump()
            stump.fit(X, y, weights) #train stump
            predictions = stump.predict(X)

            # Indicator: 1 if wrong, 0 if correct
            errors = [1 if predictions[i] != y[i] else 0 for i in range(n_samples)]
            weighted_error = sum([weights[i] * errors[i] for i in range(n_samples)])

            # Avoid division by zero
            if weighted_error == 0:
                weighted_error = 1e-10

            alpha = math.log((1 - weighted_error) / weighted_error)

            # Update weights
            for i in range(n_samples):
                weights[i] *= math.exp(alpha * errors[i])

            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            self.classifiers.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        # Weighted sum of predictions (still 0 or 1)
        final_scores = [0] * len(X)
        for alpha, clf in zip(self.alphas, self.classifiers):
            predictions = clf.predict(X)
            for i in range(len(X)):
                final_scores[i] += alpha * (1 if predictions[i] == 1 else -1)
        return [1 if score > 0 else 0 for score in final_scores]  # if score>0 output is 1, else 0

# Example usage
X = [[1], [2], [3], [4], [5]]  #implies threshold in between 2 and 3 due to where 0 and 1 split
y = [0, 0, 1, 1, 1]

# Train AdaBoost
model = AdaBoost(n_estimators=3)
model.fit(X, y)

# Predict
predictions = model.predict(X)

# Output
print("Input X:", X)
print("True labels:", y)
print("Predictions:", predictions)